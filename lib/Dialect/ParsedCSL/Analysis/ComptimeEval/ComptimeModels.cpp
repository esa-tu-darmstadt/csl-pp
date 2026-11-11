//===- ComptimeModels.cpp - Comptime Evaluation External Models ---------===//
//
// External models for comptime evaluation of ParsedCSL operations.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/BuiltinEvaluators.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/TensorAccessAnalysis.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <atomic>
#include <cstdint>
#include <variant>

using namespace mlir;
using namespace mlir::parsed_csl;
using namespace mlir::csl;

#define DEBUG_TYPE "comptime-eval"

namespace {

template <typename ConcreteT, typename ConcreteOp>
struct ComptimeEvaluableOpInterfaceBaseModel
    : public ComptimeEvaluableOpInterface::ExternalModel<ConcreteT,
                                                         ConcreteOp> {
  // Base model provides default implementations; specific ops override as
  // needed.

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    return ComptimeError::semanticError(op, context)
           << "evaluate() not implemented for this operation";
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false;
  }
};

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

struct ConstantOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ConstantOpComptimeModel,
                                                   ConstantOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto constOp = cast<ConstantOp>(op);
    // Constants are inherently comptime-known - just return their value
    // The type is derived from the value attribute
    return constOp.getValue();
  }
};

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

/// Whether `type` is or contains a comptime_int or comptime_float.
static bool containsComptimeNumber(Type type) {
  if (isa<ComptimeIntType, ComptimeFloatType>(type))
    return true;
  if (auto structType = dyn_cast<StructType>(type))
    return llvm::any_of(structType.getFieldTypes(), containsComptimeNumber);
  if (auto arrayType = dyn_cast<ArrayType>(type))
    return containsComptimeNumber(arrayType.getElementType());
  return false;
}

struct VariableOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<VariableOpComptimeModel,
                                                   VariableOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto varOp = cast<VariableOp>(op);

    // ========== PHASE 1: Gather operand info ==========
    Type varType;
    Type initType;

    if (varOp.getDynamicType()) {
      auto typeAttrResult =
          context.getOrLoadComptimeRValue(varOp.getDynamicType());
      if (failed(typeAttrResult)) {
        return ComptimeError::semanticError(op, context)
               << "variable type is not comptime-known";
      }
      if (!mlir::isa<parsed_csl::TypeAttr>(*typeAttrResult)) {
        return ComptimeError::semanticError(op, context)
               << "specified variable type is not a type";
      }
      varType = mlir::cast<parsed_csl::TypeAttr>(*typeAttrResult).getValue();
    }

    if (varOp.getInitializer()) {
      FailureOr<Type> maybeInitType =
          context.getInferredRValueType(varOp.getInitializer());
      if (failed(maybeInitType)) {
        return ComptimeError::semanticError(op, context)
               << "cannot infer type from initializer";
      }
      initType = *maybeInitType;
    }

    // ========== PHASE 2: Validation ==========
    if (varOp.getIsConst() && !varOp.getInitializer()) {
      return ComptimeError::semanticError(op, context)
             << "const variable must have an initializer";
    }

    bool isComptimeVar = varOp.getIsComptime() || context.isComptimeContext();
    if (isComptimeVar && !varOp.getInitializer()) {
      return ComptimeError::semanticError(op, context)
             << "comptime variable must have an initializer";
    }

    if (!varType && !initType) {
      return ComptimeError::semanticError(op, context)
             << "variable must have either a type or an initializer";
    }

    // If both dynamicType and initializer exist, check implicit cast is allowed
    if (varType && initType) {
      bool comptimeKnown =
          context.isRValueComptimeKnown(varOp.getInitializer());
      if (!csl::isImplicitCastAllowed(initType, varType, comptimeKnown)) {
        return ComptimeError::semanticError(op, context)
               << "implicit cast from " << prettyPrintType(initType) << " to "
               << prettyPrintType(varType) << " is not allowed";
      }
    }

    // Modules can only be stored in local variables
    Type valueType = varType ? varType : initType;
    if (mlir::isa<ImportedModuleType>(valueType)) {
      return ComptimeError::semanticError(op, context)
             << "modules can only be stored in global variables";
    }

    // ========== PHASE 3: Type Inference ==========
    bool isConst = varOp.getIsConst();
    Type resultType = LValueType::get(valueType, isConst);

    // ========== PHASE 4: Comptime Evaluation ==========
    auto runtimeVariableError = [&]() {
      ComptimeError error = ComptimeError::semanticError(op, context);
      error << "runtime variable '" << varOp.getName()
            << "' cannot have type '" << prettyPrintType(valueType)
            << "', which only exists at compile time";
      // A comptime number gets a runtime type from an explicit type.
      if (containsComptimeNumber(valueType))
        error << "; declare the variable with an explicit runtime type";
      return error;
    };
    if (!isComptimeVar && !isConst) {
      if (!csl::isRuntimeType(valueType))
        return runtimeVariableError();
      // Mutable non-comptime variables are not comptime-known lvalues
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "not a comptime-known variable";
    }

    // Comptime/const variables must have initializers (validated above)
    assert(varOp.getInitializer());

    // Evaluate the initializer (auto-load lvalues)
    FailureOr<TypedAttr> initValue =
        context.getOrLoadComptimeRValue(varOp.getInitializer());
    if (failed(initValue) && !isComptimeVar &&
        !csl::isRuntimeType(valueType) && !isa<FunctionType>(valueType))
      return runtimeVariableError();
    if (failed(initValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(varOp.getInitializer())
             << "initializer is not comptime-known";
    }

    // Declare the variable as comptime in this context
    context.declareComptimeVariable(varOp.getOperation());

    TypedAttr typedInitValue = dyn_cast<TypedAttr>(*initValue);
    assert(typedInitValue);

    // If there's a dynamic type, cast the initializer to that type
    if (varType) {
      InFlightDiagnostic diag = emitRemark(op->getLoc());
      auto castResult =
          csl::evaluateImplicitCast(typedInitValue, varType, &diag);
      if (failed(castResult)) {
        return ComptimeError::executionError(op, context).withType(resultType)
               << "failed to cast initializer to variable type"
               << std::move(diag);
      }
      diag.abandon();
      typedInitValue = *castResult;
    }

    // Create an LValueDescriptor for the variable and store the initial value
    LValueDescriptor varDescriptor(varOp, {}, valueType);
    if (failed(context.storeLValue(varDescriptor, typedInitValue))) {
      return ComptimeError::semanticError(op, context)
             << "failed to store comptime variable value";
    }

    // Return an LValueDescriptor pointing to the VariableOp
    return LValueDescriptor(varOp, {}, valueType);
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    VariableOp varOp = cast<VariableOp>(op);
    return varOp.getIsComptime();
  }
};

//===----------------------------------------------------------------------===//
// ReadOp
//===----------------------------------------------------------------------===//

/// The name of what `source` refers to, for diagnostics: a variable, a
/// parameter, a loop capture or a global. Empty if it has no name.
static std::string getReadSourceName(Value source) {
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(source)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    if (auto forOp = mlir::dyn_cast_if_present<ForOp>(owner)) {
      if (blockArg.getArgNumber() == 0)
        return forOp.getElemName().str();
      if (std::optional<StringRef> indexName = forOp.getIndexName())
        return indexName->str();
      return "";
    }
    if (auto funcOp = mlir::dyn_cast_if_present<FuncOp>(owner))
      if (std::optional<ArrayAttr> names = funcOp.getParamNames();
          names && blockArg.getArgNumber() < names->size())
        if (auto name =
                mlir::dyn_cast<StringAttr>((*names)[blockArg.getArgNumber()]))
          return name.str();
    return "";
  }
  Operation *def = source.getDefiningOp();
  if (auto varOp = mlir::dyn_cast<VariableOp>(def))
    return varOp.getName().str();
  if (auto symbolOp = mlir::dyn_cast<GetSymbolOp>(def))
    return symbolOp.getName().str();
  if (auto locOp = mlir::dyn_cast<LocOp>(def))
    return getReadSourceName(locOp.getValue());
  return "";
}

struct ReadOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ReadOpComptimeModel,
                                                   ReadOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    Value source = cast<ReadOp>(op).getSource();

    // ========== PHASE 1 & 2: Validation and Type Inference ==========
    FailureOr<Type> resultType = context.getInferredRValueType(source);
    if (failed(resultType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of the read value";
    }

    // ========== PHASE 3: Comptime Evaluation ==========
    // A location is read now, which is where the source reads it; anything
    // else reads as itself.
    FailureOr<TypedAttr> value = context.getOrLoadComptimeRValue(source);
    if (failed(value)) {
      std::string name = getReadSourceName(source);
      ComptimeError error =
          ComptimeError::comptimeFailure(op, context, *resultType)
              .withDependency(source);
      if (name.empty())
        error << "value is not comptime-known";
      else
        error << "'" << name << "' is not comptime-known";
      return error;
    }
    return *value;
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Reading a comptime variable happens at comptime.
    FailureOr<LValueDescriptor> descriptor =
        context.getComptimeLValue(cast<ReadOp>(op).getSource());
    return succeeded(descriptor) && context.isLValueComptime(*descriptor);
  }
};

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

struct LoadOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<LoadOpComptimeModel,
                                                   LoadOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    Value lvalue = cast<LoadOp>(op).getLvalue();

    // ========== PHASE 1 & 2: Validation and Type Inference ==========
    FailureOr<Type> lvalueType = context.getInferredType(lvalue);
    if (failed(lvalueType))
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of the loaded location";
    auto locationType = dyn_cast<LValueType>(*lvalueType);
    if (!locationType)
      return ComptimeError::semanticError(op, context)
             << "loaded value is not a location";
    Type resultType = locationType.getValueType();

    // ========== PHASE 3: Comptime Evaluation ==========
    FailureOr<LValueDescriptor> descriptor = context.getComptimeLValue(lvalue);
    if (failed(descriptor))
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(lvalue)
             << "location is not comptime-known";
    FailureOr<TypedAttr> value = context.loadLValue(*descriptor);
    if (failed(value))
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(lvalue)
             << "failed to load comptime location";
    return *value;
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    FailureOr<LValueDescriptor> descriptor =
        context.getComptimeLValue(cast<LoadOp>(op).getLvalue());
    return succeeded(descriptor) && context.isLValueComptime(*descriptor);
  }
};

//===----------------------------------------------------------------------===//
// SyntaxErrorOp
//===----------------------------------------------------------------------===//

struct SyntaxErrorOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<SyntaxErrorOpComptimeModel,
                                                   SyntaxErrorOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    return ComptimeError::semanticError(op, context)
           << "unrecoverable parse error";
  }
};

//===----------------------------------------------------------------------===//
// DeferredErrorOp
//===----------------------------------------------------------------------===//

struct DeferredErrorOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<DeferredErrorOpComptimeModel,
                                                   DeferredErrorOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto errorOp = cast<DeferredErrorOp>(op);
    return ComptimeError::semanticError(op, context) << errorOp.getMessage();
  }
};

//===----------------------------------------------------------------------===//
// LocOp
//===----------------------------------------------------------------------===//

struct LocOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<LocOpComptimeModel, LocOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto locOp = cast<LocOp>(op);
    Value operand = locOp.getValue();

    // ========== PHASE 1 & 2: Validation and Type Inference ==========
    auto inferredType = context.getInferredType(operand);
    if (failed(inferredType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of operand";
    }

    // ========== PHASE 3: Comptime Evaluation ==========
    // Transparently forward lvalue or rvalue from the operand.
    auto lvalue = context.getComptimeLValue(operand);
    if (succeeded(lvalue))
      return *lvalue;

    auto rvalue = context.getComptimeRValue(operand);
    if (succeeded(rvalue))
      return *rvalue;

    return ComptimeError::comptimeFailure(op, context, *inferredType)
               .withDependency(operand)
           << "operand is not comptime-known";
  }
};

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

struct AssignOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<AssignOpComptimeModel,
                                                   AssignOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto assignOp = cast<AssignOp>(op);
    Value lvalue = assignOp.getLvalue();

    // ========== PHASE 1 & 2: Validation ==========
    auto varTypeResult = context.getInferredType(lvalue);
    if (failed(varTypeResult)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of variable being assigned to";
    }
    auto varLValueType = llvm::dyn_cast<LValueType>(*varTypeResult);
    if (!varLValueType) {
      return ComptimeError::semanticError(op, context)
             << "cannot assign to non-lvalue of type "
             << prettyPrintType(*varTypeResult);
    }

    if (varLValueType.getIsConst()) {
      return ComptimeError::semanticError(op, context)
             << "cannot assign to const lvalue";
    }

    auto valueType = context.getInferredRValueType(assignOp.getValue());
    if (failed(valueType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of assigned value";
    }

    bool comptimeKnown = context.isRValueComptimeKnown(assignOp.getValue());
    if (!csl::isImplicitCastAllowed(*valueType, varLValueType.getValueType(),
                                    comptimeKnown)) {
      return ComptimeError::semanticError(op, context)
             << "implicit cast from " << prettyPrintType(*valueType) << " to "
             << prettyPrintType(varLValueType.getValueType())
             << " is not allowed";
    }

    // ========== PHASE 3: Comptime Evaluation ==========
    // (No result type needed - AssignOp has no results)

    auto lvalueDescriptor = context.getComptimeLValue(lvalue);
    if (failed(lvalueDescriptor)) {
      return ComptimeError::comptimeFailure(op, context).withDependency(lvalue)
             << "lvalue is not comptime-known";
    }

    if (!context.isLValueComptime(*lvalueDescriptor)) {
      return ComptimeError::comptimeFailure(op, context)
             << "lvalue is not comptime-evaluable";
    }

    FailureOr<TypedAttr> rhsAttr =
        context.getOrLoadComptimeRValue(assignOp.getValue());
    if (failed(rhsAttr)) {
      return ComptimeError::comptimeFailure(op, context)
                 .withDependency(assignOp.getValue())
             << "assigned value is not comptime-known";
    }
    TypedAttr castedRhsAttr = *rhsAttr;

    if (rhsAttr->getType() != varLValueType.getValueType()) {
      auto castResult =
          csl::evaluateImplicitCast(*rhsAttr, varLValueType.getValueType());
      assert(succeeded(castResult));
      castedRhsAttr = *castResult;
    }

    if (failed(context.storeLValue(*lvalueDescriptor, castedRhsAttr))) {
      return ComptimeError::semanticError(op, context)
             << "failed to assign to comptime lvalue";
    }

    return success();
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    auto assignOp = cast<AssignOp>(op);
    auto lvalueDescriptor = context.getComptimeLValue(assignOp.getLvalue());
    if (failed(lvalueDescriptor)) {
      return false;
    }
    return context.isLValueComptime(*lvalueDescriptor);
  }
};

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

struct AddressOfOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<AddressOfOpComptimeModel,
                                                   AddressOfOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto addressOfOp = cast<AddressOfOp>(op);

    // ========== PHASE 1 & 2: Validation and Type Inference ==========
    FailureOr<Type> operandType =
        context.getInferredType(addressOfOp.getOperand());
    if (failed(operandType)) {
      return ComptimeError::semanticError(op, context)
             << "operand type not inferred";
    }

    Type resultType;
    if (auto lvalueType = dyn_cast<LValueType>(*operandType)) {
      // Only values in memory have an address. A function value stands for
      // the function, whose address is a function pointer.
      Type valueType = lvalueType.getValueType();
      if (!isa<FunctionType>(valueType) && !csl::isRuntimeType(valueType)) {
        ComptimeError error = ComptimeError::semanticError(op, context);
        error << "cannot take the address of a value of type '"
              << prettyPrintType(valueType)
              << "', which only exists at compile time";
        if (isa<ComptimeIntType>(valueType))
          error << "; to take its address, give it a runtime integer type";
        else if (isa<ComptimeFloatType>(valueType))
          error << "; to take its address, give it a runtime float type";
        return error;
      }
      resultType =
          PointerType::get(valueType, lvalueType.getIsConst(),
                           /*isArray=*/false);
    } else if (auto funcType = dyn_cast<FunctionType>(*operandType)) {
      resultType = PointerType::get(funcType, /*isConst=*/true,
                                    /*isArray=*/false);
    } else {
      return ComptimeError::semanticError(op, context)
             << "operand must be an lvalue or function type, got: "
             << csl::prettyPrintType(*operandType);
    }

    // ========== PHASE 3: Comptime Evaluation ==========
    if (isa<FunctionType>(*operandType)) {
      FailureOr<TypedAttr> operandValue =
          context.getOrLoadComptimeRValue(addressOfOp.getOperand());
      if (failed(operandValue)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "function operand is not comptime-known";
      }

      auto funcRef = dyn_cast<csl::FuncRefAttr>(*operandValue);
      if (!funcRef) {
        return ComptimeError::semanticError(op, context)
               << "operand is expected to be a function reference";
      }

      // The referenced function is instantiated once the address reaches
      // the runtime program (ComptimeEvaluator::instantiateRuntimeReferences).
      return (TypedAttr)FuncAddrAttr::get(funcRef);
    } else if (FailureOr<LValueDescriptor> lvalue =
                   context.getComptimeLValue(addressOfOp.getOperand());
               succeeded(lvalue) && lvalue->isGlobalBased() &&
               lvalue->isDirectReference()) {
      GlobalOp globalOp = lvalue->getBaseGlobal();
      std::optional<uint64_t> moduleId = std::nullopt;
      if (auto module = globalOp->getParentOfType<parsed_csl::ModuleOp>()) {
        moduleId = module.getModuleIdentifier();
      }
      auto lvalueType = llvm::dyn_cast_if_present<LValueType>(*operandType);
      return (TypedAttr)GlobalAddrAttr::get(GlobalRefAttr::get(
          op->getContext(), moduleId,
          FlatSymbolRefAttr::get(globalOp.getSymNameAttr()), lvalueType));
    }

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "address-of produces runtime pointer (not comptime-evaluable)";
  }
};

//===----------------------------------------------------------------------===//
// DereferenceOp
//===----------------------------------------------------------------------===//

struct DereferenceOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<DereferenceOpComptimeModel,
                                                   DereferenceOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto derefOp = cast<DereferenceOp>(op);

    // ========== PHASE 1 & 2: Validation and Type Inference ==========
    FailureOr<Type> operandType =
        context.getInferredRValueType(derefOp.getOperand());
    if (failed(operandType)) {
      return ComptimeError::semanticError(op, context)
             << "operand type not inferred";
    }

    auto ptrType = dyn_cast<PointerType>(*operandType);
    if (!ptrType) {
      return ComptimeError::semanticError(op, context)
             << "dereference operand must be a pointer type, got: "
             << csl::prettyPrintType(*operandType);
    }

    // Result type: function pointers dereference to FunctionType directly;
    // all other pointers dereference to an LValueType.
    Type pointeeType = ptrType.getPointeeType();
    Type resultType;
    if (isa<FunctionType>(pointeeType)) {
      resultType = pointeeType;
    } else {
      resultType = LValueType::get(pointeeType, ptrType.getIsConst());
    }

    // ========== PHASE 3: Comptime Evaluation ==========
    FailureOr<TypedAttr> operandValue =
        context.getOrLoadComptimeRValue(derefOp.getOperand());
    if (failed(operandValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "pointer dereference cannot be evaluated at comptime";
    }

    // Dereference a GlobalAddrAttr: return an LValueDescriptor for the global.
    if (auto globalAddr = dyn_cast<GlobalAddrAttr>(*operandValue)) {
      GlobalRefAttr globalRef = globalAddr.getGlobalRef();
      auto cu = op->getParentOfType<csl::CompilationUnitOp>();
      FailureOr<GlobalOp> globalOp = lookupGlobalByRefAttr(globalRef, cu);
      if (failed(globalOp)) {
        return ComptimeError::semanticError(op, context)
               << "global '" << globalRef.getSymbolName().getValue()
               << "' not found";
      }
      return LValueDescriptor(*globalOp, {}, pointeeType);
    }

    // Dereference a FuncAddrAttr: return the FuncRefAttr as an rvalue.
    if (auto funcAddr = dyn_cast<FuncAddrAttr>(*operandValue)) {
      return (TypedAttr)funcAddr.getFuncRef();
    }

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "pointer dereference cannot be evaluated at comptime";
  }
};

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

struct BinaryOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<BinaryOpComptimeModel,
                                                   BinaryOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto binaryOp = cast<BinaryOp>(op);

    // ========== PHASE 1: Gather operand types ==========
    auto lhsType = context.getInferredRValueType(binaryOp.getLhs());
    if (failed(lhsType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of left operand";
    }

    auto rhsType = context.getInferredRValueType(binaryOp.getRhs());
    if (failed(rhsType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of right operand";
    }

    // ========== PHASE 2: Build TypeOrAttr with values when available
    // ==========
    csl::TypeOrAttr lhsOp(*lhsType);
    csl::TypeOrAttr rhsOp(*rhsType);

    auto lhsValue = context.getOrLoadComptimeRValue(binaryOp.getLhs());
    if (succeeded(lhsValue)) {
      if (auto typed = dyn_cast<TypedAttr>(*lhsValue))
        lhsOp = typed;
    }

    auto rhsValue = context.getOrLoadComptimeRValue(binaryOp.getRhs());
    if (succeeded(rhsValue)) {
      if (auto typed = dyn_cast<TypedAttr>(*rhsValue))
        rhsOp = typed;
    }

    // ========== PHASE 3: Infer types and evaluate ==========
    ComptimeError error = ComptimeError::semanticError(op, context);
    auto result =
        csl::inferBinaryOp(binaryOp.getKind(), lhsOp, rhsOp, error.getDiag());
    if (failed(result)) {
      return error;
    }

    Type resultType = csl::getType(result->result);

    // If fully evaluated at comptime, return the value.
    if (csl::isComptimeKnown(result->result))
      return std::get<TypedAttr>(result->result);

    // Not fully evaluated — report dependency failures.
    if (failed(lhsValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(binaryOp.getLhs())
             << "left operand is not comptime-known";
    }
    if (failed(rhsValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(binaryOp.getRhs())
             << "right operand is not comptime-known";
    }

    return ComptimeError::semanticError(op, context)
           << "operation cannot be fully evaluated at comptime, but operands "
              "are comptime-known (missing evaluation logic?)";
  }
};

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

struct UnaryOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<UnaryOpComptimeModel,
                                                   UnaryOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto unaryOp = cast<UnaryOp>(op);

    // ========== PHASE 1 & 2: Validation and Type Inference ==========
    auto operandType = context.getInferredRValueType(unaryOp.getOperand());
    if (failed(operandType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of operand";
    }

    InFlightDiagnostic diag = emitRemark(op->getLoc());
    auto resultTypeOrFailure =
        csl::inferUnaryResultType(unaryOp.getKind(), *operandType, &diag);
    if (failed(resultTypeOrFailure)) {
      return ComptimeError::semanticError(op, context)
             << "incompatible type for unary operation" << std::move(diag);
    }
    diag.abandon();

    Type resultType = *resultTypeOrFailure;

    // ========== PHASE 3: Comptime Evaluation ==========
    auto operandValue = context.getOrLoadComptimeRValue(unaryOp.getOperand());
    if (failed(operandValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(unaryOp.getOperand())
             << "operand is not comptime-known";
    }

    auto operandTyped = dyn_cast<TypedAttr>(*operandValue);
    if (!operandTyped) {
      return ComptimeError::semanticError(op, context)
                 .withDependency(unaryOp.getOperand())
             << "operand is not a typed attribute";
    }

    InFlightDiagnostic evalDiag = emitRemark(op->getLoc());
    auto result =
        csl::evaluateUnary(unaryOp.getKind(), operandTyped, &evalDiag);
    if (failed(result)) {
      return ComptimeError::executionError(op, context)
                 .withType(resultType)
                 .withDependency(unaryOp.getOperand())
             << "failed to evaluate unary operation during comptime"
             << std::move(evalDiag);
    }
    evalDiag.abandon();

    return *result;
  }
};

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

struct CastOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<CastOpComptimeModel,
                                                   CastOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto castOp = cast<CastOp>(op);

    // ========== PHASE 1: Gather operand info ==========
    auto sourceType = context.getInferredRValueType(castOp.getSource());
    if (failed(sourceType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of source operand";
    }

    auto targetTypeValue =
        context.getOrLoadComptimeRValue(castOp.getTargetType());
    if (failed(targetTypeValue)) {
      return ComptimeError::semanticError(op, context)
             << "target type is not comptime-known";
    }
    auto typeAttr = dyn_cast<csl::TypeAttr>(*targetTypeValue);
    if (!typeAttr) {
      return ComptimeError::semanticError(op, context)
             << "target type operand is not a type attribute";
    }
    Type targetType = typeAttr.getValue();

    // ========== PHASE 2: Validation ==========
    if (castOp.getImplicit()) {
      bool comptimeKnown = context.isRValueComptimeKnown(castOp.getSource());
      if (!csl::isImplicitCastAllowed(*sourceType, targetType, comptimeKnown)) {
        return ComptimeError::semanticError(op, context)
               << "implicit cast from " << prettyPrintType(*sourceType)
               << " to " << prettyPrintType(targetType) << " is not allowed";
      }
    } else {
      if (!csl::areCastCompatible(*sourceType, targetType)) {
        return ComptimeError::semanticError(op, context)
               << "explicit cast from " << *sourceType << " to " << targetType
               << " is not allowed";
      }
    }

    // ========== PHASE 3: Type Inference ==========
    // Result type is the target type
    Type resultType = targetType;

    // ========== PHASE 4: Comptime Evaluation ==========
    auto sourceValue = context.getOrLoadComptimeRValue(castOp.getSource());
    if (failed(sourceValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(castOp.getSource())
             << "source value is not comptime-known";
    }

    auto sourceTyped = dyn_cast<TypedAttr>(*sourceValue);
    if (!sourceTyped) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "source value is not a typed attribute";
    }

    InFlightDiagnostic diag = emitRemark(op->getLoc());
    auto result = csl::evaluateCast(sourceTyped, targetType, &diag);
    if (failed(result)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(castOp.getSource())
             << "cannot cast value to target type" << std::move(diag);
    }
    diag.abandon();

    return *result;
  }
};

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

struct SelectOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<SelectOpComptimeModel,
                                                   SelectOp> {

  /// Extract the rvalue type from a ComptimeValueOrType, stripping LValueType
  /// if present (matching the behavior of getInferredRValueType).
  static Type getRValueType(const ComptimeValueOrType &vt) {
    Type type = vt.getType();
    if (auto lvalTy = mlir::dyn_cast<csl::LValueType>(type))
      return lvalTy.getValueType();
    return type;
  }

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto selectOp = cast<SelectOp>(op);

    // ========== PHASE 1: Validate condition ==========
    auto condType = context.getInferredRValueType(selectOp.getCondition());
    if (failed(condType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer condition type";
    }
    auto boolType = IntegerType::get(op->getContext(), 1);
    bool comptimeKnown = context.isRValueComptimeKnown(selectOp.getCondition());
    if (!csl::isImplicitCastAllowed(*condType, boolType, comptimeKnown)) {
      return ComptimeError::semanticError(op, context)
             << "condition type " << *condType
             << " cannot be implicitly cast to boolean";
    }

    // ========== PHASE 2: Determine branch evaluation ==========
    auto condValue = context.getOrLoadComptimeRValue(selectOp.getCondition());
    bool conditionIsComptimeTrue = false;
    bool conditionIsComptimeFalse = false;

    if (succeeded(condValue)) {
      auto condTyped = dyn_cast<TypedAttr>(*condValue);
      if (condTyped) {
        auto condBoolResult = csl::evaluateCast(condTyped, boolType);
        if (succeeded(condBoolResult)) {
          auto boolAttr = cast<IntegerAttr>(*condBoolResult);
          conditionIsComptimeTrue = boolAttr.getValue().getBoolValue();
          conditionIsComptimeFalse = !conditionIsComptimeTrue;
        }
      }
    }

    bool isRuntimeConditional = failed(condValue);
    bool thenRegionEvaluated = false;
    bool elseRegionEvaluated = false;

    // ========== PHASE 3: Evaluate branches ==========
    // Store outcomes so Phase 4/5 can access yield information via
    // ControlFlowOutcomes::mayYield instead of querying yield SSA values.
    std::optional<ControlFlowOutcomes> thenOutcome;
    std::optional<ControlFlowOutcomes> elseOutcome;

    // If NOT comptime false → evaluate then_region
    if (!conditionIsComptimeFalse) {
      Region &thenRegion = selectOp.getThenRegion();
      thenOutcome =
          context.evaluateRegion(thenRegion, isRuntimeConditional, false);
      if (thenOutcome->hasFatalError())
        return thenOutcome->getFatalError();
      thenRegionEvaluated = true;
    }

    // If NOT comptime true → evaluate else_region (must exist!)
    if (!conditionIsComptimeTrue) {
      if (!selectOp.hasElseBranch()) {
        // Need else branch but don't have one
        if (conditionIsComptimeFalse) {
          return ComptimeError::semanticError(op, context)
                 << "if-expression without else requires condition to be true, "
                    "but condition evaluates to false at comptime";
        }
        // Runtime condition without else
        return ComptimeError::semanticError(op, context)
               << "if-expression without else requires a comptime-known "
                  "condition, but condition is not comptime-known";
      }

      Region &elseRegion = selectOp.getElseRegion();
      elseOutcome =
          context.evaluateRegion(elseRegion, isRuntimeConditional, false);
      if (elseOutcome->hasFatalError())
        return elseOutcome->getFatalError();
      elseRegionEvaluated = true;
    }

    // ========== PHASE 4: Type inference ==========
    // Use mayYield from evaluation outcomes instead of querying the yield
    // op's SSA value directly.
    Type resultType;

    if (thenRegionEvaluated && elseRegionEvaluated) {
      // Both branches evaluated - check type compatibility and compute common
      // type
      if (thenOutcome->mayYield.empty()) {
        return ComptimeError::semanticError(op, context)
               << "then region must yield a value";
      }
      if (elseOutcome->mayYield.empty()) {
        return ComptimeError::semanticError(op, context)
               << "else region must yield a value";
      }
      Type trueType = getRValueType(thenOutcome->mayYield[0]);
      Type falseType = getRValueType(elseOutcome->mayYield[0]);

      auto commonType = csl::getCommonType(trueType, falseType);
      if (failed(commonType)) {
        return ComptimeError::semanticError(op, context)
               << "incompatible types for select operation: "
               << csl::prettyPrintType(trueType) << " and "
               << csl::prettyPrintType(falseType);
      }
      resultType = *commonType;
    } else if (thenRegionEvaluated) {
      // Only then branch evaluated (condition is comptime true)
      if (thenOutcome->mayYield.empty()) {
        return ComptimeError::semanticError(op, context)
               << "then region must yield a value";
      }
      resultType = getRValueType(thenOutcome->mayYield[0]);
    } else {
      // Only else branch evaluated (condition is comptime false)
      if (elseOutcome->mayYield.empty()) {
        return ComptimeError::semanticError(op, context)
               << "else region must yield a value";
      }
      resultType = getRValueType(elseOutcome->mayYield[0]);
    }

    // ========== PHASE 5: Comptime evaluation ==========
    if (failed(condValue)) {
      // Condition not comptime-known - return type with comptime failure
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(selectOp.getCondition())
             << "select condition is not comptime-known";
    }

    auto condTyped = dyn_cast<TypedAttr>(*condValue);
    if (!condTyped) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "condition is not a typed attribute";
    }

    // Cast condition to bool (should succeed since we validated above)
    InFlightDiagnostic diag = emitRemark(op->getLoc());
    auto condBoolResult = csl::evaluateCast(condTyped, boolType, &diag);
    if (failed(condBoolResult)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(selectOp.getCondition())
             << "cannot cast condition to boolean" << std::move(diag);
    }
    diag.abandon();

    // Get the boolean value and select the branch outcome
    auto boolAttr = cast<IntegerAttr>(*condBoolResult);
    bool conditionValue = boolAttr.getValue().getBoolValue();

    // Get the yielded value from the selected branch's evaluation outcome
    ControlFlowOutcomes &selectedOutcome =
        conditionValue ? *thenOutcome : *elseOutcome;
    if (selectedOutcome.mayYield.empty()) {
      return ComptimeError::semanticError(op, context)
             << "selected branch must yield a value";
    }
    auto yieldedValue = selectedOutcome.mayYield[0].getValue();
    if (!yieldedValue) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << (conditionValue ? "then" : "else")
             << " branch value is not comptime-known";
    }

    return *yieldedValue;
  }
};

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

struct SwitchOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<SwitchOpComptimeModel,
                                                   SwitchOp> {

  // Helper to extract integer value from either IntegerAttr or EnumValueAttr
  static std::optional<int64_t> getIntegerValue(TypedAttr attr) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      return intAttr.getValue().getSExtValue();
    }
    if (auto enumAttr = mlir::dyn_cast<csl::EnumValueAttr>(attr)) {
      return enumAttr.getIntValue();
    }
    return std::nullopt;
  }

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto switchOp = cast<SwitchOp>(op);

    // ========== PHASE 1: Validate input type ==========
    auto inputType = context.getInferredRValueType(switchOp.getInput());
    if (failed(inputType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer switch input type";
    }

    // Must be fixed-width integer or enum (not comptime_int)
    bool isValidInputType = (mlir::isa<IntegerType>(*inputType) &&
                             !mlir::isa<csl::ComptimeIntType>(*inputType)) ||
                            mlir::isa<csl::EnumType>(*inputType);
    if (!isValidInputType) {
      return ComptimeError::semanticError(op, context)
             << "switch input must be fixed-width integer or enum, got "
             << csl::prettyPrintType(*inputType);
    }

    // ========== PHASE 2: Validate case values ==========
    // Resolve all case values and validate uniqueness
    SmallVector<SmallVector<TypedAttr>> resolvedCaseValues;
    DenseSet<int64_t> seenValues;
    auto caseOperands = switchOp.getCaseOperands();
    auto segments = switchOp.getCaseOperandSegments();

    unsigned offset = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
      SmallVector<TypedAttr> thisCaseValues;
      for (int32_t j = 0; j < segments[i]; ++j) {
        Value caseOperand = caseOperands[offset + j];

        auto caseValue = context.getOrLoadComptimeRValue(caseOperand);
        if (failed(caseValue)) {
          return ComptimeError::semanticError(op, context)
                 << "case value must be comptime-known";
        }

        auto typedCaseValue = mlir::dyn_cast<TypedAttr>(*caseValue);
        if (!typedCaseValue) {
          return ComptimeError::semanticError(op, context)
                 << "case value must be a typed value";
        }

        // Cast to input type
        InFlightDiagnostic diag = emitRemark(op->getLoc());
        auto castResult = csl::evaluateCast(typedCaseValue, *inputType, &diag);
        if (failed(castResult)) {
          return ComptimeError::semanticError(op, context)
                 << "case value cannot be cast to input type "
                 << csl::prettyPrintType(*inputType) << std::move(diag);
        }
        diag.abandon();

        // Get integer value for uniqueness check (works for both int and enum)
        auto caseIntValue = getIntegerValue(*castResult);
        if (!caseIntValue) {
          return ComptimeError::semanticError(op, context)
                 << "case value must be an integer or enum value";
        }

        // Check uniqueness
        if (!seenValues.insert(*caseIntValue).second) {
          return ComptimeError::semanticError(op, context)
                 << "duplicate case value in switch";
        }

        thisCaseValues.push_back(*castResult);
      }
      resolvedCaseValues.push_back(std::move(thisCaseValues));
      offset += segments[i];
    }

    // ========== PHASE 2b: Check exhaustiveness ==========
    // Switch must either have an else branch or cover all possible values
    if (!switchOp.hasElseBranch()) {
      uint64_t numCasesCovered = seenValues.size();
      uint64_t numPossibleValues = 0;

      if (auto enumType = mlir::dyn_cast<csl::EnumType>(*inputType)) {
        numPossibleValues = enumType.getNumElements();
      } else if (auto intType = mlir::dyn_cast<IntegerType>(*inputType)) {
        unsigned bitWidth = intType.getWidth();
        assert(bitWidth < 64 && "64-bit integers not supported in CSL");
        numPossibleValues = 1ULL << bitWidth;
      }

      if (numCasesCovered != numPossibleValues) {
        return ComptimeError::semanticError(op, context)
               << "switch must cover all values or have an else branch";
      }
    }

    // ========== PHASE 3: Determine branch evaluation ==========
    auto inputValue = context.getOrLoadComptimeRValue(switchOp.getInput());
    std::optional<size_t> matchingCaseIdx;
    bool inputIsComptimeKnown = false;

    if (succeeded(inputValue)) {
      auto inputTyped = mlir::dyn_cast<TypedAttr>(*inputValue);
      if (inputTyped) {
        auto inputIntValue = getIntegerValue(inputTyped);
        if (inputIntValue) {
          inputIsComptimeKnown = true;
          // Find matching case
          for (size_t i = 0; i < resolvedCaseValues.size(); ++i) {
            for (auto caseVal : resolvedCaseValues[i]) {
              auto caseIntValue = getIntegerValue(caseVal);
              if (caseIntValue && *caseIntValue == *inputIntValue) {
                matchingCaseIdx = i;
                break;
              }
            }
            if (matchingCaseIdx)
              break;
          }
        }
      }
    }

    bool isRuntimeSwitch = !inputIsComptimeKnown;
    SmallVector<Type> evaluatedBranchTypes;
    std::optional<ControlFlowOutcomes> comptimeBranchOutcome;
    std::optional<ControlFlowOutcomes> joinedOutcome;

    // Helper to process a comptime branch outcome (only one branch evaluated).
    // Returns result to propagate, or nullopt to continue.
    auto processComptimeBranchOutcome =
        [&](ControlFlowOutcomes outcome) -> std::optional<ComptimeEvalResult> {
      if (outcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                                   op->getContext());

      if (outcome.hasYields()) {
        // Branch yielded a value - collect type, store outcome
        Type branchType = outcome.mayYield[0].getType();
        if (branchType)
          evaluatedBranchTypes.push_back(branchType);
        comptimeBranchOutcome = std::move(outcome);
        return std::nullopt;
      }

      if (outcome.isFallthroughOnly()) {
        // Switch branch must yield or use control flow
        return ComptimeError::semanticError(op, context)
               << "switch branch did not yield a value";
      }

      // Break/Continue/Return - propagate control flow immediately
      return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                                 op->getContext());
    };

    // Helper to process a runtime branch outcome (all branches evaluated).
    // Returns result to propagate, or nullopt to continue.
    auto processRuntimeBranchOutcome =
        [&](ControlFlowOutcomes &outcome) -> std::optional<ComptimeEvalResult> {
      if (outcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                                   op->getContext());

      if (outcome.hasYields()) {
        // Branch yielded a value - collect type for type inference
        Type branchType = outcome.mayYield[0].getType();
        if (branchType)
          evaluatedBranchTypes.push_back(branchType);
      } else if (outcome.isFallthroughOnly()) {
        // Switch branch must yield or use control flow
        return ComptimeError::semanticError(op, context)
               << "switch branch did not yield a value";
      }
      // Break/Continue/Return: valid, but doesn't contribute a type
      // (actual control flow happens at runtime, we're just type-checking)

      return std::nullopt;
    };

    // ========== PHASE 4: Evaluate branches ==========
    if (matchingCaseIdx) {
      // Comptime-known input with matching case: only evaluate that case
      Region &caseRegion = switchOp.getCaseRegions()[*matchingCaseIdx];
      ControlFlowOutcomes outcome =
          context.evaluateRegion(caseRegion, /*isRuntimeConditional=*/false,
                                 /*requiresComptime=*/false);
      if (auto result = processComptimeBranchOutcome(std::move(outcome)))
        return *result;

    } else if (inputIsComptimeKnown && !switchOp.hasElseBranch()) {
      // Comptime-known input with no match and no else
      return ComptimeError::semanticError(op, context)
             << "switch input matches no case and no else branch provided";

    } else if (inputIsComptimeKnown && switchOp.hasElseBranch()) {
      // Comptime-known input with no match: evaluate else
      Region &elseRegion = switchOp.getElseRegion();
      ControlFlowOutcomes outcome =
          context.evaluateRegion(elseRegion, /*isRuntimeConditional=*/false,
                                 /*requiresComptime=*/false);
      if (auto result = processComptimeBranchOutcome(std::move(outcome)))
        return *result;

    } else {
      // Runtime switch: evaluate all branches as runtime conditional.
      // Join all branch outcomes to track control flow (like IfOp does).
      for (Region &caseRegion : switchOp.getCaseRegions()) {
        ControlFlowOutcomes outcome =
            context.evaluateRegion(caseRegion, /*isRuntimeConditional=*/true,
                                   /*requiresComptime=*/false);
        if (auto result = processRuntimeBranchOutcome(outcome))
          return *result;
        joinedOutcome = joinedOutcome
                            ? ControlFlowOutcomes::join(*joinedOutcome, outcome)
                            : outcome;
      }

      if (switchOp.hasElseBranch()) {
        Region &elseRegion = switchOp.getElseRegion();
        ControlFlowOutcomes outcome =
            context.evaluateRegion(elseRegion, /*isRuntimeConditional=*/true,
                                   /*requiresComptime=*/false);
        if (auto result = processRuntimeBranchOutcome(outcome))
          return *result;
        joinedOutcome = joinedOutcome
                            ? ControlFlowOutcomes::join(*joinedOutcome, outcome)
                            : outcome;
      } else {
        // No else: switch may fall through without matching
        ControlFlowOutcomes fallthrough = ControlFlowOutcomes::fallthrough();
        joinedOutcome =
            joinedOutcome
                ? ControlFlowOutcomes::join(*joinedOutcome, fallthrough)
                : fallthrough;
      }
    }

    // ========== PHASE 5: Type inference ==========
    Type resultType;
    if (evaluatedBranchTypes.empty()) {
      // All branches use control flow (break/continue/return) - void result
      resultType = csl::VoidType::get(op->getContext());
    } else if (evaluatedBranchTypes.size() == 1) {
      resultType = evaluatedBranchTypes[0];
    } else {
      // Find common type across all branches
      std::pair<Type, Type> conflicting;
      auto unified = ControlFlowOutcomes::unifyTypes(
          evaluatedBranchTypes, op->getContext(), &conflicting);
      if (failed(unified)) {
        return ComptimeError::semanticError(op, context)
               << "incompatible types in switch branches: "
               << csl::prettyPrintType(conflicting.first) << " and "
               << csl::prettyPrintType(conflicting.second);
      }
      resultType = *unified;
    }

    // ========== PHASE 6: Comptime evaluation ==========
    if (isRuntimeSwitch) {
      // Input not comptime-known - return type with comptime failure
      ComptimeEvalResult r(
          ComptimeError::comptimeFailure(op, context, resultType)
              .withDependency(switchOp.getInput())
          << "switch input is not comptime-known");

      // Translate branch outcomes into the switch op's own control flow.
      // Branch yields are consumed by the switch (they become its result),
      // so they translate to fallthrough for the switch op itself.
      // Returns/breaks/continues propagate upward as-is.
      if (joinedOutcome) {
        if (joinedOutcome->hasYields())
          joinedOutcome->mayFallthrough = true;
        joinedOutcome->mayYield.clear();
        r.controlFlow = std::move(*joinedOutcome);
      }
      return r;
    }

    // Input is comptime-known: return the value from the evaluated branch
    // (if branch used control flow, we would have returned earlier)
    if (!comptimeBranchOutcome || comptimeBranchOutcome->mayYield.empty()) {
      return ComptimeError::semanticError(op, context)
             << "switch branch did not yield a value";
    }
    auto yieldedVal = comptimeBranchOutcome->mayYield[0].getValue();
    if (yieldedVal)
      return *yieldedVal;

    // No value (void) - return type with comptime failure
    return ComptimeError::comptimeFailure(op, context, resultType)
           << "switch branch yielded a non-comptime-known value";
  }
};

//===----------------------------------------------------------------------===//
// ShortCircuitOp
//===----------------------------------------------------------------------===//

struct ShortCircuitOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ShortCircuitOpComptimeModel,
                                                   ShortCircuitOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto shortCircuitOp = cast<ShortCircuitOp>(op);
    bool isAnd = shortCircuitOp.getKind() == csl::ShortCircuitKind::And;
    auto boolType = IntegerType::get(op->getContext(), 1);

    // ========== PHASE 1: Validate LHS ==========
    auto lhsType = context.getInferredRValueType(shortCircuitOp.getLhs());
    if (failed(lhsType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer LHS type";
    }
    if (*lhsType != boolType) {
      return ComptimeError::semanticError(op, context)
             << "expected LHS type to be boolean, got "
             << prettyPrintType(*lhsType);
    }

    // ========== PHASE 2: Check LHS comptime value ==========
    auto lhsValue = context.getOrLoadComptimeRValue(shortCircuitOp.getLhs());
    bool lhsIsComptimeTrue = false;
    bool lhsIsComptimeFalse = false;

    if (succeeded(lhsValue)) {
      auto lhsTyped = dyn_cast<TypedAttr>(*lhsValue);
      assert(lhsTyped && "LHS must be a typed attribute");
      auto lhsBoolResult = csl::evaluateCast(lhsTyped, boolType);
      assert(succeeded(lhsBoolResult) &&
             "LHS cast to bool must succeed (validated in step 1)");
      auto boolAttr = cast<IntegerAttr>(*lhsBoolResult);
      lhsIsComptimeTrue = boolAttr.getValue().getBoolValue();
      lhsIsComptimeFalse = !lhsIsComptimeTrue;
    }

    bool isRuntimeConditional = failed(lhsValue);

    // ========== PHASE 3: Conditionally evaluate RHS ==========
    // For &&: evaluate RHS if LHS is NOT comptime false
    // For ||: evaluate RHS if LHS is NOT comptime true
    bool shouldEvaluateRhs = isAnd ? !lhsIsComptimeFalse : !lhsIsComptimeTrue;

    // Store outcome so Phase 5 can access yield information via
    // ControlFlowOutcomes::mayYield instead of querying yield SSA values.
    std::optional<ControlFlowOutcomes> rhsOutcome;

    if (shouldEvaluateRhs) {
      Region &rhsRegion = shortCircuitOp.getRhsRegion();
      rhsOutcome =
          context.evaluateRegion(rhsRegion, isRuntimeConditional, false);
      if (rhsOutcome->hasFatalError())
        return rhsOutcome->getFatalError();

      // Validate that RHS can also be cast to bool, using mayYield for type
      // information instead of querying the yield op's SSA value directly.
      if (!rhsOutcome->hasYields()) {
        return ComptimeError::semanticError(op, context)
               << "RHS region did not yield a value";
      }
      Type rhsType =
          SelectOpComptimeModel::getRValueType(rhsOutcome->mayYield[0]);
      if (rhsType != boolType) {
        return ComptimeError::semanticError(op, context)
               << "expected RHS type to be boolean, got "
               << prettyPrintType(rhsType);
      }
    }

    // ========== PHASE 4: Type inference (always bool) ==========
    Type resultType = boolType;

    // ========== PHASE 5: Comptime evaluation ==========
    if (failed(lhsValue)) {
      // LHS not comptime-known
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "left-hand side of short-circuit operation is not "
                "comptime-known";
    }

    auto lhsTyped = dyn_cast<TypedAttr>(*lhsValue);
    assert(lhsTyped && "LHS must be a typed attribute");

    // Cast LHS to bool - must succeed since we validated above
    InFlightDiagnostic diag = emitRemark(op->getLoc());
    auto lhsBoolResult = csl::evaluateCast(lhsTyped, boolType);
    if (failed(lhsBoolResult)) {
      return ComptimeError::executionError(op, context)
                 .withType(resultType)
                 .withDependency(shortCircuitOp.getLhs())
             << "cannot cast LHS to boolean" << std::move(diag);
    }
    diag.abandon();

    auto boolAttr = cast<IntegerAttr>(*lhsBoolResult);
    bool lhsBool = boolAttr.getValue().getBoolValue();

    // Check for short-circuit condition
    // &&: short-circuit if LHS is false
    // ||: short-circuit if LHS is true
    bool shouldShortCircuit = isAnd ? !lhsBool : lhsBool;

    if (shouldShortCircuit) {
      // Return the LHS value (false for &&, true for ||)
      return cast<TypedAttr>(IntegerAttr::get(boolType, lhsBool ? 1 : 0));
    }

    // Otherwise, return the RHS value from the evaluation outcome
    assert(rhsOutcome && "RHS must have been evaluated if not short-circuited");
    if (!rhsOutcome->hasYields()) {
      return ComptimeError::semanticError(op, context)
             << "RHS region did not yield a value";
    }
    auto rhsYieldedValue = rhsOutcome->mayYield[0].getValue();
    if (!rhsYieldedValue) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "RHS value is not comptime-known";
    }

    // Cast RHS to bool - must succeed since we validated above
    auto rhsTyped = dyn_cast<TypedAttr>(*rhsYieldedValue);
    assert(rhsTyped && "RHS must be a typed attribute");

    auto rhsBoolResult = csl::evaluateCast(rhsTyped, boolType);
    assert(succeeded(rhsBoolResult) && "RHS cast to bool must succeed");

    return *rhsBoolResult;
  }
};

//===----------------------------------------------------------------------===//
// BuiltinTypeOp
//===----------------------------------------------------------------------===//

struct BuiltinTypeOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<BuiltinTypeOpComptimeModel,
                                                   BuiltinTypeOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinOp = cast<BuiltinTypeOp>(op);
    // Return the TypeAttr as a TypedAttr
    return cast<TypedAttr>(builtinOp.getValueAttr());
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Type construction operations are always comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// PtrTypeOp
//===----------------------------------------------------------------------===//

struct PtrTypeOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<PtrTypeOpComptimeModel,
                                                   PtrTypeOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto ptrOp = cast<PtrTypeOp>(op);

    auto pointeeType = context.getOrLoadComptimeRValue(ptrOp.getPointee());
    if (failed(pointeeType)) {
      return ComptimeError::semanticError(op, context)
             << "pointee type is not comptime-known";
    }
    auto pointeeTypeAttr = dyn_cast<csl::TypeAttr>(*pointeeType);
    if (!pointeeTypeAttr) {
      return ComptimeError::semanticError(op, context)
             << "pointee type is not a type";
    }
    // Return the TypeAttr as a TypedAttr
    return cast<TypedAttr>(csl::TypeAttr::get(csl::PointerType::get(
        pointeeTypeAttr.getValue(), ptrOp.getIsConst(), ptrOp.getIsManyPtr())));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Type construction operations are always comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// FuncTypeOp
//===----------------------------------------------------------------------===//

struct FuncTypeOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<FuncTypeOpComptimeModel,
                                                   FuncTypeOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto funcTypeOp = cast<FuncTypeOp>(op);
    llvm::SmallVector<Type> paramTypes;
    llvm::SmallVector<Type> returnTypes;

    // Get comptime values of all parameter types
    for (Value paramTypeVal : funcTypeOp.getParamTypes()) {
      auto paramType = context.getOrLoadComptimeRValue(paramTypeVal);
      if (failed(paramType)) {
        return ComptimeError::semanticError(op, context)
                   .withDependency(paramTypeVal)
               << "parameter type is not comptime-known";
      }
      auto paramTypeAttr = dyn_cast<csl::TypeAttr>(*paramType);
      if (!paramTypeAttr) {
        return ComptimeError::semanticError(op, context)
               << "parameter type is not a type";
      }
      paramTypes.push_back(paramTypeAttr.getValue());
    }

    // Get comptime values of all return types
    for (Value returnTypeVal : funcTypeOp.getReturnTypes()) {
      auto returnType = context.getOrLoadComptimeRValue(returnTypeVal);
      if (failed(returnType)) {
        return ComptimeError::semanticError(op, context)
               << "return type is not comptime-known";
      }
      auto returnTypeAttr = dyn_cast<csl::TypeAttr>(*returnType);
      if (!returnTypeAttr) {
        return ComptimeError::semanticError(op, context)
               << "return type is not a type";
      }
      returnTypes.push_back(returnTypeAttr.getValue());
    }
    // Return the TypeAttr as a TypedAttr
    return cast<TypedAttr>(csl::TypeAttr::get(
        FunctionType::get(op->getContext(), paramTypes, returnTypes)));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Type construction operations are always comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Helpers for function reference resolution
//===----------------------------------------------------------------------===//

/// Try to extract a FuncRefAttr from a TypedAttr value (handles both
/// FuncRefAttr and FuncAddrAttr).
static std::optional<csl::FuncRefAttr> tryExtractFuncRef(TypedAttr value) {
  if (auto funcRef = mlir::dyn_cast<csl::FuncRefAttr>(value))
    return funcRef;
  if (auto funcAddr = mlir::dyn_cast<csl::FuncAddrAttr>(value))
    return funcAddr.getFuncRef();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ArrayTypeOp
//===----------------------------------------------------------------------===//

struct ArrayTypeOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ArrayTypeOpComptimeModel,
                                                   ArrayTypeOp> {

  /// Cast a single integer value to u16, returning the u16 value or an error.
  static std::variant<ComptimeError, int64_t>
  castDimToU16(TypedAttr dimAttr, Operation *op, Value dimVal,
               ComptimeEvalContext &context) {
    InFlightDiagnostic diag = emitRemark(op->getLoc());
    FailureOr<TypedAttr> dimCasted = csl::evaluateCast(
        dimAttr, IntegerType::get(op->getContext(), 16, IntegerType::Unsigned));
    if (failed(dimCasted)) {
      return ComptimeError::semanticError(op, context).withDependency(dimVal)
             << "cannot cast dimension to u16" << std::move(diag);
    }
    diag.abandon();
    return static_cast<int64_t>(
        cast<IntegerAttr>(*dimCasted).getValue().getZExtValue());
  }

  /// Get the rectangle dimensions from global eval context, returning an error
  /// if not set.
  static std::variant<ComptimeError, std::pair<int64_t, int64_t>>
  getGridDims(Operation *op, ComptimeEvalContext &context) {
    auto *gctx = context.getGlobalEvalContext();
    if (!gctx) {
      return ComptimeError::semanticError(op, context)
             << "per-PE array dimension requires global eval context";
    }
    const auto &flags = gctx->getOptions().cslFlags;
    if (!flags.rectangle) {
      return ComptimeError::semanticError(op, context)
             << "per-PE array dimension requires rectangle to be specified "
                "(@set_rectangle)";
    }
    return std::pair{static_cast<int64_t>(flags.rectangle.width),
                     static_cast<int64_t>(flags.rectangle.height)};
  }

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto arrayTypeOp = cast<ArrayTypeOp>(op);

    // Get element type
    auto elementType =
        context.getOrLoadComptimeRValue(arrayTypeOp.getElementType());
    if (failed(elementType)) {
      return ComptimeError::semanticError(op, context)
                 .withDependency(arrayTypeOp.getElementType())
             << "element type is not comptime-known";
    }
    auto elementTypeAttr = dyn_cast<csl::TypeAttr>(*elementType);
    if (!elementTypeAttr) {
      return ComptimeError::semanticError(op, context)
             << "element type is not a type";
    }
    if (isa<csl::ArrayType, csl::PerPEArrayType>(elementTypeAttr.getValue()))
      return ComptimeError::semanticError(op, context)
             << "Cerebras' cslc compiler does not support arrays of arrays "
                "(here of '"
             << prettyPrintType(elementTypeAttr.getValue())
             << "'); list all dimensions in one pair of brackets instead, "
                "e.g. '[4, 5]i32'";

    // Classify each dimension as static, per-PE array, or per-PE function.
    // Per-PE dims may have different extents (array shape vs grid size for
    // functions). We collect them separately, then normalize to the max extent.
    struct PerPEDimData {
      int64_t width, height;
      SmallVector<int64_t> values; // width*height values in x-major order
    };

    bool hasPerPE = false;
    SmallVector<bool> dimIsPerPE;
    SmallVector<std::variant<int64_t, PerPEDimData>> dims;
    SmallVector<int64_t> staticShape; // for the all-static fast path

    for (Value dimVal : arrayTypeOp.getDimensions()) {
      auto dimAttr = context.getOrLoadComptimeRValue(dimVal);
      if (failed(dimAttr)) {
        return ComptimeError::semanticError(op, context).withDependency(dimVal)
               << "dimension is not comptime-known";
      }

      // Case 1: ArrayValueAttr with 2D shape → per-PE dimension
      if (auto arrayVal = mlir::dyn_cast<csl::ArrayValueAttr>(*dimAttr)) {
        auto arrShape = arrayVal.getShape();
        if (arrShape.size() != 2) {
          return ComptimeError::semanticError(op, context)
                     .withDependency(dimVal)
                 << "per-PE dimension array must be 2D, got " << arrShape.size()
                 << "D";
        }
        int64_t arrW = arrShape[0], arrH = arrShape[1];

        PerPEDimData data{arrW, arrH, {}};
        for (int64_t i = 0; i < arrW * arrH; ++i) {
          auto result =
              castDimToU16(arrayVal.getElement(i), op, dimVal, context);
          if (std::holds_alternative<ComptimeError>(result))
            return std::get<ComptimeError>(result);
          data.values.push_back(std::get<int64_t>(result));
        }

        hasPerPE = true;
        dimIsPerPE.push_back(true);
        dims.push_back(std::move(data));
        staticShape.push_back(0); // placeholder
        continue;
      }

      // Case 2: FuncRefAttr fn(u16, u16) -> integer → call for all (x, y)
      if (auto funcRef = tryExtractFuncRef(*dimAttr)) {
        auto gridOrErr = getGridDims(op, context);
        if (std::holds_alternative<ComptimeError>(gridOrErr))
          return std::get<ComptimeError>(gridOrErr);
        auto [gridW, gridH] = std::get<std::pair<int64_t, int64_t>>(gridOrErr);

        auto u16Ty =
            IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);

        // Look up the function — always a GenericFuncOp during analysis
        auto cu = op->getParentOfType<csl::CompilationUnitOp>();
        auto lookupResult = lookupFuncByRefAttr(*funcRef, cu);
        if (failed(lookupResult)) {
          return ComptimeError::semanticError(op, context)
                     .withDependency(dimVal)
                 << "cannot find function '"
                 << funcRef->getSymbolName().getValue() << "'";
        }

        auto *gfPtr = std::get_if<GenericFuncOp>(&*lookupResult);
        if (!gfPtr) {
          return ComptimeError::semanticError(op, context)
                     .withDependency(dimVal)
                 << "per-PE dimension function must be a generic function";
        }
        GenericFuncOp genericFunc = *gfPtr;

        auto funcTypeResult = context.evaluateFuncType(genericFunc, {}, {});
        if (std::holds_alternative<ComptimeError>(funcTypeResult)) {
          return std::get<ComptimeError>(funcTypeResult);
        }
        FunctionType funcType = std::get<FunctionType>(funcTypeResult);

        // Validate signature: fn(u16, u16) -> integer
        if (funcType.getNumInputs() != 2 || funcType.getInput(0) != u16Ty ||
            funcType.getInput(1) != u16Ty) {
          return ComptimeError::semanticError(op, context)
                     .withDependency(dimVal)
                 << "per-PE dimension function must take (u16, u16), got "
                 << prettyPrintType(funcType);
        }
        if (funcType.getNumResults() != 1 ||
            !mlir::isa<IntegerType>(funcType.getResult(0))) {
          return ComptimeError::semanticError(op, context)
                     .withDependency(dimVal)
                 << "per-PE dimension function must return an integer type";
        }

        PerPEDimData data{gridW, gridH, {}};
        // x-major order to match [W, H] ArrayValueAttr layout
        for (int64_t x = 0; x < gridW; ++x) {
          for (int64_t y = 0; y < gridH; ++y) {
            SmallVector<TypedAttr> runtimeArgValues = {
                IntegerAttr::get(u16Ty, x), IntegerAttr::get(u16Ty, y)};
            auto callResult = context.evaluateComptimeCallViaTemp(
                genericFunc, funcType, /*comptimeArgValues=*/{},
                /*anytypeArgTypes=*/{}, runtimeArgValues);
            if (std::holds_alternative<ComptimeError>(callResult)) {
              auto &err = std::get<ComptimeError>(callResult);
              err << "while evaluating per-PE dimension function at (" << x
                  << ", " << y << ")";
              return err;
            }
            auto resultAttr = std::get<TypedAttr>(callResult);
            auto castedResult = castDimToU16(resultAttr, op, dimVal, context);
            if (std::holds_alternative<ComptimeError>(castedResult))
              return std::get<ComptimeError>(castedResult);
            data.values.push_back(std::get<int64_t>(castedResult));
          }
        }

        hasPerPE = true;
        dimIsPerPE.push_back(true);
        dims.push_back(std::move(data));
        staticShape.push_back(0); // placeholder
        continue;
      }

      // Case 3: Scalar integer → static dimension (existing path)
      auto result = castDimToU16(*dimAttr, op, dimVal, context);
      if (std::holds_alternative<ComptimeError>(result))
        return std::get<ComptimeError>(result);
      int64_t dimSize = std::get<int64_t>(result);
      dimIsPerPE.push_back(false);
      dims.push_back(dimSize);
      staticShape.push_back(dimSize);
    }

    // If all dims are static, return regular ArrayType (unchanged behavior)
    if (!hasPerPE) {
      return cast<TypedAttr>(csl::TypeAttr::get(
          csl::ArrayType::get(staticShape, elementTypeAttr.getValue())));
    }

    // Compute the max data extent across all per-PE dimensions
    int64_t dataW = 0, dataH = 0;
    for (auto &dim : dims) {
      if (auto *ppd = std::get_if<PerPEDimData>(&dim)) {
        dataW = std::max(dataW, ppd->width);
        dataH = std::max(dataH, ppd->height);
      }
    }

    // Pack dimData, expanding smaller per-PE dims with zeros
    SmallVector<int64_t> dimData;
    for (auto &dim : dims) {
      if (auto *staticDim = std::get_if<int64_t>(&dim)) {
        dimData.push_back(*staticDim);
      } else {
        auto &ppd = std::get<PerPEDimData>(dim);
        for (int64_t x = 0; x < dataW; ++x) {
          for (int64_t y = 0; y < dataH; ++y) {
            if (x < ppd.width && y < ppd.height)
              dimData.push_back(ppd.values[x * ppd.height + y]);
            else
              dimData.push_back(0);
          }
        }
      }
    }

    auto *gctx = context.getGlobalEvalContext();
    const auto &flags = gctx->getOptions().cslFlags;
    int64_t offX = flags.fabricOffsets.width;
    int64_t offY = flags.fabricOffsets.height;

    return cast<TypedAttr>(csl::TypeAttr::get(
        csl::PerPEArrayType::get(dataW, dataH, offX, offY, dimIsPerPE, dimData,
                                 elementTypeAttr.getValue())));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Type construction operations are always comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

struct CallOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<CallOpComptimeModel,
                                                   CallOp> {
  /// Whether `use` is a type annotation: the type of a variable or global, or
  /// an operand of a type expression.  Type annotations are evaluated at
  /// comptime.
  static bool isTypeAnnotation(OpOperand &use) {
    Operation *user = use.getOwner();
    if (auto varOp = mlir::dyn_cast<VariableOp>(user))
      return use.get() == varOp.getDynamicType();
    if (auto globalOp = mlir::dyn_cast<GlobalOp>(user))
      return use.get() == globalOp.getDynamicType();
    return mlir::isa<PtrTypeOp, ArrayTypeOp, FuncTypeOp>(user);
  }

  /// A call that only provides type annotations is evaluated at comptime,
  /// like cslc does.  Any other call is a runtime call unless it appears in
  /// comptime code.
  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return llvm::any_of(op->getResults(),
                        [](Value result) { return !result.use_empty(); }) &&
           llvm::all_of(op->getUses(), isTypeAnnotation);
  }

  /// Check if this is an indirect call (callee is not comptime-known as an
  /// rvalue). This correctly handles lvalues: if the callee is a comptime-known
  /// lvalue but the value of that lvalue isn't comptime known, it's indirect.
  static bool isIndirectCall(CallOp callOp, ComptimeEvalContext &context) {
    return failed(context.getOrLoadComptimeRValue(callOp.getCallee()));
  }

  /// Get the function type for the callee from its inferred type.
  /// For function types, returns directly. For function pointer types, unwraps.
  /// Returns error if the callee type is not callable.
  static std::variant<ComptimeError, FunctionType>
  getCalleeFunctionType(CallOp callOp, ComptimeEvalContext &context) {
    FailureOr<Type> calleeType =
        context.getInferredRValueType(callOp.getCallee());
    if (failed(calleeType)) {
      return ComptimeError::semanticError(callOp, context)
             << "cannot infer type of callee";
    }

    // Handle function pointer types - unwrap to get the function type
    if (auto ptrType = mlir::dyn_cast<csl::PointerType>(*calleeType)) {
      if (auto funcType =
              mlir::dyn_cast<FunctionType>(ptrType.getPointeeType())) {
        return funcType;
      }
      return ComptimeError::semanticError(callOp, context)
             << "pointer callee does not point to a function type, got "
             << ptrType.getPointeeType();
    }

    // Direct function type
    if (auto funcType = mlir::dyn_cast<FunctionType>(*calleeType)) {
      return funcType;
    }

    return ComptimeError::semanticError(callOp, context)
           << "callee is not callable, got " << *calleeType;
  }

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto callOp = cast<CallOp>(op);

    auto shouldTrace = [&]() -> bool {
      auto *gctx = context.getGlobalEvalContext();
      return gctx && gctx->getOptions().cslFlags.traceErrors;
    };

    // ========== PHASE 1: Resolve callee ==========
    bool indirect = isIndirectCall(callOp, context);
    FunctionType funcType;
    GenericFuncOp genericFunc; // only set for direct calls
    ArrayRef<int32_t> comptimeParamIndices;
    SmallVector<int32_t> comptimeParamIndicesStorage;
    // Collected once in Phase 2, reused in Phase 7a/7b.
    SmallVector<TypedAttr> comptimeArgValues;
    SmallVector<Type> anytypeArgTypes;

    if (indirect) {
      // Indirect call: get function type from callee's inferred type
      auto maybeFuncType = getCalleeFunctionType(callOp, context);
      if (std::holds_alternative<ComptimeError>(maybeFuncType))
        return std::get<ComptimeError>(maybeFuncType);
      funcType = std::get<FunctionType>(maybeFuncType);

      if (funcType.getNumInputs() != callOp.getArgs().size()) {
        return ComptimeError::semanticError(op, context)
               << "expected " << funcType.getNumInputs()
               << " argument(s), but got " << callOp.getArgs().size();
      }
    } else {
      // Direct call: resolve callee
      FailureOr<TypedAttr> calleeValue =
          context.getOrLoadComptimeRValue(callOp.getCallee());
      std::optional<csl::FuncRefAttr> funcRef;
      if (succeeded(calleeValue))
        funcRef = tryExtractFuncRef(*calleeValue);

      if (!funcRef) {
        return ComptimeError::semanticError(op, context)
               << "direct call callee is not a function reference";
      }

      // Look up the function — always a GenericFuncOp during analysis
      // (GenericFuncOps are never erased; instantiated FuncOps get _inst
      // suffixed names and are never referenced by FuncRefAttr at this stage).
      auto cu = callOp->getParentOfType<csl::CompilationUnitOp>();
      auto lookupResult = lookupFuncByRefAttr(*funcRef, cu);
      if (failed(lookupResult)) {
        return ComptimeError::semanticError(op, context)
               << "function '" << funcRef->getSymbolName().getValue()
               << "' not found";
      }
      auto *gfPtr = std::get_if<GenericFuncOp>(&*lookupResult);
      assert(gfPtr && "direct call callee must be a GenericFuncOp");
      genericFunc = *gfPtr;

      // Early argument count check using the stored num_params attribute.
      if (static_cast<uint32_t>(callOp.getArgs().size()) !=
          genericFunc.getNumParams()) {
        return ComptimeError::semanticError(op, context)
               << "expected " << genericFunc.getNumParams()
               << " argument(s), but got " << callOp.getArgs().size();
      }

      // ========== PHASE 2: Get function type WITHOUT instantiation ==========
      // Collect comptime arg values and anytype arg types from call arguments.
      for (auto [idx, arg] : llvm::enumerate(callOp.getArgs())) {
        if (llvm::is_contained(genericFunc.getComptimeParams(), idx)) {
          FailureOr<TypedAttr> argValue = context.getOrLoadComptimeRValue(arg);
          if (failed(argValue)) {
            return ComptimeError::semanticError(op, context).withDependency(arg)
                   << "comptime argument is not comptime-known";
          }
          comptimeArgValues.push_back(*argValue);
        }
        if (llvm::is_contained(genericFunc.getAnytypeParams(), idx)) {
          FailureOr<Type> argType = context.getInferredRValueType(arg);
          if (failed(argType)) {
            return ComptimeError::semanticError(op, context)
                   << "cannot infer type of argument";
          }
          anytypeArgTypes.push_back(*argType);
        }
      }

      // Evaluate type region to get FunctionType.
      // This also validates comptime/anytype argument counts internally.
      auto funcTypeResult = context.evaluateFuncType(
          genericFunc, comptimeArgValues, anytypeArgTypes);
      if (std::holds_alternative<ComptimeError>(funcTypeResult)) {
        auto &err = std::get<ComptimeError>(funcTypeResult);
        if (shouldTrace())
          err.addCallStackEntry(callOp.getLoc(),
                                "in call to '" + genericFunc.getName() + "'");
        return err;
      }
      funcType = std::get<FunctionType>(funcTypeResult);

      comptimeParamIndicesStorage.assign(
          genericFunc.getComptimeParams().begin(),
          genericFunc.getComptimeParams().end());
      comptimeParamIndices = comptimeParamIndicesStorage;
    }

    // ========== PHASE 3: Validate argument types ==========
    for (auto [argIdx, arg] : llvm::enumerate(callOp.getArgs())) {
      if (llvm::is_contained(comptimeParamIndices,
                             static_cast<int32_t>(argIdx)))
        continue;
      Type paramType = funcType.getInput(argIdx);
      auto argType = context.getInferredRValueType(arg);
      if (failed(argType)) {
        return ComptimeError::semanticError(op, context)
               << "cannot infer type of argument " << (argIdx + 1);
      }
      bool comptimeKnown = context.isRValueComptimeKnown(arg);
      if (!csl::isImplicitCastAllowed(*argType, paramType, comptimeKnown)) {
        return ComptimeError::semanticError(op, context)
               << "implicit cast from " << prettyPrintType(*argType) << " to "
               << prettyPrintType(paramType) << " is not allowed for argument "
               << (argIdx + 1);
      }
    }

    // ========== PHASE 4: Type inference ==========
    ArrayRef<Type> results = funcType.getResults();
    Type resultType =
        results.empty() ? VoidType::get(op->getContext()) : results[0];

    // ========== PHASE 5: Determine comptime vs runtime ==========
    if (indirect) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "indirect calls cannot be evaluated at comptime";
    }

    bool mustEvalAtComptime = context.opMustEvaluateAtComptime(callOp);

    // Async functions can only be called from async contexts.
    if (genericFunc.getIsAsync()) {
      auto parentFunc = op->getParentOfType<FuncOp>();
      if (!parentFunc || !parentFunc.getIsAsync()) {
        return ComptimeError::semanticError(op, context)
               << "calls to async functions are only allowed in async "
                  "functions or tasks";
      }

      // An async call has no value of its own: it is awaited where it is
      // made, directly by @await or @await_parallel.
      auto isAwaitArgument = [](OpOperand &use) {
        auto builtin = dyn_cast<BuiltinCallOp>(use.getOwner());
        if (!builtin)
          return false;
        std::optional<csl::BuiltinKind> kind = builtin.resolveBuiltin();
        return kind && (*kind == csl::BuiltinKind::Await ||
                        *kind == csl::BuiltinKind::AwaitParallel);
      };
      Value result = callOp->getResult(0);
      if (!result.hasOneUse() || !isAwaitArgument(*result.use_begin()))
        return ComptimeError::semanticError(op, context)
               << "the call of async function '" << genericFunc.getName()
               << "' must be awaited where it is made: write `@await("
               << genericFunc.getName()
               << "(...))`, or pass the call to @await_parallel";
    }

    // ========== PHASE 6a: COMPTIME path ==========
    if (mustEvalAtComptime && genericFunc.getIsAsync()) {
      return ComptimeError::semanticError(op, context)
             << "async functions cannot be evaluated at comptime";
    }
    if (mustEvalAtComptime) {
      SmallVector<TypedAttr> runtimeArgValues;
      for (auto [idx, arg] : llvm::enumerate(callOp.getArgs())) {
        if (llvm::is_contained(genericFunc.getComptimeParams(), idx))
          continue;
        FailureOr<TypedAttr> argValue = context.getOrLoadComptimeRValue(arg);
        if (failed(argValue)) {
          return ComptimeError::comptimeFailure(op, context, resultType)
                     .withDependency(arg)
                 << "argument is not comptime-known";
        }
        runtimeArgValues.push_back(*argValue);
      }

      auto result = context.evaluateComptimeCallViaTemp(
          genericFunc, funcType, comptimeArgValues, anytypeArgTypes,
          runtimeArgValues);
      if (std::holds_alternative<ComptimeError>(result)) {
        auto &err = std::get<ComptimeError>(result);
        if (shouldTrace())
          err.addCallStackEntry(callOp.getLoc(), "in comptime call to '" +
                                                     genericFunc.getName() +
                                                     "'");
        return err.withType(resultType);
      }
      return std::get<TypedAttr>(result);
    }

    // ========== PHASE 6b: RUNTIME path ==========

    // ========== PHASE 6b-inline: Inline expansion ==========
    // Must happen before the comptime-only type check below, because inline
    // functions may accept comptime-only types (like `type`) without the
    // `comptime` keyword — the inline expansion substitutes them directly.
    if (genericFunc.getIsInline()) {
      OpBuilder builder(callOp);

      // Unique label to avoid conflicts
      static std::atomic<int64_t> inlineCounter{0};
      const int64_t inlineId =
          inlineCounter.fetch_add(1, std::memory_order_relaxed);
      std::string label = "__inline_" + genericFunc.getName().str() + "_" +
                          std::to_string(inlineId);

      // Create labeled block before the call
      auto labeledBlock = LabeledBlockOp::create(builder, callOp.getLoc(),
                                                 builder.getStringAttr(label),
                                                 /*inferred_type=*/nullptr);

      // Clone generic function body into the labeled block
      Block &bodyBlock = labeledBlock.getBody().emplaceBlock();
      builder.setInsertionPointToStart(&bodyBlock);

      IRMapping mapping;
      Block &genericBodyBlock = genericFunc.getBody().front();

      // Map generic block args: comptime → materialized constants,
      // runtime → call args (by full index, comptime args not yet stripped)
      size_t comptimeIdx = 0;
      for (size_t i = 0; i < genericBodyBlock.getNumArguments(); ++i) {
        Value genericArg = genericBodyBlock.getArgument(i);
        if (llvm::is_contained(genericFunc.getComptimeParams(),
                               static_cast<int32_t>(i))) {
          // Materialize comptime param as constant
          TypedAttr comptimeValue = comptimeArgValues[comptimeIdx++];
          InFlightDiagnostic diag = genericFunc.emitRemark();
          FailureOr<TypedAttr> castedValue = csl::evaluateImplicitCast(
              comptimeValue, funcType.getInput(i), &diag);
          // Phase 3 already validated this cast
          assert(succeeded(castedValue) &&
                 "implicit cast should succeed (validated in Phase 3)");
          diag.abandon();
          Operation *inlinedConst =
              genericFunc->getDialect()->materializeConstant(
                  builder, *castedValue, comptimeValue.getType(),
                  genericFunc.getLoc());
          mapping.map(genericArg, inlinedConst->getResult(0));
        } else {
          // Map runtime param to a const variable that acts as the
          // parameter proxy. NormalizeCastsPass handles type mismatches;
          // the body reads the parameter where it uses it as a value.
          Value arg = callOp.getArgs()[i];
          Type paramType = funcType.getInput(i);
          auto lvalueType = csl::LValueType::get(paramType, /*isConst=*/true);
          std::string paramName = "param_" + std::to_string(i);
          if (auto names = genericFunc.getParamNames()) {
            paramName = mlir::cast<StringAttr>((*names)[i]).getValue().str();
          }
          // Labeled blocks are eliminated later, so parameter proxies from
          // separate inline expansions can end up in the same lexical scope.
          // Include the atomic expansion ID to keep their names unique even
          // when compilation evaluates multiple PEs concurrently.
          paramName = label + "_" + paramName;
          Value dynamicType = BuiltinTypeOp::create(
              builder, callOp.getLoc(), parsed_csl::TypeAttr::get(paramType));
          auto varOp = VariableOp::create(
              builder, callOp.getLoc(),
              parsed_csl::UnresolvedType::get(callOp->getContext()),
              builder.getStringAttr(paramName),
              /*dynamicType=*/dynamicType, /*initializer=*/arg,
              /*is_const=*/builder.getUnitAttr(),
              /*is_comptime=*/UnitAttr(), mlir::TypeAttr::get(lvalueType));
          mapping.map(genericArg, varOp.getResult());
        }
      }

      // Clone all operations from the generic body
      for (Operation &bodyOp : genericBodyBlock.getOperations()) {
        Operation *clonedOp = builder.clone(bodyOp, mapping);
        if (auto definingModule =
                genericFunc->getParentOfType<parsed_csl::ModuleOp>()) {
          clonedOp->walk([&](GetSymbolOp symbol) {
            if (!symbol.getLexicalModuleIdentifier())
              symbol.setLexicalModuleIdentifier(
                  definingModule.getModuleIdentifier());
          });
        }
      }

      // Replace all ReturnOps with BreakOps targeting the labeled block
      SmallVector<ReturnOp> returns;
      labeledBlock.getBody().walk(
          [&](ReturnOp ret) { returns.push_back(ret); });
      for (ReturnOp ret : returns) {
        OpBuilder retBuilder(ret);
        BreakOp::create(retBuilder, ret.getLoc(),
                        retBuilder.getStringAttr(label), ret.getOperand());
        ret.erase();
      }

      // Insert implicit cast: labeled_block_result → function return type
      builder.setInsertionPointAfter(labeledBlock);
      Value targetTypeValue = BuiltinTypeOp::create(
          builder, callOp.getLoc(), parsed_csl::TypeAttr::get(resultType));
      CastOp castOp = CastOp::create(
          builder, callOp.getLoc(), labeledBlock.getResult(), targetTypeValue,
          mlir::TypeAttr::get(resultType), builder.getUnitAttr());

      // Replace call result with cast result and erase call
      callOp.getResult().replaceAllUsesWith(castOp.getResult());
      callOp.erase();

      return ComptimeEvalResult::replaced();
    }

    // A function returning a comptime-only type cannot exist at runtime.
    if (!mlir::isa<VoidType>(resultType) &&
        csl::isComptimeOnlyType(resultType)) {
      return ComptimeError::semanticError(op, context)
             << "'" << genericFunc.getName() << "' returns '"
             << csl::prettyPrintType(resultType)
             << "', which is not valid at runtime; call it at comptime with "
                "`comptime "
             << genericFunc.getName() << "(...)`";
    }

    // Validate that no non-comptime parameter has a comptime-only type.
    // Such params must be marked with the `comptime` keyword.
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      if (llvm::is_contained(genericFunc.getComptimeParams(),
                             static_cast<int32_t>(i)))
        continue;
      if (csl::isComptimeOnlyType(funcType.getInput(i))) {
        return ComptimeError::semanticError(op, context)
               << "function parameter type is not valid at runtime: '"
               << csl::prettyPrintType(funcType.getInput(i)) << "'";
      }
    }

    // ========== PHASE 6b-normal: Non-inline runtime path ==========
    // Instantiate for runtime use (cached). Reuses comptimeArgValues and
    // anytypeArgTypes collected in Phase 2.
    auto instResult = context.getOrInstantiateFunc(
        genericFunc, comptimeArgValues, anytypeArgTypes);
    if (std::holds_alternative<ComptimeError>(instResult)) {
      auto &err = std::get<ComptimeError>(instResult);
      if (shouldTrace())
        err.addCallStackEntry(callOp.getLoc(), "in runtime call to '" +
                                                   genericFunc.getName() + "'");
      return err;
    }
    FuncOp instFuncOp = std::get<FuncOp>(instResult);

    // Rewrite the CallOp in-place: replace callee with a constant referencing
    // the instantiated FuncOp, and strip comptime arguments.
    // This is safe because runtime calls only appear in runtime function bodies
    // which are evaluated exactly once.
    std::optional<uint64_t> moduleId;
    if (auto moduleOp = instFuncOp->getParentOfType<parsed_csl::ModuleOp>())
      moduleId = moduleOp.getModuleIdentifier();

    OpBuilder builder(callOp);
    auto funcRef = csl::FuncRefAttr::get(
        callOp->getContext(), moduleId,
        FlatSymbolRefAttr::get(instFuncOp.getSymNameAttr()),
        instFuncOp.getFunctionType());
    ConstantOp constOp = ConstantOp::create(builder, callOp.getLoc(), funcRef);
    context.markOpAlive(constOp);
    context.setComptimeValue(constOp.getResult(), funcRef);
    callOp.getCalleeMutable().assign(constOp.getResult());

    // Remove comptime arguments in reverse order to preserve indices
    for (int32_t idx : llvm::reverse(genericFunc.getComptimeParams()))
      callOp.getArgsMutable().erase(idx);

    // Use the instantiated function's result type (which already has
    // CoroFrameType wrapping for async functions).
    resultType = instFuncOp.getFunctionType().getResult(0);

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "runtime call";
  }
};

//===----------------------------------------------------------------------===//
// ComptimeExprOp
//===----------------------------------------------------------------------===//

struct ComptimeExprOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ComptimeExprOpComptimeModel,
                                                   ComptimeExprOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto comptimeExprOp = cast<ComptimeExprOp>(op);

    // Evaluate the entire expression region
    Region &exprRegion = comptimeExprOp.getExprRegion();
    ControlFlowOutcomes outcome =
        context.evaluateRegion(exprRegion, /*isRuntimeConditional=*/false,
                               /*requiresComptime=*/true);

    // If evaluation failed, propagate the failure with its diagnostic chain
    if (outcome.hasFatalError()) {
      return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                                 op->getContext());
    }

    // Check that we got a yield
    if (outcome.mayYield.empty()) {
      return ComptimeError::semanticError(op, context)
             << "comptime expression did not yield a value";
    }

    // In comptime context, exactly one yield should be produced
    if (outcome.mayYield.size() != 1) {
      return ComptimeError::semanticError(op, context)
             << "comptime expression should yield exactly once";
    }
    auto yieldedValue = outcome.mayYield[0].getValue();
    if (!yieldedValue) {
      return ComptimeError::comptimeFailure(op, context,
                                            outcome.mayYield[0].getType())
             << "comptime expression yield is not comptime-known";
    }

    context.setInferredTypeDirectly(comptimeExprOp.getResult(),
                                    yieldedValue->getType());
    return *yieldedValue;
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // comptime expressions must always be evaluated at comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ComptimeBlockOp
//===----------------------------------------------------------------------===//

struct ComptimeBlockOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ComptimeBlockOpComptimeModel,
                                                   ComptimeBlockOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto comptimeBlockOp = cast<ComptimeBlockOp>(op);

    // Evaluate the region and extract the result
    ControlFlowOutcomes outcome =
        context.evaluateRegion(comptimeBlockOp.getBodyRegion(),
                               /*isRuntimeConditional=*/false,
                               /*requiresComptime=*/true);

    return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                               op->getContext());
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // comptime blocks must always be evaluated at comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// LayoutBlockOp
//===----------------------------------------------------------------------===//

struct LayoutBlockOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<LayoutBlockOpComptimeModel,
                                                   LayoutBlockOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto layoutBlockOp = cast<LayoutBlockOp>(op);

    context.getGlobalEvalContext()->setEvaluatingLayoutBlock(true);

    // Evaluate the region and extract the result
    // Layout blocks are evaluated the same as comptime blocks
    ControlFlowOutcomes outcome =
        context.evaluateRegion(layoutBlockOp.getBodyRegion(),
                               /*isRuntimeConditional=*/false,
                               /*requiresComptime=*/true);

    context.getGlobalEvalContext()->setEvaluatingLayoutBlock(false);

    return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                               op->getContext());
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // layout blocks must always be evaluated at comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

struct ReturnOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ReturnOpComptimeModel,
                                                   ReturnOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto returnOp = cast<ReturnOp>(op);
    auto operands = returnOp.getOperands();

    // ========== PHASE 1-2: Validation ==========
    // Void return is always valid
    if (!operands.empty()) {
      // Find parent function
      auto funcOp = op->getParentOfType<FuncOp>();
      if (!funcOp) {
        return ComptimeError::semanticError(op, context)
               << "return not within function";
      }

      // Get function return type
      Type expectedType = funcOp.getBodyReturnType();

      // Check implicit cast is allowed for return value
      Value returnValue = operands[0];
      auto returnValueType = context.getInferredRValueType(returnValue);
      if (failed(returnValueType)) {
        return ComptimeError::semanticError(op, context)
               << "cannot infer return value type";
      }

      bool comptimeKnown = context.isRValueComptimeKnown(returnValue);
      if (!csl::isImplicitCastAllowed(*returnValueType, expectedType,
                                      comptimeKnown)) {
        return ComptimeError::semanticError(op, context)
               << "implicit cast from " << prettyPrintType(*returnValueType)
               << " to " << prettyPrintType(expectedType) << " is not allowed";
      }
    }

    // ========== PHASE 3-5: Evaluation ==========
    // ReturnOp has no results, so no type inference needed.
    assert(operands.size() <= 1 &&
           "CSL does not support multiple return values");

    if (operands.empty()) {
      return ComptimeEvalResult::returnVoid(op->getContext());
    }

    auto returnValue = context.getOrLoadComptimeRValue(operands[0]);
    if (succeeded(returnValue)) {
      return ComptimeEvalResult::returnWith(*returnValue);
    }

    // Value is not comptime-known — infer type (must succeed).
    auto retType = context.getInferredRValueType(operands[0]);
    if (failed(retType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of return value";
    }

    // Build from comptimeFailure, then override controlFlow with return.
    ComptimeEvalResult r(
        ComptimeError::comptimeFailure(op, context).withDependency(operands[0])
        << "return value is not comptime-known");
    r.controlFlow = ControlFlowOutcomes::returnWith(
        ComptimeValueOrType::fromType(*retType));
    return r;
  }
};

//===----------------------------------------------------------------------===//
// SuspendOp
//===----------------------------------------------------------------------===//

struct SuspendOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<SuspendOpComptimeModel,
                                                   SuspendOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    // Suspend is runtime-only; in comptime context it's an error.
    if (context.opMustEvaluateAtComptime(op)) {
      return ComptimeError::semanticError(op, context)
             << "suspend cannot be used in comptime context";
    }

    // Validate: must be inside an async function.
    auto funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || !funcOp.getIsAsync()) {
      return ComptimeError::semanticError(op, context)
             << "suspend is only valid inside an async function";
    }

    return ComptimeError::comptimeFailure(op, context) << "runtime suspend";
  }
};

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

struct AwaitOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<AwaitOpComptimeModel,
                                                   AwaitOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto awaitOp = cast<AwaitOp>(op);

    // Await is runtime-only; in comptime context it's an error.
    if (context.opMustEvaluateAtComptime(op)) {
      return ComptimeError::semanticError(op, context)
             << "await cannot be used in comptime context";
    }

    // Validate: must be inside an async function.
    auto funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || !funcOp.getIsAsync()) {
      return ComptimeError::semanticError(op, context)
             << "await is only valid inside an async function";
    }

    // Infer the result type from the FutureType of the operand.
    FailureOr<Type> frameType =
        context.getInferredRValueType(awaitOp.getFrameHandle());
    if (failed(frameType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of await operand";
    }

    auto futureType = mlir::dyn_cast<csl::FutureType>(*frameType);
    if (!futureType) {
      return ComptimeError::semanticError(op, context)
             << "await operand must be a future type, got "
             << csl::prettyPrintType(*frameType);
    }

    // Look up the async function to get its result type.
    auto cu = op->getParentOfType<csl::CompilationUnitOp>();
    auto calleeFuncOp = lookupFuncByFuture(futureType, cu);
    if (failed(calleeFuncOp)) {
      return ComptimeError::semanticError(op, context)
             << "cannot find async function '"
             << futureType.getSymbolName().getValue() << "'";
    }
    auto asyncResultTypeAttr = calleeFuncOp->getAsyncResultType();
    Type resultType = asyncResultTypeAttr
                          ? *asyncResultTypeAttr
                          : csl::VoidType::get(op->getContext());
    return ComptimeError::comptimeFailure(op, context, resultType)
           << "runtime await";
  }
};

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

struct YieldOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<YieldOpComptimeModel,
                                                   YieldOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto yieldOp = cast<YieldOp>(op);

    // YieldOp has no results, so no type inference needed.
    // Get the comptime value of the yielded value
    FailureOr<TypedAttr> yieldRValue =
        context.getOrLoadComptimeRValue(yieldOp.getValue());
    if (succeeded(yieldRValue)) {
      return ComptimeEvalResult::yieldWith(*yieldRValue);
    }

    // Module imports return lvalues, so also try getting an lvalue
    FailureOr<LValueDescriptor> yieldLValue =
        context.getComptimeLValue(yieldOp.getValue());
    if (succeeded(yieldLValue)) {
      return ComptimeEvalResult::yieldWith(*yieldLValue);
    }

    // Value is not comptime-known. We must still be able to infer its type.
    FailureOr<Type> inferredType =
        context.getInferredRValueType(yieldOp.getValue());
    if (failed(inferredType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of yielded value";
    }

    // Build result from comptimeFailure (preserves diagnostic chain for
    // fatalization), then override controlFlow with the yield so that
    // enclosing expressions (e.g. runtime switch) can do type inference.
    ComptimeEvalResult r(ComptimeError::comptimeFailure(op, context)
                             .withDependency(yieldOp.getValue())
                         << "yielded value is not comptime-known");
    r.controlFlow = ControlFlowOutcomes::yieldWith(
        ComptimeValueOrType::fromType(*inferredType));
    return r;
  }
};

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

struct IfOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<IfOpComptimeModel, IfOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto ifOp = cast<IfOp>(op);

    // IfOp has no results, so no type inference needed.
    // Evaluate the condition
    auto conditionValue = context.getOrLoadComptimeRValue(ifOp.getCondition());

    bool mustEvaluateAtComptime = context.opMustEvaluateAtComptime(op);

    if (mustEvaluateAtComptime && failed(conditionValue)) {
      return ComptimeError::comptimeFailure(op, context)
                 .withDependency(ifOp.getCondition())
             << "if condition is not comptime-known";
    }

    if (mlir::succeeded(conditionValue)) {
      // Condition is comptime-known
      auto condAttr = dyn_cast<IntegerAttr>(*conditionValue);
      if (!condAttr ||
          condAttr.getType() != IntegerType::get(op->getContext(), 1)) {
        return ComptimeError::comptimeFailure(op, context)
               << "if condition must be a boolean value";
      }
      bool condition = condAttr.getValue().getBoolValue();

      // Evaluate the appropriate branch
      Region &branchToEvaluate =
          condition ? ifOp.getThenRegion() : ifOp.getElseRegion();

      // If the else branch doesn't exist and condition is false, just succeed
      if (!condition && ifOp.getElseRegion().empty()) {
        return success();
      }

      // Evaluate the branch region
      ControlFlowOutcomes branchOutcome = context.evaluateRegion(
          branchToEvaluate, false, context.opMustEvaluateAtComptime(op));

      return ComptimeEvalResult::fromControlFlow(std::move(branchOutcome),
                                                 op->getContext());

    } else {
      // Condition is not comptime-known
      // Evaluate both branches as runtime conditional and JOIN outcomes
      assert(!mustEvaluateAtComptime);
      ControlFlowOutcomes thenOutcome = context.evaluateRegion(
          ifOp.getThenRegion(),
          /*isRuntimeConditional=*/true, mustEvaluateAtComptime);

      if (thenOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(thenOutcome),
                                                   op->getContext());

      ControlFlowOutcomes elseOutcome;
      if (!ifOp.getElseRegion().empty()) {
        elseOutcome = context.evaluateRegion(ifOp.getElseRegion(),
                                             /*isRuntimeConditional=*/true,
                                             mustEvaluateAtComptime);
        if (elseOutcome.hasFatalError())
          return ComptimeEvalResult::fromControlFlow(std::move(elseOutcome),
                                                     op->getContext());
      } else {
        // No else branch = implicit fallthrough
        elseOutcome = ControlFlowOutcomes::fallthrough();
      }

      // JOIN: union of all possible outcomes from both branches
      return ComptimeEvalResult::fromControlFlow(
          ControlFlowOutcomes::join(thenOutcome, elseOutcome),
          op->getContext());
    }
  }
};

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

struct ForOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ForOpComptimeModel, ForOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto forOp = cast<ForOp>(op);
    bool mustEvaluateAtComptime = context.opMustEvaluateAtComptime(op);

    if (FailureOr<Type> iterableType =
            context.getInferredRValueType(forOp.getIterable());
        succeeded(iterableType)) {
      if (auto arrayType = dyn_cast<csl::ArrayType>(*iterableType);
          arrayType && arrayType.getRank() > 1)
        return ComptimeError::semanticError(op, context)
               << "Cerebras' cslc compiler does not support for loops over "
                  "multi-dimensional arrays such as '"
               << prettyPrintType(arrayType)
               << "'; loop over the indices with @range instead";
    }

    // ForOp has no results, so no type inference needed.
    // Get the iterable value
    auto iterableValue = context.getOrLoadComptimeRValue(forOp.getIterable());

    if (mustEvaluateAtComptime) {
      if (failed(iterableValue)) {
        return ComptimeError::comptimeFailure(op, context)
                   .withDependency(forOp.getIterable())
               << "for loop iterable is not comptime-known";
      }

      // Unroll the loop at comptime
      return evaluateUnrolled(forOp, *iterableValue, context);
    }

    // Runtime path: if iterable is comptime-known, check for zero iterations
    // and handle inline expansion.
    if (succeeded(iterableValue)) {
      auto valuesOrErr = collectIterationValues(forOp, *iterableValue, context);
      if (std::holds_alternative<ComptimeError>(valuesOrErr))
        return std::get<ComptimeError>(valuesOrErr);
      auto &[elemAttrs, elemType] = std::get<1>(valuesOrErr);

      // Zero iterations: erase the loop entirely (dead code).
      if (elemAttrs.empty()) {
        forOp.erase();
        return ComptimeEvalResult::replaced();
      }

      // Inline for: physically unroll by cloning the body N times with the
      // loop variable replaced by a materialized constant.
      if (forOp.getIsInline())
        return expandInlineFor(forOp, elemAttrs, elemType, context);
    } else if (forOp.getIsInline()) {
      return ComptimeError::semanticError(op, context)
             << "inline for loop iterable is not comptime-known";
    }

    // Runtime loop: infer types and evaluate body as runtime conditional
    auto iterableType = context.getInferredRValueType(forOp.getIterable());
    if (failed(iterableType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of for loop iterable";
    }

    // Infer element type from iterable
    Type elemType = getElementType(*iterableType, op, context);
    if (!elemType)
      return ComptimeError::semanticError(op, context)
             << "cannot determine element type from iterable type "
             << *iterableType;

    // Set inferred types for loop variables
    context.setInferredTypeDirectly(forOp.getElemArg(), elemType);
    if (forOp.hasIndex()) {
      // Index is always u16
      auto u16Type =
          IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
      context.setInferredTypeDirectly(forOp.getIndexArg(), u16Type);
    }

    // Evaluate body as runtime conditional
    ControlFlowOutcomes bodyOutcome = context.evaluateRegion(
        forOp.getBody(), /*isRuntimeConditional=*/true, false);

    if (bodyOutcome.hasFatalError())
      return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                 op->getContext());

    // Consume breaks and continues targeting this loop
    bodyOutcome.consumeBreaksTo(op);
    bodyOutcome.consumeContinuesTo(op);

    // The loop itself falls through after completing
    bodyOutcome.mayFallthrough = true;

    return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                               op->getContext());
  }

private:
  /// Get the element type from an iterable type (range or array)
  Type getElementType(Type iterableType, Operation *op,
                      ComptimeEvalContext &context) const {
    if (auto rangeType = mlir::dyn_cast<csl::RangeType>(iterableType))
      return rangeType.getElementType();
    if (auto arrayType = mlir::dyn_cast<csl::ArrayType>(iterableType))
      return arrayType.getElementType();
    // Comptime range - element type is comptime_int
    if (mlir::isa<csl::ComptimeIntType>(iterableType))
      return iterableType;
    return Type();
  }

  /// Collect iteration values from an iterable attribute.
  /// Returns (elemAttrs, elemType) or an error.
  std::variant<ComptimeError, std::pair<SmallVector<TypedAttr>, Type>>
  collectIterationValues(ForOp forOp, TypedAttr iterableValue,
                         ComptimeEvalContext &context) const {
    if (auto rangeAttr = mlir::dyn_cast<csl::RangeAttr>(iterableValue)) {
      Type elemType = rangeAttr.getElementType();
      int64_t start = rangeAttr.getStart();
      int64_t stop = rangeAttr.getStop();
      int64_t step = rangeAttr.getStep();
      SmallVector<TypedAttr> values;
      for (int64_t i = start; (step > 0 ? i < stop : i > stop); i += step) {
        if (mlir::isa<csl::ComptimeIntType>(elemType)) {
          values.push_back(csl::ComptimeIntAttr::get(
              forOp->getContext(), APInt(64, i, /*isSigned=*/true)));
        } else {
          values.push_back(IntegerAttr::get(elemType, i));
        }
      }
      return std::make_pair(std::move(values), elemType);
    }

    if (auto arrayAttr = mlir::dyn_cast<csl::ArrayValueAttr>(iterableValue)) {
      Type elemType = arrayAttr.getType().getElementType();
      int64_t numElements = arrayAttr.getNumElements();
      SmallVector<TypedAttr> values;
      for (int64_t idx = 0; idx < numElements; ++idx)
        values.push_back(arrayAttr.getElement(idx));
      return std::make_pair(std::move(values), elemType);
    }

    return ComptimeError::semanticError(forOp, context)
           << "unsupported iterable type for for loop: "
           << prettyPrintType(iterableValue.getType());
  }

  /// Expand an inline for loop in runtime context by cloning the body
  /// N times with loop variables replaced by materialized constants.
  /// Caller guarantees elemAttrs is non-empty (zero iterations handled above).
  ComptimeEvalResult expandInlineFor(ForOp forOp, ArrayRef<TypedAttr> elemAttrs,
                                     Type elemType,
                                     ComptimeEvalContext &context) const {
    assert(!elemAttrs.empty() && "zero iterations should be handled by caller");

    OpBuilder builder(forOp);
    Block &bodyBlock = forOp.getBody().front();
    auto u16Type =
        IntegerType::get(forOp->getContext(), 16, IntegerType::Unsigned);

    for (auto [idx, elemAttr] : llvm::enumerate(elemAttrs)) {
      IRMapping mapping;

      // Materialize element constant
      Operation *elemConst = forOp->getDialect()->materializeConstant(
          builder, elemAttr, elemAttr.getType(), forOp.getLoc());
      mapping.map(forOp.getElemArg(), elemConst->getResult(0));

      // Materialize index constant if present
      if (forOp.hasIndex()) {
        auto idxAttr = IntegerAttr::get(u16Type, idx);
        Operation *idxConst = forOp->getDialect()->materializeConstant(
            builder, idxAttr, u16Type, forOp.getLoc());
        mapping.map(forOp.getIndexArg(), idxConst->getResult(0));
      }

      // Clone all operations from the body
      for (Operation &bodyOp : bodyBlock.getOperations())
        builder.clone(bodyOp, mapping);
    }

    forOp.erase();

    return ComptimeEvalResult::replaced();
  }

  /// Unroll a for loop when comptime evaluation is required.
  /// Evaluates the body N times with the loop variable set to each value.
  ComptimeEvalResult evaluateUnrolled(ForOp forOp, TypedAttr iterableValue,
                                      ComptimeEvalContext &context) const {
    auto valuesOrErr = collectIterationValues(forOp, iterableValue, context);
    if (std::holds_alternative<ComptimeError>(valuesOrErr))
      return std::get<ComptimeError>(valuesOrErr);
    auto &[elemAttrs, elemType] = std::get<1>(valuesOrErr);

    auto u16Type =
        IntegerType::get(forOp->getContext(), 16, IntegerType::Unsigned);

    for (auto [idx, elemAttr] : llvm::enumerate(elemAttrs)) {
      context.setComptimeValue(forOp.getElemArg(), elemAttr);
      context.setInferredTypeDirectly(forOp.getElemArg(), elemType);

      if (forOp.hasIndex()) {
        auto idxAttr = IntegerAttr::get(u16Type, idx);
        context.setComptimeValue(forOp.getIndexArg(), idxAttr);
        context.setInferredTypeDirectly(forOp.getIndexArg(), u16Type);
      }

      ControlFlowOutcomes bodyOutcome =
          context.evaluateRegion(forOp.getBody(), false, true);

      if (bodyOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                   forOp->getContext());

      auto breaks = bodyOutcome.consumeBreaksTo(forOp.getOperation());
      if (!breaks.empty())
        break;

      bool hadContinue = bodyOutcome.consumeContinuesTo(forOp.getOperation());

      if (!bodyOutcome.mayFallthrough && !hadContinue)
        return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                   forOp->getContext());
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

struct WhileOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<WhileOpComptimeModel,
                                                   WhileOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto whileOp = cast<WhileOp>(op);
    bool mustEvaluateAtComptime = context.opMustEvaluateAtComptime(op);

    // WhileOp has no results, so no type inference needed.
    if (mustEvaluateAtComptime) {
      // Comptime while loop: unroll by evaluating condition and body repeatedly
      return evaluateComptimeWhile(whileOp, context);
    }

    // Runtime loop: evaluate condition region once to infer types
    ControlFlowOutcomes condOutcome = context.evaluateRegion(
        whileOp.getCondRegion(), /*isRuntimeConditional=*/true, false);

    if (condOutcome.hasFatalError())
      return ComptimeEvalResult::fromControlFlow(std::move(condOutcome),
                                                 op->getContext());

    // Validate condition type using mayYield from the evaluation outcome
    // instead of querying the yield op's SSA value directly.
    if (condOutcome.mayYield.empty()) {
      return ComptimeError::semanticError(op, context)
             << "while loop condition region did not yield a value";
    }
    Type condType =
        SelectOpComptimeModel::getRValueType(condOutcome.mayYield[0]);

    // Condition should be boolean
    if (!mlir::isa<IntegerType>(condType) ||
        mlir::cast<IntegerType>(condType).getWidth() != 1) {
      return ComptimeError::semanticError(op, context)
             << "while loop condition must be boolean, got " << condType;
    }

    // Runtime loop: evaluate body as runtime conditional
    ControlFlowOutcomes bodyOutcome = context.evaluateRegion(
        whileOp.getBody(), /*isRuntimeConditional=*/true, false);

    if (bodyOutcome.hasFatalError())
      return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                 op->getContext());

    // Evaluate continuation if present
    if (!whileOp.getContinuation().empty()) {
      ControlFlowOutcomes contOutcome = context.evaluateRegion(
          whileOp.getContinuation(), /*isRuntimeConditional=*/true, false);

      if (contOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(contOutcome),
                                                   op->getContext());
    }

    // Consume breaks and continues targeting this loop
    bodyOutcome.consumeBreaksTo(op);
    bodyOutcome.consumeContinuesTo(op);

    // The loop itself falls through after completing
    bodyOutcome.mayFallthrough = true;

    return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                               op->getContext());
  }

private:
  /// Evaluate a comptime while loop by unrolling
  ComptimeEvalResult evaluateComptimeWhile(WhileOp whileOp,
                                           ComptimeEvalContext &context) const {
    Operation *loopOp = whileOp.getOperation();
    // Loop until condition becomes false or we hit a control flow statement
    while (true) {
      // Evaluate the condition region
      ControlFlowOutcomes condOutcome =
          context.evaluateRegion(whileOp.getCondRegion(), false, true);

      if (condOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(condOutcome),
                                                   whileOp->getContext());

      // Get the yielded condition value from the evaluation outcome
      if (condOutcome.mayYield.empty()) {
        return ComptimeError::semanticError(whileOp, context)
               << "while loop condition region did not yield a value";
      }
      auto condValue = condOutcome.mayYield[0].getValue();
      if (!condValue) {
        return ComptimeError::comptimeFailure(whileOp, context)
               << "while loop condition is not comptime-known";
      }

      // Check if condition is true
      auto condAttr = mlir::dyn_cast<IntegerAttr>(*condValue);
      if (!condAttr) {
        return ComptimeError::comptimeFailure(whileOp, context)
               << "while loop condition must be a boolean value";
      }

      // If condition is false, exit the loop
      if (condAttr.getValue().isZero())
        break;

      // Evaluate loop body
      ControlFlowOutcomes bodyOutcome =
          context.evaluateRegion(whileOp.getBody(), false, true);

      if (bodyOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                   whileOp->getContext());

      // Consume breaks/continues targeting this loop
      auto breaks = bodyOutcome.consumeBreaksTo(loopOp);
      if (!breaks.empty())
        break; // Break targets this loop - exit

      bool hadContinue = bodyOutcome.consumeContinuesTo(loopOp);

      // If there are remaining non-consumed control flow signals, propagate
      if (!bodyOutcome.mayFallthrough && !hadContinue) {
        return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                   whileOp->getContext());
      }

      // Evaluate continuation if present
      if (!whileOp.getContinuation().empty()) {
        ControlFlowOutcomes contOutcome =
            context.evaluateRegion(whileOp.getContinuation(), false, true);

        if (contOutcome.hasFatalError())
          return ComptimeEvalResult::fromControlFlow(std::move(contOutcome),
                                                     whileOp->getContext());

        // If continuation has non-fallthrough control flow, propagate
        if (!contOutcome.mayFallthrough) {
          return ComptimeEvalResult::fromControlFlow(std::move(contOutcome),
                                                     whileOp->getContext());
        }
      }
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// LabeledBlockOp
//===----------------------------------------------------------------------===//

struct LabeledBlockOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<LabeledBlockOpComptimeModel,
                                                   LabeledBlockOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto labeledBlockOp = cast<LabeledBlockOp>(op);
    bool mustEval = context.opMustEvaluateAtComptime(op);

    // Evaluate the body region
    ControlFlowOutcomes bodyOutcome = context.evaluateRegion(
        labeledBlockOp.getBody(), /*isRuntimeConditional=*/false, mustEval);

    if (bodyOutcome.hasFatalError())
      return ComptimeEvalResult::fromControlFlow(std::move(bodyOutcome),
                                                 op->getContext());

    // Consume breaks targeting this block (by Operation*).
    // SetVector deduplicates identical values/types from different branches,
    // enabling comptime detection when all branches yield the same value.
    SetVector<ComptimeValueOrType> breakVTs = bodyOutcome.consumeBreaksTo(op);

    // If body may fall through, that counts as yielding void
    if (bodyOutcome.mayFallthrough) {
      breakVTs.insert(
          ComptimeValueOrType::fromValue(csl::VoidAttr::get(op->getContext())));
      bodyOutcome.mayFallthrough = false; // We consume the fallthrough
    }

    // Validate all break types are compatible
    std::pair<Type, Type> conflicting;
    auto unifiedType = ControlFlowOutcomes::unifyTypes(
        breakVTs.getArrayRef(), op->getContext(), &conflicting);
    if (failed(unifiedType)) {
      return ComptimeError::semanticError(op, context)
             << "labeled block has breaks with incompatible types: "
             << csl::prettyPrintType(conflicting.first) << " and "
             << csl::prettyPrintType(conflicting.second);
    }

    // Build result with all channels populated.
    // evaluateOperation() will bind value + type to the SSA result.
    ComptimeEvalResult result(success());
    result.inferredType = *unifiedType;

    // If exactly one break and it's comptime-known, cast to the unified type
    // and set as the block's comptime result value.
    if (breakVTs.size() == 1 && breakVTs[0].hasValue()) {
      TypedAttr breakVal = *breakVTs[0].getValue();
      InFlightDiagnostic diag = emitRemark(op->getLoc());
      FailureOr<TypedAttr> castResult =
          csl::evaluateImplicitCast(breakVal, *unifiedType, &diag);
      if (failed(castResult)) {
        return ComptimeError::executionError(op, context).withType(*unifiedType)
               << "failed to cast break value to labeled block result type"
               << std::move(diag);
      }
      diag.abandon();
      result.value = *castResult;
    }

    // The block falls through only if it has at least one completion path
    // (a break targeting this block, or body fallthrough). If all paths exit
    // via outer breaks or returns, the block itself does not complete.
    bodyOutcome.mayFallthrough = !breakVTs.empty();
    result.controlFlow = std::move(bodyOutcome);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

struct BreakOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<BreakOpComptimeModel,
                                                   BreakOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto breakOp = cast<BreakOp>(op);
    auto targetLabel = breakOp.getTargetLabel();
    Value breakValue = breakOp.getValue();

    // ========== Validation ==========
    // Value can only be provided with a target label (targeting labeled block)
    if (breakValue && !targetLabel) {
      return ComptimeError::semanticError(op, context)
             << "break with value requires a target label";
    }

    // Verify break is inside a loop or labeled block
    Operation *target = op->getParentOp();
    bool targetIsLoop = false;

    while (target) {
      if (isa<ForOp, WhileOp>(target)) {
        if (!targetLabel) {
          targetIsLoop = true;
          break;
        }
        StringAttr loopLabel;
        if (auto forOp = dyn_cast<ForOp>(target))
          loopLabel = forOp.getLabelAttr();
        else if (auto whileOp = dyn_cast<WhileOp>(target))
          loopLabel = whileOp.getLabelAttr();

        if (loopLabel && loopLabel.getValue() == *targetLabel) {
          targetIsLoop = true;
          break;
        }
      }

      if (auto labeledBlock = dyn_cast<LabeledBlockOp>(target)) {
        if (targetLabel && labeledBlock.getLabel() == *targetLabel)
          break;
      }

      target = target->getParentOp();
    }

    if (!target) {
      if (targetLabel)
        return ComptimeError::semanticError(op, context)
               << "break label :" << *targetLabel << " not found";
      return ComptimeError::semanticError(op, context)
             << "break statement not within a loop or labeled block";
    }

    // Loops cannot receive values
    if (targetIsLoop && breakValue) {
      return ComptimeError::semanticError(op, context)
             << "break with value cannot target a loop";
    }

    // ========== Control flow result ==========
    // Determine the value/type of the break.
    // Use getOrLoadComptimeRValue() and getInferredRValueType() to unwrap
    // lvalues — break values are always rvalues.
    if (!breakValue) {
      return ComptimeEvalResult::breakTo(
          target,
          ComptimeValueOrType::fromValue(csl::VoidAttr::get(op->getContext())));
    }

    auto comptimeVal = context.getOrLoadComptimeRValue(breakValue);
    if (succeeded(comptimeVal)) {
      return ComptimeEvalResult::breakTo(
          target, ComptimeValueOrType::fromValue(*comptimeVal));
    }

    // Value is not comptime-known — infer type (must succeed).
    auto inferredType = context.getInferredRValueType(breakValue);
    if (failed(inferredType)) {
      return ComptimeError::semanticError(op, context)
             << "cannot infer type of break value";
    }

    // Build from comptimeFailure, then override controlFlow with break.
    ComptimeEvalResult r(
        ComptimeError::comptimeFailure(op, context).withDependency(breakValue)
        << "break value is not comptime-known");
    r.controlFlow = ControlFlowOutcomes::breakTo(
        target, ComptimeValueOrType::fromType(*inferredType));
    return r;
  }
};

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

struct ContinueOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<ContinueOpComptimeModel,
                                                   ContinueOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto continueOp = cast<ContinueOp>(op);
    auto targetLabel = continueOp.getTargetLabel();

    // ========== Validation ==========
    // Verify continue is inside a loop (not a labeled block)
    Operation *target = op->getParentOp();

    while (target) {
      // Continue cannot target labeled blocks
      if (auto labeledBlock = dyn_cast<LabeledBlockOp>(target)) {
        if (targetLabel && labeledBlock.getLabel() == *targetLabel) {
          return ComptimeError::semanticError(op, context)
                 << "continue cannot target labeled block :" << *targetLabel
                 << "; only loops can be continued";
        }
      }

      if (isa<ForOp, WhileOp>(target)) {
        if (!targetLabel)
          break; // Unlabeled continue targets innermost loop

        StringAttr loopLabel;
        if (auto forOp = dyn_cast<ForOp>(target))
          loopLabel = forOp.getLabelAttr();
        else if (auto whileOp = dyn_cast<WhileOp>(target))
          loopLabel = whileOp.getLabelAttr();

        if (loopLabel && loopLabel.getValue() == *targetLabel)
          break;
      }

      target = target->getParentOp();
    }

    if (!target) {
      if (targetLabel)
        return ComptimeError::semanticError(op, context)
               << "continue label :" << *targetLabel << " not found";
      return ComptimeError::semanticError(op, context)
             << "continue statement not within a loop";
    }

    // ContinueOp has no results, so no type inference needed.
    return ComptimeEvalResult::continueTo(target);
  }
};

//===----------------------------------------------------------------------===//
// NamedStructTypeOp
//===----------------------------------------------------------------------===//

struct NamedStructTypeOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<
          NamedStructTypeOpComptimeModel, NamedStructTypeOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto structTypeOp = cast<NamedStructTypeOp>(op);

    // Get field types from operands
    SmallVector<Type> fieldTypes;
    for (Value fieldTypeVal : structTypeOp.getFieldTypes()) {
      auto fieldTypeAttr = context.getOrLoadComptimeRValue(fieldTypeVal);
      if (failed(fieldTypeAttr))
        return ComptimeError::semanticError(op, context)
               << "field type is not comptime-known";

      auto typeAttr = dyn_cast<csl::TypeAttr>(*fieldTypeAttr);
      if (!typeAttr)
        return ComptimeError::semanticError(op, context)
               << "field type is not a type";

      fieldTypes.push_back(typeAttr.getValue());
    }

    // Get field names from attribute
    SmallVector<StringAttr> fieldNames;
    for (auto nameAttr : structTypeOp.getFieldNames()) {
      fieldNames.push_back(cast<StringAttr>(nameAttr));
    }

    // Named tuples (no field names but with a def_location) are not allowed.
    if (fieldNames.empty())
      return ComptimeError::semanticError(op, context)
             << "empty named struct types are not allowed";

    // Create the named struct type with def_location
    StructType structType =
        StructType::get(op->getContext(), fieldNames, fieldTypes,
                        LocationAttr(structTypeOp.getDefLocation()));

    // Return as TypeAttr
    return cast<TypedAttr>(csl::TypeAttr::get(structType));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Type construction operations are always comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// EnumTypeOp
//===----------------------------------------------------------------------===//

struct EnumTypeOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<EnumTypeOpComptimeModel,
                                                   EnumTypeOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto enumTypeOp = cast<EnumTypeOp>(op);

    // Get underlying integer type from operand
    auto underlyingTypeAttr =
        context.getOrLoadComptimeRValue(enumTypeOp.getUnderlyingType());
    if (failed(underlyingTypeAttr))
      return ComptimeError::semanticError(op, context)
             << "underlying type is not comptime-known";

    auto typeAttr = dyn_cast<csl::TypeAttr>(*underlyingTypeAttr);
    if (!typeAttr)
      return ComptimeError::semanticError(op, context)
             << "underlying type is not a type";

    auto underlyingType = dyn_cast<IntegerType>(typeAttr.getValue());
    if (!underlyingType)
      return ComptimeError::semanticError(op, context)
             << "enum underlying type must be an integer type, got "
             << typeAttr.getValue();

    // Get element names from attribute
    SmallVector<StringAttr> elementNames;
    for (auto nameAttr : enumTypeOp.getElementNames()) {
      elementNames.push_back(cast<StringAttr>(nameAttr));
    }

    // Compute element values: auto-assign from previous + 1
    SmallVector<int64_t> elementValues(elementNames.size(), 0);
    llvm::DenseSet<int64_t> hasExplicitValue;

    // First pass: collect explicitly assigned values
    auto explicitIndices = enumTypeOp.getExplicitIndicesAttr();
    auto explicitValues = enumTypeOp.getExplicitValues();

    if (explicitIndices) {
      for (size_t i = 0; i < explicitIndices.size(); ++i) {
        int64_t idx = cast<IntegerAttr>(explicitIndices[i]).getInt();
        Value valueOperand = explicitValues[i];

        auto valueAttr = context.getOrLoadComptimeRValue(valueOperand);
        if (failed(valueAttr))
          return ComptimeError::semanticError(op, context)
                 << "enum element value for '" << elementNames[idx]
                 << "' is not comptime-known";

        auto comptimeInt = dyn_cast<ComptimeIntAttr>(*valueAttr);
        if (!comptimeInt)
          return ComptimeError::semanticError(op, context)
                 << "enum element value must be a comptime integer";

        int64_t intVal = comptimeInt.getValue().getSExtValue();
        elementValues[idx] = intVal;
        hasExplicitValue.insert(idx);
      }
    }

    // Second pass: assign auto-values for elements without explicit values
    // Each auto-assigned value is previous element's value + 1
    int64_t nextAutoValue = 0;
    for (size_t i = 0; i < elementNames.size(); ++i) {
      if (!hasExplicitValue.contains(i)) {
        // Auto-assign: use nextAutoValue (previous + 1)
        elementValues[i] = nextAutoValue;
      }
      // Next auto value is always current + 1
      nextAutoValue = elementValues[i] + 1;
    }

    // Check for duplicate values (shouldn't happen but verify)
    llvm::DenseSet<int64_t> seenValues;
    for (size_t i = 0; i < elementNames.size(); ++i) {
      if (seenValues.contains(elementValues[i]))
        return ComptimeError::semanticError(op, context)
               << "duplicate enum value " << elementValues[i];
      seenValues.insert(elementValues[i]);
    }

    // Create the EnumType
    EnumType enumType =
        EnumType::get(op->getContext(), underlyingType, elementNames,
                      elementValues, enumTypeOp.getDefLocation());

    // Return as TypeAttr
    return cast<TypedAttr>(csl::TypeAttr::get(enumType));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // Type construction operations are always comptime
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ConstructTupleOp
//===----------------------------------------------------------------------===//

struct ConstructTupleOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<
          ConstructTupleOpComptimeModel, ConstructTupleOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto tupleOp = cast<ConstructTupleOp>(op);

    // ========== PHASE 1: Gather and validate field types ==========
    SmallVector<Type> fieldTypes;
    for (Value fieldVal : tupleOp.getFieldValues()) {
      auto fieldType = context.getInferredRValueType(fieldVal);
      if (failed(fieldType)) {
        return ComptimeError::semanticError(op, context)
               << "field value type not inferred";
      }
      fieldTypes.push_back(*fieldType);
    }

    // ========== PHASE 2: Type inference ==========
    StructType resultType = StructType::get(op->getContext(), fieldTypes);

    // ========== PHASE 3: Comptime evaluation ==========
    // Get comptime values for all fields
    SmallVector<TypedAttr> fieldValues;
    for (Value fieldVal : tupleOp.getFieldValues()) {
      auto fieldAttr = context.getOrLoadComptimeRValue(fieldVal);
      if (failed(fieldAttr))
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(fieldVal)
               << "field value is not comptime-known";
      fieldValues.push_back(*fieldAttr);
    }

    // Create StructValueAttr
    return cast<TypedAttr>(StructValueAttr::get(
        tupleOp->getContext(), fieldValues, resultType.getDefLocation(), {}));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false; // Can be runtime
  }
};

//===----------------------------------------------------------------------===//
// ConstructStructOp
//===----------------------------------------------------------------------===//

struct ConstructStructOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<
          ConstructStructOpComptimeModel, ConstructStructOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto structOp = cast<ConstructStructOp>(op);

    // ========== PHASE 1: Validation and type inference ==========
    StructType resultType;

    // Check if this is a named struct (has struct_type operand) or anonymous
    if (structOp.getStructType()) {
      // Named struct: validate and get the struct type
      auto structTypeValue =
          context.getOrLoadComptimeRValue(structOp.getStructType());
      if (failed(structTypeValue))
        return ComptimeError::semanticError(op, context)
               << "struct type is not comptime-known";

      auto typeAttr = dyn_cast<csl::TypeAttr>(*structTypeValue);
      if (!typeAttr)
        return ComptimeError::semanticError(op, context)
               << "struct type operand is not a type";

      auto structType = dyn_cast<StructType>(typeAttr.getValue());
      if (!structType)
        return ComptimeError::semanticError(op, context)
               << "expected StructType";

      // Get field names from the struct construction
      ArrayRef<Attribute> fieldNamesAttr = structOp.getFieldNames().getValue();
      OperandRange fieldValues = structOp.getFieldValues();

      if (fieldNamesAttr.size() != fieldValues.size())
        return ComptimeError::semanticError(op, context)
               << "field names count does not match field values count";

      // Check all fields are specified exactly once and implicit casts are
      // allowed
      llvm::DenseSet<StringRef> seenFields;
      llvm::DenseSet<StringRef> expectedFields;

      // Collect expected fields from struct type
      for (StringRef fieldName : structType.getFieldNames()) {
        expectedFields.insert(fieldName);
      }

      for (auto [idx, fieldValue] : llvm::enumerate(fieldValues)) {
        // Get the field name for this value
        auto fieldNameAttr = mlir::dyn_cast<StringAttr>(fieldNamesAttr[idx]);
        if (!fieldNameAttr)
          return ComptimeError::semanticError(op, context)
                 << "field name " << idx << " is not a string attribute";

        StringRef fieldName = fieldNameAttr.getValue();

        // Check for duplicates
        if (!seenFields.insert(fieldName).second)
          return ComptimeError::semanticError(op, context)
                 << "field '" << fieldName << "' specified more than once";

        // Look up the expected type for this field name in the struct type
        Type expectedType = structType.getFieldType(fieldName);
        if (!expectedType)
          return ComptimeError::semanticError(op, context)
                 << "field '" << fieldName << "' not found in struct type";

        // Check implicit cast is allowed
        auto actualType = context.getInferredRValueType(fieldValue);
        if (failed(actualType))
          return ComptimeError::semanticError(op, context)
                 << "cannot infer type of field '" << fieldName << "'";

        bool comptimeKnown = context.isRValueComptimeKnown(fieldValue);
        if (!csl::isImplicitCastAllowed(*actualType, expectedType,
                                        comptimeKnown))
          return ComptimeError::semanticError(op, context)
                 << "implicit cast from " << prettyPrintType(*actualType)
                 << " to " << prettyPrintType(expectedType)
                 << " is not allowed for field '" << fieldName << "'";
      }

      // Check all fields are specified
      for (StringRef expectedField : expectedFields) {
        if (!seenFields.contains(expectedField))
          return ComptimeError::semanticError(op, context)
                 << "field '" << expectedField << "' not specified";
      }

      resultType = structType;
    } else {
      // Anonymous struct: infer from field names and field value types
      SmallVector<Type> fieldTypes;
      for (Value fieldVal : structOp.getFieldValues()) {
        auto fieldType = context.getInferredRValueType(fieldVal);
        if (failed(fieldType)) {
          return ComptimeError::semanticError(op, context)
                 << "field value type not inferred";
        }
        fieldTypes.push_back(*fieldType);
      }

      // Get field names from attribute
      SmallVector<StringAttr> fieldNames;
      for (auto nameAttr : structOp.getFieldNames()) {
        fieldNames.push_back(cast<StringAttr>(nameAttr));
      }

      // Create anonymous struct type (no def_location)
      resultType = StructType::get(op->getContext(), fieldNames, fieldTypes);
    }

    // ========== PHASE 2: Comptime evaluation ==========
    // Get comptime values for all fields
    SmallVector<TypedAttr> fieldValues;

    for (auto [fieldVal, fieldName] :
         llvm::zip(structOp.getFieldValues(), resultType.getFieldNames())) {
      FailureOr<TypedAttr> fieldAttr =
          context.getOrLoadComptimeRValue(fieldVal);
      if (failed(fieldAttr))
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(fieldVal)
               << "field value is not comptime-known";
      TypedAttr castedFieldAttr = *fieldAttr;
      Type fieldType = resultType.getFieldType(fieldName.getValue());

      // Implicitly cast field value to field type if needed
      if (fieldType != castedFieldAttr.getType()) {
        auto castedAttr = csl::evaluateImplicitCast(*fieldAttr, fieldType);
        if (failed(castedAttr)) {
          return ComptimeError::comptimeFailure(op, context, resultType)
                     .withDependency(fieldVal)
                 << "failed to implicitly cast value of field "
                 << fieldName.getValue() << " from " << fieldAttr->getType()
                 << " to " << fieldType;
        }
        castedFieldAttr = *castedAttr;
      }

      fieldValues.push_back(castedFieldAttr);
    }

    // Create StructValueAttr
    return cast<TypedAttr>(StructValueAttr::get(
        structOp->getContext(), fieldValues, resultType.getDefLocation(),
        resultType.getFieldNames()));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false; // Can be runtime
  }
};

//===----------------------------------------------------------------------===//
// ConstructArrayOp
//===----------------------------------------------------------------------===//

struct ConstructArrayOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<
          ConstructArrayOpComptimeModel, ConstructArrayOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto arrayOp = cast<ConstructArrayOp>(op);

    // ========== PHASE 1: Validate array type ==========
    auto arrayTypeValue =
        context.getOrLoadComptimeRValue(arrayOp.getArrayType());
    if (failed(arrayTypeValue))
      return ComptimeError::semanticError(op, context)
             << "array type is not comptime-known";

    auto typeAttr = dyn_cast<csl::TypeAttr>(*arrayTypeValue);
    if (!typeAttr)
      return ComptimeError::semanticError(op, context)
             << "array type operand is not a type";

    auto arrayType = dyn_cast<ArrayType>(typeAttr.getValue());
    if (!arrayType)
      return ComptimeError::semanticError(op, context)
             << "expected array type, got "
             << csl::prettyPrintType(typeAttr.getValue());

    // Check element count matches array size
    int64_t expectedSize = arrayType.getNumElements();
    if (static_cast<int64_t>(arrayOp.getElements().size()) != expectedSize)
      return ComptimeError::semanticError(op, context)
             << "element count mismatch: expected " << expectedSize
             << " elements for array type " << csl::prettyPrintType(arrayType)
             << ", got " << arrayOp.getElements().size();

    // ========== PHASE 2: Validate element types ==========
    Type expectedElemType = arrayType.getElementType();
    for (auto [idx, elem] : llvm::enumerate(arrayOp.getElements())) {
      auto elemType = context.getInferredRValueType(elem);
      if (failed(elemType))
        return ComptimeError::semanticError(op, context)
               << "cannot infer type of element " << idx;

      bool comptimeKnown = context.isRValueComptimeKnown(elem);
      if (!csl::isImplicitCastAllowed(*elemType, expectedElemType,
                                      comptimeKnown))
        return ComptimeError::semanticError(op, context)
               << "cannot implicitly cast element " << idx << " from "
               << csl::prettyPrintType(*elemType) << " to "
               << csl::prettyPrintType(expectedElemType);
    }

    // ========== PHASE 3: Type inference ==========
    Type resultType = arrayType;

    // ========== PHASE 4: Comptime evaluation ==========
    // Get comptime values for all elements, applying implicit casts
    SmallVector<TypedAttr> elementValues;
    for (Value elem : arrayOp.getElements()) {
      auto elemAttr = context.getOrLoadComptimeRValue(elem);
      if (failed(elemAttr))
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(elem)
               << "element value is not comptime-known";

      TypedAttr castedElemAttr = *elemAttr;

      // Implicitly cast to element type if needed
      if (elemAttr->getType() != expectedElemType) {
        auto castedAttr =
            csl::evaluateImplicitCast(*elemAttr, expectedElemType);
        if (failed(castedAttr))
          return ComptimeError::comptimeFailure(op, context, resultType)
                 << "failed to cast element to array element type";
        castedElemAttr = *castedAttr;
      }

      elementValues.push_back(castedElemAttr);
    }

    // Create ArrayValueAttr
    return cast<TypedAttr>(
        ArrayValueAttr::get(arrayOp->getContext(), arrayType.getShape(),
                            expectedElemType, elementValues));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false; // Can be runtime
  }
};

//===----------------------------------------------------------------------===//
// FieldAccessOp
//===----------------------------------------------------------------------===//

struct FieldAccessOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<FieldAccessOpComptimeModel,
                                                   FieldAccessOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto fieldAccessOp = cast<FieldAccessOp>(op);

    // ========== PHASE 1: Validate base type ==========
    FailureOr<Type> baseType = context.getInferredType(fieldAccessOp.getBase());
    if (failed(baseType)) {
      return ComptimeError::semanticError(op, context)
             << "base type not inferred";
    }

    // Unpack lvalue if needed, but keep lvalue type info for constness
    Type valueType = *baseType;
    auto lvalueType = mlir::dyn_cast<csl::LValueType>(valueType);
    if (lvalueType) {
      valueType = lvalueType.getValueType();
    }

    // ========== PHASE 2: Validate field access and infer type ==========
    Type resultType;

    if (auto structType = dyn_cast<StructType>(valueType)) {
      if (structType.isTuple()) {
        return ComptimeError::semanticError(op, context)
               << "cannot access fields of tuple struct";
      }
      // Check the field exists in the struct
      Type fieldType = structType.getFieldType(fieldAccessOp.getFieldName());
      if (!fieldType) {
        return ComptimeError::semanticError(op, context)
               << "field '" << fieldAccessOp.getFieldName()
               << "' not found in struct";
      }
      // if base was an lvalue, return an lvalue type preserving constness
      if (lvalueType) {
        resultType = LValueType::get(fieldType, lvalueType.getIsConst());
      } else {
        resultType = fieldType;
      }
    } else if (auto moduleType = dyn_cast<ImportedModuleType>(valueType)) {
      // Modules must always be accessed via comptime-known lvalues
      if (*baseType == valueType) {
        return ComptimeError::semanticError(op, context)
               << "expected module lvalue for imported module type";
      }

      // Look up the member in the comptime-known module by name
      FailureOr<LValueDescriptor> lvalue =
          context.getComptimeLValue(fieldAccessOp.getBase());
      if (failed(lvalue)) {
        return ComptimeError::semanticError(op, context)
               << "imported module must be comptime-known";
      }
      assert(lvalue->isModuleBased());
      parsed_csl::ModuleOp baseModule = lvalue->getBaseModule();

      // Try to find as a variable first
      FailureOr<GlobalOp> fieldGlobalOp =
          baseModule.getMemberVariable(fieldAccessOp.getFieldName());
      if (succeeded(fieldGlobalOp)) {
        // Trigger on-demand evaluation so the global's type is resolved.
        ControlFlowOutcomes outcome =
            context.getGlobalEvalContext()->evaluateGlobalOnDemand(
                *fieldGlobalOp);
        if (outcome.hasFatalError())
          return outcome.getFatalError();

        FailureOr<Type> fieldType = context.getGlobalValueType(*fieldGlobalOp);
        if (failed(fieldType)) {
          return ComptimeError::semanticError(op, context)
                 << "cannot infer type of module member variable '"
                 << fieldAccessOp.getFieldName() << "'";
        }
        resultType = LValueType::get(*fieldType, fieldGlobalOp->getIsConst());
      } else {
        // Try to find as a function
        auto funcResult =
            baseModule.getMemberFunc(fieldAccessOp.getFieldName());

        if (failed(funcResult)) {
          return ComptimeError::semanticError(op, context)
                 << "member '" << fieldAccessOp.getFieldName()
                 << "' not found in module";
        }

        // During analysis, all function declarations are GenericFuncOps.
        auto genericFuncOp = std::get<GenericFuncOp>(funcResult.value());

        if (!genericFuncOp.isActuallyGeneric()) {
          // Non-generic: evaluate type region to get concrete function type
          auto funcTypeResult = context.evaluateFuncType(genericFuncOp, {}, {});
          if (std::holds_alternative<ComptimeError>(funcTypeResult)) {
            return std::get<ComptimeError>(funcTypeResult);
          }
          resultType = std::get<FunctionType>(funcTypeResult);
        } else {
          // Truly generic: return GenericFunctionType marker
          resultType = GenericFunctionType::get(op->getContext());
        }
      }
    } else if (mlir::isa<TypeType>(valueType)) {
      // Could be an enum type access - get the comptime value to check
      FailureOr<TypedAttr> baseAttr =
          context.getOrLoadComptimeRValue(fieldAccessOp.getBase());
      if (failed(baseAttr)) {
        return ComptimeError::semanticError(op, context)
                   .withDependencies(fieldAccessOp.getBase())
               << "type value must be comptime-known for field access";
      }
      auto typeAttr = dyn_cast<csl::TypeAttr>(*baseAttr);
      if (!typeAttr) {
        return ComptimeError::semanticError(op, context)
               << "expected type value for field access";
      }
      // Check if it's an enum type
      auto enumType = dyn_cast<EnumType>(typeAttr.getValue());
      if (!enumType) {
        return ComptimeError::semanticError(op, context)
               << "expected enum type for field access, got "
               << typeAttr.getValue();
      }
      // Validate the element exists in the enum
      std::optional<int64_t> elementValue =
          enumType.getElementValue(fieldAccessOp.getFieldName());
      if (!elementValue) {
        return ComptimeError::semanticError(op, context)
               << "element '" << fieldAccessOp.getFieldName()
               << "' not found in enum type";
      }
      // Enum element access always returns the enum type (as an rvalue)
      resultType = enumType;
    } else {
      return ComptimeError::semanticError(op, context)
             << "expected struct or module base type, got " << valueType;
    }

    // Annotate base type. Needed to detect module member accesses during
    // module inlining
    fieldAccessOp.setInferredBaseType(*baseType);

    // ========== PHASE 3: Comptime evaluation ==========
    // Check if this is a function member access (result is FunctionType or
    // GenericFunctionType, not LValueType)
    if (mlir::isa<mlir::FunctionType, GenericFunctionType>(resultType)) {
      // This is a module function member access - return FuncRefAttr
      FailureOr<LValueDescriptor> lvalueDescr =
          context.getComptimeLValue(fieldAccessOp.getBase());
      assert(succeeded(lvalueDescr) && lvalueDescr->isModuleBased());
      parsed_csl::ModuleOp baseModule = lvalueDescr->getBaseModule();

      // Get the function (instantiated or generic)
      auto funcResult = baseModule.getMemberFunc(fieldAccessOp.getFieldName());
      assert(succeeded(funcResult) && "Function should exist after validation");

      uint64_t moduleId =
          static_cast<uint64_t>(baseModule.getModuleIdentifier());

      if (auto *funcOp = std::get_if<FuncOp>(&funcResult.value())) {
        return (TypedAttr)csl::FuncRefAttr::get(
            op->getContext(), moduleId,
            FlatSymbolRefAttr::get(funcOp->getSymNameAttr()),
            funcOp->getFunctionType());
      }
      // GenericFuncOp
      auto *genericFuncOp = std::get_if<GenericFuncOp>(&funcResult.value());
      if (!(*genericFuncOp).isActuallyGeneric()) {
        // Return with concrete type using original name
        return (TypedAttr)csl::FuncRefAttr::get(
            op->getContext(), moduleId,
            FlatSymbolRefAttr::get(genericFuncOp->getSymNameAttr()),
            mlir::cast<FunctionType>(resultType));
      }
      return (TypedAttr)csl::FuncRefAttr::get(
          op->getContext(), moduleId,
          FlatSymbolRefAttr::get(genericFuncOp->getSymNameAttr()),
          GenericFunctionType::get(op->getContext()));
    }

    // Check if this is an enum element access (e.g., direction.EAST).
    // Only match when the base is a type value — a struct field that happens
    // to have EnumType should fall through to the struct handling below.
    if (mlir::isa<TypeType>(valueType)) {
      if (auto enumType = dyn_cast<EnumType>(resultType)) {
        std::optional<int64_t> elementValue =
            enumType.getElementValue(fieldAccessOp.getFieldName());
        if (!elementValue) {
          return ComptimeError::semanticError(op, context)
                 << "element '" << fieldAccessOp.getFieldName()
                 << "' not found in enum type";
        }
        return (TypedAttr)EnumValueAttr::get(
            op->getContext(), enumType,
            StringAttr::get(op->getContext(), fieldAccessOp.getFieldName()),
            *elementValue);
      }
    }

    // Try to get the struct as an lvalue descriptor first
    FailureOr<LValueDescriptor> lvalueDescr =
        context.getComptimeLValue(fieldAccessOp.getBase());
    if (succeeded(lvalueDescr)) {
      auto lvalType = mlir::dyn_cast<LValueType>(resultType);
      if (lvalType) {
        Type fieldType = lvalType.getValueType();
        // Extend the access path with this field
        return lvalueDescr->extendWithField(fieldAccessOp.getFieldName(),
                                            fieldType, context);
      }
    }

    // Otherwise, try to get it as a comptime rvalue (auto-load if lvalue)
    FailureOr<TypedAttr> rvalue =
        context.getOrLoadComptimeRValue(fieldAccessOp.getBase());
    if (failed(rvalue))
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(fieldAccessOp.getBase())
             << "base value is not comptime-known";

    auto structValue = dyn_cast<StructValueAttr>(*rvalue);
    if (!structValue)
      return ComptimeError::semanticError(op, context)
                 .withDependency(fieldAccessOp.getBase())
             << "base value is not a struct";

    // Get the field value by name
    TypedAttr fieldValue = structValue.getField(fieldAccessOp.getFieldName());
    assert(fieldValue);
    return fieldValue;
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false; // Can be runtime
  }
};

//===----------------------------------------------------------------------===//
// IndexAccessOp
//===----------------------------------------------------------------------===//

struct IndexAccessOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<IndexAccessOpComptimeModel,
                                                   IndexAccessOp> {
  // Helper to convert index SSA values to int64_t values
  static FailureOr<SmallVector<int64_t>>
  getIndexValues(IndexAccessOp indexAccessOp, ComptimeEvalContext &context) {
    SmallVector<int64_t> indexValues;
    // Cast to comptime_int to handle any integer type (runtime types coerce)
    Type comptimeIntType = ComptimeIntType::get(indexAccessOp->getContext());

    for (Value indexOperand : indexAccessOp.getIndices()) {
      FailureOr<TypedAttr> indexAttr =
          context.getOrLoadComptimeRValue(indexOperand);
      if (failed(indexAttr))
        return failure();

      FailureOr<TypedAttr> indexCastResult =
          csl::evaluateImplicitCast(*indexAttr, comptimeIntType);
      if (failed(indexCastResult))
        return failure();

      auto comptimeInt = cast<ComptimeIntAttr>(*indexCastResult);
      indexValues.push_back(comptimeInt.getValue().getSExtValue());
    }
    return indexValues;
  }

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto indexAccessOp = cast<IndexAccessOp>(op);

    // ========== PHASE 1: Validate base type ==========
    auto baseTypeResult = context.getInferredType(indexAccessOp.getBase());
    if (failed(baseTypeResult)) {
      return ComptimeError::semanticError(op, context)
             << "base type not inferred";
    }

    // Unpack lvalue if needed, but keep lvalue type info for constness
    Type valueType = *baseTypeResult;
    auto lvalueType = mlir::dyn_cast<csl::LValueType>(valueType);
    if (lvalueType) {
      valueType = lvalueType.getValueType();
    }

    // Must have at least one index
    if (indexAccessOp.getIndices().empty()) {
      return ComptimeError::semanticError(op, context)
             << "must have at least one index";
    }

    // Validate that all indices are integer types
    for (auto [idx, indexOperand] :
         llvm::enumerate(indexAccessOp.getIndices())) {
      FailureOr<Type> indexType = context.getInferredRValueType(indexOperand);
      if (failed(indexType)) {
        return ComptimeError::semanticError(op, context)
               << "failed to infer type of index " << idx;
      }
      if (!csl::isAnyIntegerType(*indexType)) {
        return ComptimeError::semanticError(op, context)
               << "index must be an integer type, got "
               << csl::prettyPrintType(*indexType);
      }
    }

    // ========== PHASE 2: Validate access and infer type ==========
    Type resultType;

    if (auto structType = dyn_cast<StructType>(valueType)) {
      // Struct/tuple access: single index, must be comptime-known
      if (indexAccessOp.getIndices().size() != 1) {
        return ComptimeError::semanticError(op, context)
               << "struct/tuple access requires exactly one index";
      }

      // Index must be comptime-known for heterogeneous structs
      FailureOr<SmallVector<int64_t>> indices =
          getIndexValues(indexAccessOp, context);
      if (failed(indices)) {
        return ComptimeError::semanticError(op, context)
               << "struct/tuple index must be comptime-known";
      }

      int64_t index = (*indices)[0];
      if (index < 0 ||
          static_cast<size_t>(index) >= structType.getNumFields()) {
        return ComptimeError::semanticError(op, context)
               << "index " << index << " out of bounds for struct with "
               << structType.getNumFields() << " fields";
      }

      Type resultValueType = structType.getFieldTypes()[index];
      if (lvalueType) {
        resultType = LValueType::get(resultValueType, lvalueType.getIsConst());
      } else {
        resultType = resultValueType;
      }
    } else if (auto arrayType = dyn_cast<ArrayType>(valueType)) {
      // Array access: indices count must match array rank for element access
      ArrayRef<int64_t> shape = arrayType.getShape();

      if (indexAccessOp.getIndices().size() > shape.size()) {
        return ComptimeError::semanticError(op, context)
               << "too many indices (" << indexAccessOp.getIndices().size()
               << ") for array with rank " << shape.size();
      }

      // If indices are comptime-known, validate bounds
      FailureOr<SmallVector<int64_t>> indices =
          getIndexValues(indexAccessOp, context);
      if (succeeded(indices)) {
        for (size_t i = 0; i < indices->size(); ++i) {
          int64_t idx = (*indices)[i];
          if (idx < 0 || idx >= shape[i]) {
            return ComptimeError::semanticError(op, context)
                   << "index " << idx << " at dimension " << i
                   << " is out of bounds (shape is " << shape[i] << ")";
          }
        }
      }

      // Infer result type based on number of indices
      size_t numIndices = indexAccessOp.getIndices().size();
      Type resultValueType;
      if (numIndices == shape.size()) {
        // Full indexing: result is element type
        resultValueType = arrayType.getElementType();
      } else {
        // Partial indexing: result is sub-array
        SmallVector<int64_t> remainingShape(shape.begin() + numIndices,
                                            shape.end());
        resultValueType = ArrayType::get(op->getContext(), remainingShape,
                                         arrayType.getElementType());
      }

      if (lvalueType) {
        resultType = LValueType::get(resultValueType, lvalueType.getIsConst());
      } else {
        resultType = resultValueType;
      }
    } else if (auto ptrType = dyn_cast<PointerType>(valueType)) {
      // Many-item pointer access: single index, runtime allowed
      if (!ptrType.getIsManyPtr()) {
        return ComptimeError::semanticError(op, context)
               << "cannot index into single-item pointer; use .* to "
                  "dereference";
      }
      if (indexAccessOp.getIndices().size() != 1) {
        return ComptimeError::semanticError(op, context)
               << "pointer indexing requires exactly one index";
      }
      // Pointer indexing always returns an lvalue with constness from the
      // pointer
      resultType =
          LValueType::get(ptrType.getPointeeType(), ptrType.getIsConst());
    } else {
      return ComptimeError::semanticError(op, context)
             << "expected struct, array, or many-item pointer type, got "
             << csl::prettyPrintType(valueType);
    }

    // ========== PHASE 3: Comptime evaluation ==========
    // Get indices as integer values
    FailureOr<SmallVector<int64_t>> indices =
        getIndexValues(indexAccessOp, context);
    if (failed(indices)) {
      // Runtime indices
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "index values are not comptime-known";
    }

    // Pointer indexing dereferences into separate memory and cannot be
    // evaluated at comptime. Return early with the correctly inferred type.
    if (isa<PointerType>(valueType))
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "pointer indexing is not comptime-evaluable";

    // Try to get the base as an lvalue descriptor first
    FailureOr<LValueDescriptor> baseLValueDescriptor =
        context.getComptimeLValue(indexAccessOp.getBase());
    if (succeeded(baseLValueDescriptor)) {
      auto resultLvalueType = mlir::dyn_cast<LValueType>(resultType);
      if (resultLvalueType) {
        Type elementType = resultLvalueType.getValueType();

        // Compute linear index for arrays, or use single index for structs
        int64_t linearIndex;
        if (auto arrayType = dyn_cast<ArrayType>(valueType)) {
          // Compute linear index using row-major ordering
          ArrayRef<int64_t> shape = arrayType.getShape();
          linearIndex = 0;
          int64_t stride = 1;
          for (int i = static_cast<int>(indices->size()) - 1; i >= 0; --i) {
            linearIndex += (*indices)[i] * stride;
            stride *= shape[i];
          }
        } else {
          // Struct/tuple: single index
          linearIndex = (*indices)[0];
        }

        // Extend the access path with this index
        return baseLValueDescriptor->extendWithIndex(linearIndex, elementType);
      }
    }

    // Otherwise, try to get it as a comptime rvalue (auto-load if lvalue)
    FailureOr<TypedAttr> rvalue =
        context.getOrLoadComptimeRValue(indexAccessOp.getBase());
    if (failed(rvalue))
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(indexAccessOp.getBase())
             << "base value is not comptime-known";

    // For arrays: get element at computed linear index
    if (auto arrayValue = mlir::dyn_cast<ArrayValueAttr>(*rvalue)) {
      // Compute linear index using row-major ordering
      ArrayRef<int64_t> shape = arrayValue.getShape();
      int64_t linearIndex = 0;
      int64_t stride = 1;
      for (int i = static_cast<int>(indices->size()) - 1; i >= 0; --i) {
        linearIndex += (*indices)[i] * stride;
        stride *= shape[i];
      }
      return arrayValue.getElement(linearIndex);
    }

    // For structs: get field at index
    if (auto structValue = mlir::dyn_cast<StructValueAttr>(*rvalue)) {
      size_t index = static_cast<size_t>((*indices)[0]);
      return structValue.getField(index);
    }

    return ComptimeError::semanticError(op, context)
               .withDependency(indexAccessOp.getBase())
           << "base value is not an array or struct";
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false; // Can be runtime
  }
};

//===----------------------------------------------------------------------===//
// BuiltinCallOp
//===----------------------------------------------------------------------===//

/// BuiltinCallOpComptimeModel delegates to the BuiltinEvaluatorRegistry
/// to get per-builtin evaluation logic.
struct BuiltinCallOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<BuiltinCallOpComptimeModel,
                                                   BuiltinCallOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto builtinKind = builtinCallOp.resolveBuiltin();
    if (!builtinKind) {
      return ComptimeError::semanticError(op, context)
             << "unknown builtin function: @" << builtinCallOp.getBuiltinName();
    }
    const auto &evaluator = context.getBuiltinRegistry().get(*builtinKind);
    return evaluator.evaluate(&evaluator, op, context);
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto builtinKind = builtinCallOp.resolveBuiltin();
    if (!builtinKind)
      return false; // Unknown builtin — will error during evaluate()
    const auto &evaluator = context.getBuiltinRegistry().get(*builtinKind);
    return evaluator.requiresComptimeBySemantics(&evaluator, op, context);
  }
};

//===----------------------------------------------------------------------===//
// NoValueOpComptimeModel
//===----------------------------------------------------------------------===//

struct NoValueOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<NoValueOpComptimeModel,
                                                   NoValueOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto noValueOp = cast<NoValueOp>(op);

    // ========== PHASE 1: Validation ==========
    FailureOr<TypedAttr> typeAttr =
        context.getOrLoadComptimeRValue(noValueOp.getTypeValue());
    if (failed(typeAttr)) {
      return ComptimeError::semanticError(op, context)
             << "type is not comptime-known";
    }
    auto cslTypeAttr = dyn_cast<csl::TypeAttr>(*typeAttr);
    if (!cslTypeAttr) {
      return ComptimeError::semanticError(op, context)
             << "type operand is not a type";
    }

    // ========== PHASE 2: Type Inference ==========
    Type resultType = cslTypeAttr.getValue();

    // ========== PHASE 3: Comptime Evaluation ==========
    // NoValueOp always succeeds at comptime
    return (TypedAttr)csl::NoValueAttr::get(resultType);
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // evaluation always succeeds
    return true;
  }
};

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

struct GlobalOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<GlobalOpComptimeModel,
                                                   GlobalOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto globalOp = cast<GlobalOp>(op);

    // ========== PHASE 1: Validation ==========
    Type varType = Type();
    Type initType = Type();

    if (globalOp.getIsConst() && !globalOp.hasInitializer()) {
      if (globalOp.getIsUnboundParam()) {
        return ComptimeError::semanticError(op, context)
               << "param '" << globalOp.getSymName()
               << "' has no binding or default value provided";
      }
      return ComptimeError::semanticError(op, context)
             << "const global must have an initializer";
    }

    if (globalOp.getDynamicType()) {
      auto typeAttrResult =
          context.getOrLoadComptimeRValue(globalOp.getDynamicType());
      if (failed(typeAttrResult)) {
        return ComptimeError::semanticError(op, context)
               << "global type is not comptime-known";
      }
      if (!mlir::isa<parsed_csl::TypeAttr>(*typeAttrResult)) {
        return ComptimeError::semanticError(op, context)
               << "specified global type is not a type";
      }
      auto typeAttr = mlir::cast<parsed_csl::TypeAttr>(*typeAttrResult);
      varType = typeAttr.getValue();
    }

    // Evaluate the init region (if present)
    TypedAttr castedInitValue;
    if (globalOp.hasInitializer()) {
      ControlFlowOutcomes outcome =
          context.evaluateRegion(globalOp.getInitRegion(),
                                 /*isRuntimeConditional=*/false,
                                 /*requiresComptime=*/true);

      if (outcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                                   op->getContext());

      if (outcome.mayYield.empty()) {
        return ComptimeError::semanticError(op, context)
               << "global initializer did not yield a value";
      }

      if (outcome.mayYield.size() != 1) {
        return ComptimeError::semanticError(op, context)
               << "global initializer should yield exactly once";
      }
      auto yieldedValue = outcome.mayYield[0].getValue();
      if (!yieldedValue) {
        return ComptimeError::semanticError(op, context)
               << "initializer of global must be comptime-known";
      }

      initType = yieldedValue->getType();
      castedInitValue = *yieldedValue;
      if (varType && initType != varType) {
        InFlightDiagnostic diag = emitRemark(op->getLoc());
        auto castResult =
            csl::evaluateImplicitCast(castedInitValue, varType, &diag);
        if (failed(castResult)) {
          return ComptimeError::semanticError(op, context)
                 << "failed to cast initializer from '"
                 << prettyPrintType(initType) << "' to global type '"
                 << prettyPrintType(varType) << "'" << std::move(diag);
        }
        diag.abandon();
        castedInitValue = *castResult;
      }
    }

    if (!varType && !initType) {
      return ComptimeError::semanticError(op, context)
             << "global must have either a type or an initializer";
    }

    // Evaluate the linkname region (if present)
    Region &linknameRegion = globalOp.getLinknameRegion();
    if (!linknameRegion.empty()) {
      ControlFlowOutcomes linknameOutcome =
          context.evaluateRegion(linknameRegion, /*isRuntimeConditional=*/false,
                                 /*requiresComptime=*/true);
      if (linknameOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(linknameOutcome),
                                                   op->getContext());
      if (linknameOutcome.mayYield.empty())
        return ComptimeError::semanticError(op, context)
               << "linkname expression did not yield a value";
      auto yieldedValue = linknameOutcome.mayYield[0].getValue();
      if (!yieldedValue)
        return ComptimeError::semanticError(op, context)
               << "linkname must be comptime-known";
      auto strAttr = mlir::dyn_cast<ComptimeStringAttr>(*yieldedValue);
      if (!strAttr)
        return ComptimeError::semanticError(op, context)
               << "linkname must evaluate to a comptime_string";
      globalOp.setLinknameAttr(
          StringAttr::get(op->getContext(), strAttr.getValue()));
    }

    // Evaluate the align region (if present)
    Region &alignRegion = globalOp.getAlignRegion();
    if (!alignRegion.empty()) {
      ControlFlowOutcomes alignOutcome =
          context.evaluateRegion(alignRegion, /*isRuntimeConditional=*/false,
                                 /*requiresComptime=*/true);
      if (alignOutcome.hasFatalError())
        return ComptimeEvalResult::fromControlFlow(std::move(alignOutcome),
                                                   op->getContext());
      if (alignOutcome.mayYield.empty())
        return ComptimeError::semanticError(op, context)
               << "align expression did not yield a value";
      auto yieldedValue = alignOutcome.mayYield[0].getValue();
      if (!yieldedValue)
        return ComptimeError::semanticError(op, context)
               << "align must be comptime-known";
      if (auto intAttr = dyn_cast<IntegerAttr>(*yieldedValue)) {
        if (intAttr.getValue().isNegative()) {
          return ComptimeError::semanticError(op, context)
                 << "align value cannot be negative";
        }
        globalOp.setAlign(intAttr.getValue().getZExtValue());
      } else if (auto typeAttr =
                     dyn_cast<csl::ComptimeIntAttr>(*yieldedValue)) {
        if (typeAttr.getValue().isNegative()) {
          return ComptimeError::semanticError(op, context)
                 << "align value cannot be negative";
        }
        globalOp.setAlign(typeAttr.getInt());
      } else {
        return ComptimeError::semanticError(op, context)
               << "align must evaluate to an integer";
      }
    }

    // ========== PHASE 2: Type Inference ==========
    // GlobalOp has no SSA result, so we set the inferred type attribute
    // directly
    Type valueType = varType ? varType : initType;
    globalOp.setInferredType(LValueType::get(valueType, globalOp.getIsConst()));
    if (castedInitValue)
      globalOp.setInferredInitializerAttr(castedInitValue);

    // ========== PHASE 3: Comptime Evaluation ==========
    // Non-const globals are not comptime-known, but the declaration succeeds
    if (!globalOp.getIsConst()) {
      return success();
    }

    // fifo_dsds are a special case: they are comptime-known but must be
    // allocated by @allocate_fifo() in a global initializer. To keep them from
    // getting inlined into users, we treat this global as not comptime-known
    if (isa<FifoDsdType>(valueType)) {
      return success();
    }

    // Const globals: store the comptime value
    // We already have castedInitValue from validation above
    TypedAttr typedInitValue = castedInitValue;

    // Declare and store the global's value
    context.declareComptimeVariable(globalOp.getOperation());
    LValueDescriptor globalDescriptor(globalOp, {}, valueType);
    if (failed(context.storeLValue(globalDescriptor, typedInitValue))) {
      return ComptimeError::semanticError(op, context)
             << "failed to store comptime global value";
    }

    // If its a module reference and we are the first user, set the name of the
    // module
    if (auto moduleRef = dyn_cast<ModuleRefAttr>(typedInitValue)) {
      // Find the module op
      CompilationUnitOp compUnit = op->getParentOfType<CompilationUnitOp>();
      parsed_csl::ModuleOp moduleOp =
          findModuleById(compUnit, moduleRef.getModuleIdentifier());
      if (moduleOp && !moduleOp.getModuleName().has_value()) {
        moduleOp.setModuleName(globalOp.getSymName().str());
      }
    }

    // GlobalOp has no result - return success
    return success();
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // All scope-level globals require comptime evaluation (their init region
    // is evaluated at comptime)
    return true;
  }
};

//===----------------------------------------------------------------------===//
// GetSymbolOp
//===----------------------------------------------------------------------===//

struct GetSymbolOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<GetSymbolOpComptimeModel,
                                                   GetSymbolOp> {

  /// Derive the module_id from a symbol's parent. If the symbol lives inside
  /// a ModuleOp, return that module's identifier; otherwise std::nullopt.
  static std::optional<uint64_t> getModuleIdForSymbol(Operation *symbolOp) {
    if (auto moduleOp = symbolOp->getParentOfType<parsed_csl::ModuleOp>())
      return moduleOp.getModuleIdentifier();
    return std::nullopt;
  }

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto getSymbolOp = cast<GetSymbolOp>(op);
    MLIRContext *ctx = op->getContext();

    // Nearest-symbol lookup from the op's IR position (handles module scope
    // automatically since the op lives inside the right ModuleOp or
    // CompilationUnitOp).
    Operation *symbolOp = nullptr;
    if (auto moduleId = getSymbolOp.getLexicalModuleIdentifier()) {
      auto unit = op->getParentOfType<csl::CompilationUnitOp>();
      if (auto module = findModuleById(unit, static_cast<uint64_t>(*moduleId)))
        symbolOp =
            SymbolTable::lookupSymbolIn(module, getSymbolOp.getNameAttr());
    } else {
      symbolOp =
          SymbolTable::lookupNearestSymbolFrom(op, getSymbolOp.getNameAttr());
    }

    if (!symbolOp) {
      return ComptimeError::semanticError(op, context)
             << "undefined symbol '" << getSymbolOp.getName() << "'";
    }

    // --- GlobalOp path ---
    if (auto globalOp = dyn_cast<GlobalOp>(symbolOp)) {
      // Trigger on-demand evaluation. Always call evaluateGlobalOnDemand
      // (even if isOpAlive) because evaluateOperation marks ops alive eagerly
      // before their model finishes — so during circular dependencies, a
      // GlobalOp may be alive but not yet have its inferred type set.
      // evaluateGlobalOnDemand checks the cycle set first and returns early
      // if already fully evaluated.
      ControlFlowOutcomes outcome =
          context.getGlobalEvalContext()->evaluateGlobalOnDemand(globalOp);
      if (outcome.hasFatalError())
        return outcome.getFatalError();

      auto inferredType = globalOp.getInferredType();
      if (!inferredType) {
        return ComptimeError::semanticError(op, context)
               << "could not infer type for global '" << getSymbolOp.getName()
               << "'";
      }

      auto lvalueType = cast<LValueType>(*inferredType);
      LValueDescriptor descriptor(globalOp, {}, lvalueType.getValueType());
      return descriptor.resolveModuleIfNeeded(context);
    }

    // --- GenericFuncOp path ---
    if (auto genericFunc = dyn_cast<GenericFuncOp>(symbolOp)) {
      std::optional<uint64_t> moduleId = getModuleIdForSymbol(symbolOp);

      // For non-generic functions, evaluate the type region to get the
      // concrete type. Truly generic functions return the GenericFunctionType
      // marker and will be instantiated when evaluating the CallOp.
      if (!genericFunc.isActuallyGeneric()) {
        auto funcTypeResult = context.evaluateFuncType(genericFunc, {}, {});
        if (std::holds_alternative<ComptimeError>(funcTypeResult))
          return std::get<ComptimeError>(funcTypeResult);
        FunctionType concreteFuncType = std::get<FunctionType>(funcTypeResult);
        return (TypedAttr)csl::FuncRefAttr::get(
            ctx, moduleId, FlatSymbolRefAttr::get(genericFunc.getSymNameAttr()),
            concreteFuncType);
      }

      return (TypedAttr)csl::FuncRefAttr::get(
          ctx, moduleId, FlatSymbolRefAttr::get(genericFunc.getSymNameAttr()),
          GenericFunctionType::get(ctx));
    }

    return ComptimeError::semanticError(op, context)
           << "'" << getSymbolOp.getName()
           << "' is not a function or global variable";
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// TensorAccessOp
//===----------------------------------------------------------------------===//

struct TensorAccessOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<TensorAccessOpComptimeModel,
                                                   TensorAccessOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto tensorAccessOp = cast<TensorAccessOp>(op);
    MLIRContext *ctx = op->getContext();
    unsigned numDims = tensorAccessOp.getNumDimensions();

    // ========== PHASE 1: Validation ==========
    // 1. Validate base is an lvalue to an array, an lvalue or rvalue to
    //    a many-pointer, or a pointer to an array
    Value base = tensorAccessOp.getBase();
    auto baseTypeResult = context.getInferredType(base);
    if (failed(baseTypeResult)) {
      return ComptimeError::semanticError(op, context).withDependency(base)
             << "cannot infer base type";
    }
    Type baseType = *baseTypeResult;

    Type valueType = baseType;
    bool isLValue = false;
    if (auto lvalueType = dyn_cast<LValueType>(baseType)) {
      valueType = lvalueType.getValueType();
      isLValue = true;
    }

    bool isValidBase = false;
    if (isa<ArrayType>(valueType) && isLValue)
      isValidBase = true;
    else if (auto ptrTy = dyn_cast<PointerType>(valueType))
      isValidBase =
          ptrTy.getIsManyPtr() || isa<ArrayType>(ptrTy.getPointeeType());

    if (!isValidBase) {
      return ComptimeError::semanticError(op, context)
             << "tensor_access base must be an array or pointer, got "
             << prettyPrintType(valueType);
    }

    // 2. Validate that all extents are comptime-known integers
    SmallVector<TypedAttr> extentValues;
    for (Value extent : tensorAccessOp.getExtents()) {
      auto extentValue = context.getOrLoadComptimeRValue(extent);
      if (failed(extentValue)) {
        return ComptimeError::semanticError(op, context).withDependency(extent)
               << "tensor_access extent must be comptime-known";
      }
      extentValues.push_back(*extentValue);

      auto extentType = context.getInferredRValueType(extent);
      if (failed(extentType) || !csl::isAnyIntegerType(*extentType)) {
        return ComptimeError::semanticError(op, context)
               << "tensor_access extent must be an integer type";
      }
    }

    // 3. Set up induction variable types as comptime_int (symbolic indices)
    Block &body = tensorAccessOp.getBody().front();
    Type comptimeIntType = ComptimeIntType::get(ctx);
    for (BlockArgument arg : body.getArguments()) {
      context.setInferredTypeDirectly(arg, comptimeIntType);
    }

    // ========== PHASE 2: Type Inference ==========
    Type ptrType;
    if (auto pointerType = dyn_cast<PointerType>(valueType)) {
      // Pointer type: either many-pointer or single pointer to array.
      // In both cases, the pointer type itself is the DSD base_address type.
      ptrType = pointerType;
    } else if (auto arrayType = dyn_cast<ArrayType>(valueType)) {
      // Create pointer type to the array (not to element type!)
      // e.g., for [10]u16, we want *[10]u16, not [*]u16
      assert(isLValue);
      bool isConst = cast<LValueType>(baseType).getIsConst();
      ptrType = PointerType::get(arrayType, /*isConst=*/isConst,
                                 /*isArray=*/false);
    } else {
      llvm_unreachable("base should be array or pointer after validation");
    }

    // Build anonymous struct type for DSD descriptor
    SmallVector<StringAttr> resultFieldNames;
    SmallVector<Type> resultFieldTypes;

    // base_address: pointer to the array
    resultFieldNames.push_back(StringAttr::get(ctx, "base_address"));
    resultFieldTypes.push_back(ptrType);

    // offset: i16 (linear offset)
    resultFieldNames.push_back(StringAttr::get(ctx, "offset"));
    resultFieldTypes.push_back(IntegerType::get(ctx, 16, IntegerType::Signed));

    // stride: i8 (1D) or tuple of i16 (multi-D)
    resultFieldNames.push_back(StringAttr::get(ctx, "stride"));
    if (numDims == 1) {
      resultFieldTypes.push_back(IntegerType::get(ctx, 8, IntegerType::Signed));
    } else {
      SmallVector<Type> strideTypes(
          numDims, IntegerType::get(ctx, 16, IntegerType::Signed));
      resultFieldTypes.push_back(StructType::get(ctx, strideTypes));
    }

    // extent: u16 (1D) or tuple of u16 (multi-D)
    resultFieldNames.push_back(StringAttr::get(ctx, "extent"));
    if (numDims == 1) {
      resultFieldTypes.push_back(
          IntegerType::get(ctx, 16, IntegerType::Unsigned));
    } else {
      SmallVector<Type> extentTypes(
          numDims, IntegerType::get(ctx, 16, IntegerType::Unsigned));
      resultFieldTypes.push_back(StructType::get(ctx, extentTypes));
    }

    Type resultType = StructType::get(ctx, resultFieldNames, resultFieldTypes);

    // ========== PHASE 3: Comptime Evaluation ==========
    // 1. Evaluate the body region. Induction variables are not given comptime
    // values since they are symbolic — analyzeTensorAccess uses them as affine
    // dimensions. We only annotate their type so that body ops can infer types.
    // Body ops that depend on IVs (e.g. i+1) will fail to evaluate at comptime,
    // but that is fine: analyzeTensorAccess builds affine exprs by walking the
    // def-use chain directly rather than relying on comptime values.
    // NOTE: body ops that define constants not depending on IVs (e.g. globals,
    // literals) will still evaluate correctly and be visible to
    // buildAffineExprFromValue via evalContext.
    Region &bodyRegion = tensorAccessOp.getBody();
    for (BlockArgument iv : tensorAccessOp.getInductionVars()) {
      context.setInferredTypeDirectly(iv, comptimeIntType);
    }
    ControlFlowOutcomes outcome = context.evaluateRegionSuppressed(bodyRegion);

    if (outcome.hasFatalError())
      return ComptimeEvalResult::fromControlFlow(std::move(outcome),
                                                 op->getContext());

    // 2. Construct base address attribute
    TypedAttr baseAddress;
    if (isa<ArrayType>(valueType)) {
      // Array base: take address of the global array variable.
      // &array gives *[N]T which is the DSD base address.
      FailureOr<LValueDescriptor> baseLValue = context.getComptimeLValue(base);
      if (failed(baseLValue) || !baseLValue->isGlobalBased()) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(base)
               << "tensor_access base is not a comptime-known global";
      }
      GlobalOp globalOp = baseLValue->getBaseGlobal();
      std::optional<uint64_t> moduleId = std::nullopt;
      if (auto module = globalOp->getParentOfType<parsed_csl::ModuleOp>()) {
        moduleId = module.getModuleIdentifier();
      }
      auto baseLValueType = cast<LValueType>(baseType);
      baseAddress = GlobalAddrAttr::get(GlobalRefAttr::get(
          ctx, moduleId, FlatSymbolRefAttr::get(globalOp.getSymNameAttr()),
          baseLValueType));
    } else {
      // Pointer base: the pointer value IS the address. Load the comptime
      // value of the pointer directly (e.g. a GlobalAddrAttr from the calling
      // module that was bound to this param).
      auto baseValue = context.getOrLoadComptimeRValue(base);
      if (failed(baseValue)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(base)
               << "tensor_access pointer base is not comptime-known";
      }
      baseAddress = *baseValue;
    }

    // 4. Build the result struct
    SmallVector<StringAttr> fieldNames;
    SmallVector<TypedAttr> fieldValues;

    // base_address
    fieldNames.push_back(StringAttr::get(ctx, "base_address"));
    fieldValues.push_back(baseAddress);

    // 5. Convert extents to u16 values for stride calculation
    SmallVector<uint16_t> extentU16Values;
    for (auto extentAttr : extentValues) {
      auto castResult = csl::evaluateCast(extentAttr, comptimeIntType);
      if (failed(castResult)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "extent is not castable to integer";
      }
      auto comptimeInt = cast<ComptimeIntAttr>(*castResult);
      extentU16Values.push_back(
          static_cast<uint16_t>(comptimeInt.getValue().getZExtValue()));
    }

    // 6. Get array shape and element type from base type
    SmallVector<int64_t> arrayShape;
    Type elementType;
    if (auto arrayType = dyn_cast<ArrayType>(valueType)) {
      arrayShape = SmallVector<int64_t>(arrayType.getShape());
      elementType = arrayType.getElementType();
    } else if (auto pointerType = dyn_cast<PointerType>(valueType)) {
      Type pointeeType = pointerType.getPointeeType();
      if (auto arrayType = dyn_cast<ArrayType>(pointeeType)) {
        arrayShape = SmallVector<int64_t>(arrayType.getShape());
        elementType = arrayType.getElementType();
      } else {
        elementType = pointeeType;
      }
    }
    // For many-pointer, arrayShape stays empty (treated as 1D)

    // 7. Analyze tensor access to get strides and offset
    auto analysisResult = analyzeTensorAccess(
        tensorAccessOp, extentU16Values, arrayShape, elementType, &context);
    if (failed(analysisResult)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "non-affine index expression in tensor_access";
    }
    TensorAccessInfo &info = *analysisResult;

    // offset: i16
    fieldNames.push_back(StringAttr::get(ctx, "offset"));
    fieldValues.push_back(IntegerAttr::get(
        IntegerType::get(ctx, 16, IntegerType::Signed), info.offset));

    // stride: i8 (1D) or tuple of i16 (multi-D)
    // Strides are already in innermost-first order from analyzeTensorAccess
    fieldNames.push_back(StringAttr::get(ctx, "stride"));
    if (numDims == 1) {
      fieldValues.push_back(
          IntegerAttr::get(IntegerType::get(ctx, 8, IntegerType::Signed),
                           static_cast<int8_t>(info.strides[0])));
    } else {
      SmallVector<TypedAttr> strideAttrValues;
      for (int16_t s : info.strides) {
        strideAttrValues.push_back(IntegerAttr::get(
            IntegerType::get(ctx, 16, IntegerType::Signed), s));
      }
      fieldValues.push_back(StructValueAttr::get(ctx, strideAttrValues,
                                                 /*def_location=*/{}, {}));
    }

    // extent: u16 (1D) or tuple of u16 (multi-D)
    fieldNames.push_back(StringAttr::get(ctx, "extent"));
    if (numDims == 1) {
      fieldValues.push_back(
          IntegerAttr::get(IntegerType::get(ctx, 16, IntegerType::Unsigned),
                           extentU16Values[0]));
    } else {
      SmallVector<TypedAttr> extentTypedValues;
      for (uint16_t e : extentU16Values) {
        extentTypedValues.push_back(IntegerAttr::get(
            IntegerType::get(ctx, 16, IntegerType::Unsigned), e));
      }
      fieldValues.push_back(StructValueAttr::get(ctx, extentTypedValues,
                                                 /*def_location=*/{}, {}));
    }

    return cast<TypedAttr>(StructValueAttr::get(ctx, fieldValues,
                                                /*def_location=*/{},
                                                fieldNames));
  }
};

//===----------------------------------------------------------------------===//
// TensorAccessYieldOp
//===----------------------------------------------------------------------===//

struct TensorAccessYieldOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<
          TensorAccessYieldOpComptimeModel, TensorAccessYieldOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto yieldOp = cast<TensorAccessYieldOp>(op);

    // ========== PHASE 1: Validation ==========
    // Validate indices are integer types
    for (Value index : yieldOp.getIndices()) {
      auto indexType = context.getInferredRValueType(index);
      if (failed(indexType)) {
        return ComptimeError::semanticError(op, context).withDependency(index)
               << "cannot infer index type";
      }
      if (!csl::isAnyIntegerType(*indexType)) {
        return ComptimeError::semanticError(op, context)
               << "tensor_access index must be an integer type, got "
               << prettyPrintType(*indexType);
      }
    }

    // ========== PHASE 2 & 3: No result type, signal yield ==========
    // Just signal yield - the parent TensorAccessOp constructs the result
    return ComptimeEvalResult::yieldWith(csl::VoidAttr::get(op->getContext()));
  }
};

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

struct InlineAsmOpComptimeModel
    : public ComptimeEvaluableOpInterfaceBaseModel<InlineAsmOpComptimeModel,
                                                   InlineAsmOp> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto asmOp = cast<InlineAsmOp>(op);

    // ========== PHASE 1 & 2: Validation ==========

    // Validate asm_string is a comptime string
    auto asmStrType = context.getInferredRValueType(asmOp.getAsmString());
    if (failed(asmStrType)) {
      return ComptimeError::semanticError(op, context)
                 .withDependency(asmOp.getAsmString())
             << "cannot infer type of asm string";
    }
    if (!mlir::isa<ComptimeStringType>(*asmStrType)) {
      return ComptimeError::semanticError(op, context)
             << "asm string must be a comptime_string, got "
             << prettyPrintType(*asmStrType);
    }

    // Helper to validate a range of operands are comptime strings.
    // Returns a non-empty optional with an error on failure.
    auto validateComptimeStrings =
        [&](ValueRange values,
            StringRef desc) -> std::optional<ComptimeEvalResult> {
      for (Value v : values) {
        auto vType = context.getInferredRValueType(v);
        if (failed(vType)) {
          return ComptimeEvalResult(
              ComptimeError::semanticError(op, context).withDependency(v)
              << "cannot infer type of asm " << desc);
        }
        if (!mlir::isa<ComptimeStringType>(*vType)) {
          return ComptimeEvalResult(ComptimeError::semanticError(op, context)
                                    << "asm " << desc
                                    << " must be a comptime_string, got "
                                    << prettyPrintType(*vType));
        }
      }
      return std::nullopt;
    };

    // Validate all comptime string operands (constraints and clobbers)
    if (auto err = validateComptimeStrings(asmOp.getOutputConstraints(),
                                           "output constraint"))
      return *err;
    if (auto err = validateComptimeStrings(asmOp.getInputConstraints(),
                                           "input constraint"))
      return *err;
    if (auto err = validateComptimeStrings(asmOp.getClobbers(), "clobber"))
      return *err;

    // Validate outputs are mutable lvalues
    for (Value output : asmOp.getOutputs()) {
      auto outputType = context.getInferredType(output);
      if (failed(outputType)) {
        return ComptimeError::semanticError(op, context).withDependency(output)
               << "cannot infer type of asm output";
      }
      auto lvalueType = llvm::dyn_cast<LValueType>(*outputType);
      if (!lvalueType) {
        return ComptimeError::semanticError(op, context)
               << "asm output must be an lvalue, got "
               << prettyPrintType(*outputType);
      }
      if (lvalueType.getIsConst()) {
        return ComptimeError::semanticError(op, context)
               << "asm output must be a mutable lvalue";
      }
    }

    // Validate inputs have rvalue types
    for (Value input : asmOp.getInputs()) {
      auto inputType = context.getInferredRValueType(input);
      if (failed(inputType)) {
        return ComptimeError::semanticError(op, context).withDependency(input)
               << "cannot infer type of asm input";
      }
    }

    // ========== PHASE 3: Resolve comptime string operands ==========
    // Extract resolved string values and store as inferred attributes
    // so the conversion pass can use them.

    auto resolveString = [&](Value v, StringRef desc) -> FailureOr<StringRef> {
      auto attr = context.getOrLoadComptimeRValue(v);
      if (failed(attr)) {
        return failure();
      }
      auto strAttr = mlir::dyn_cast<ComptimeStringAttr>(*attr);
      if (!strAttr) {
        return failure();
      }
      return strAttr.getValue();
    };

    auto resolveStrings =
        [&](ValueRange values,
            StringRef desc) -> FailureOr<SmallVector<Attribute>> {
      SmallVector<Attribute> result;
      for (Value v : values) {
        auto s = resolveString(v, desc);
        if (failed(s)) {
          return failure();
        }
        result.push_back(StringAttr::get(op->getContext(), *s));
      }
      return result;
    };

    // Resolve asm string
    auto asmStr = resolveString(asmOp.getAsmString(), "asm string");
    if (failed(asmStr)) {
      return ComptimeError::semanticError(op, context)
             << "asm string must be comptime-known";
    }
    asmOp.setInferredAsmStringAttr(StringAttr::get(op->getContext(), *asmStr));

    // Resolve output constraints
    auto outConstraints =
        resolveStrings(asmOp.getOutputConstraints(), "output constraint");
    if (failed(outConstraints)) {
      return ComptimeError::semanticError(op, context)
             << "asm output constraints must be comptime-known";
    }
    asmOp.setInferredOutputConstraintsAttr(
        ArrayAttr::get(op->getContext(), *outConstraints));

    // Resolve input constraints
    auto inConstraints =
        resolveStrings(asmOp.getInputConstraints(), "input constraint");
    if (failed(inConstraints)) {
      return ComptimeError::semanticError(op, context)
             << "asm input constraints must be comptime-known";
    }
    asmOp.setInferredInputConstraintsAttr(
        ArrayAttr::get(op->getContext(), *inConstraints));

    // Resolve clobbers
    auto clobberStrs = resolveStrings(asmOp.getClobbers(), "clobber");
    if (failed(clobberStrs)) {
      return ComptimeError::semanticError(op, context)
             << "asm clobbers must be comptime-known";
    }
    asmOp.setInferredClobbersAttr(
        ArrayAttr::get(op->getContext(), *clobberStrs));

    // Asm is inherently runtime, no comptime result
    return ComptimeError::comptimeFailure(op, context)
           << "inline asm is not comptime-evaluable";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::parsed_csl::registerComptimeModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ParsedCSLDialect *dialect) {
    ConstantOp::attachInterface<ConstantOpComptimeModel>(*ctx);
    VariableOp::attachInterface<VariableOpComptimeModel>(*ctx);
    GlobalOp::attachInterface<GlobalOpComptimeModel>(*ctx);
    GetSymbolOp::attachInterface<GetSymbolOpComptimeModel>(*ctx);
    ReadOp::attachInterface<ReadOpComptimeModel>(*ctx);
    LoadOp::attachInterface<LoadOpComptimeModel>(*ctx);
    LocOp::attachInterface<LocOpComptimeModel>(*ctx);
    SyntaxErrorOp::attachInterface<SyntaxErrorOpComptimeModel>(*ctx);
    DeferredErrorOp::attachInterface<DeferredErrorOpComptimeModel>(*ctx);
    AssignOp::attachInterface<AssignOpComptimeModel>(*ctx);
    AddressOfOp::attachInterface<AddressOfOpComptimeModel>(*ctx);
    DereferenceOp::attachInterface<DereferenceOpComptimeModel>(*ctx);
    BinaryOp::attachInterface<BinaryOpComptimeModel>(*ctx);
    UnaryOp::attachInterface<UnaryOpComptimeModel>(*ctx);
    CastOp::attachInterface<CastOpComptimeModel>(*ctx);
    SelectOp::attachInterface<SelectOpComptimeModel>(*ctx);
    SwitchOp::attachInterface<SwitchOpComptimeModel>(*ctx);
    ShortCircuitOp::attachInterface<ShortCircuitOpComptimeModel>(*ctx);
    BuiltinTypeOp::attachInterface<BuiltinTypeOpComptimeModel>(*ctx);
    PtrTypeOp::attachInterface<PtrTypeOpComptimeModel>(*ctx);
    FuncTypeOp::attachInterface<FuncTypeOpComptimeModel>(*ctx);
    ArrayTypeOp::attachInterface<ArrayTypeOpComptimeModel>(*ctx);
    CallOp::attachInterface<CallOpComptimeModel>(*ctx);
    ComptimeExprOp::attachInterface<ComptimeExprOpComptimeModel>(*ctx);
    ComptimeBlockOp::attachInterface<ComptimeBlockOpComptimeModel>(*ctx);
    LayoutBlockOp::attachInterface<LayoutBlockOpComptimeModel>(*ctx);
    ReturnOp::attachInterface<ReturnOpComptimeModel>(*ctx);
    SuspendOp::attachInterface<SuspendOpComptimeModel>(*ctx);
    AwaitOp::attachInterface<AwaitOpComptimeModel>(*ctx);
    YieldOp::attachInterface<YieldOpComptimeModel>(*ctx);
    IfOp::attachInterface<IfOpComptimeModel>(*ctx);
    ForOp::attachInterface<ForOpComptimeModel>(*ctx);
    WhileOp::attachInterface<WhileOpComptimeModel>(*ctx);
    LabeledBlockOp::attachInterface<LabeledBlockOpComptimeModel>(*ctx);
    BreakOp::attachInterface<BreakOpComptimeModel>(*ctx);
    ContinueOp::attachInterface<ContinueOpComptimeModel>(*ctx);
    NamedStructTypeOp::attachInterface<NamedStructTypeOpComptimeModel>(*ctx);
    EnumTypeOp::attachInterface<EnumTypeOpComptimeModel>(*ctx);
    ConstructTupleOp::attachInterface<ConstructTupleOpComptimeModel>(*ctx);
    ConstructStructOp::attachInterface<ConstructStructOpComptimeModel>(*ctx);
    ConstructArrayOp::attachInterface<ConstructArrayOpComptimeModel>(*ctx);
    FieldAccessOp::attachInterface<FieldAccessOpComptimeModel>(*ctx);
    IndexAccessOp::attachInterface<IndexAccessOpComptimeModel>(*ctx);
    BuiltinCallOp::attachInterface<BuiltinCallOpComptimeModel>(*ctx);
    NoValueOp::attachInterface<NoValueOpComptimeModel>(*ctx);
    TensorAccessOp::attachInterface<TensorAccessOpComptimeModel>(*ctx);
    TensorAccessYieldOp::attachInterface<TensorAccessYieldOpComptimeModel>(
        *ctx);
    InlineAsmOp::attachInterface<InlineAsmOpComptimeModel>(*ctx);
  });
}
