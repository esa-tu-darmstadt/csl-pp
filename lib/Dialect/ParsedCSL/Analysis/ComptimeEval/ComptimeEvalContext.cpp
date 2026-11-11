//===- ComptimeEvalContext.cpp - Comptime Evaluation Context ------------===//
//
// Implementation of compile-time evaluation context.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeFunctionEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Remarks.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::parsed_csl;

#define DEBUG_TYPE "comptime-eval"

//===----------------------------------------------------------------------===//
// Value Queries
//===----------------------------------------------------------------------===//

bool ComptimeEvalContext::isComptimeKnown(Value v,
                                          bool voidAlwaysComptimeKnown) const {
  if (voidAlwaysComptimeKnown) {
    // Void and structs of only void have a single value, which is always
    // comptime-known.
    if (auto inferredType = getInferredRValueType(v);
        succeeded(inferredType) && csl::isVoidLikeType(*inferredType)) {
      return true;
    }
  }

  // Check local storage first
  auto it = comptimeValues.find(v);
  if (it != comptimeValues.end()) {
    if (it->second.hasRValue())
      return true;
    if (it->second.hasLValue())
      return isLValueComptime(it->second.getLValue());
    // Type-only or error: not comptime-known
    return false;
  }

  return false;
}

FailureOr<TypedAttr>
ComptimeEvalContext::getComptimeRValue(Value v,
                                       bool voidAlwaysComptimeKnown) const {
  if (voidAlwaysComptimeKnown) {
    // Void and structs of only void have a single value, which is always
    // comptime-known.
    if (auto inferredType = getInferredRValueType(v);
        succeeded(inferredType) && csl::isVoidLikeType(*inferredType)) {
      return csl::getVoidLikeValue(*inferredType);
    }
  }

  // Check local storage first
  auto it = comptimeValues.find(v);
  if (it != comptimeValues.end()) {
    if (it->second.hasRValue()) {
      return it->second.getRValue();
    }
    // Value evaluated but has no result, or failed
    return failure();
  }

  // Check parent context if available
  if (parentContext) {
    return parentContext->getComptimeRValue(v);
  }

  // Value is not comptime-known
  return failure();
}

FailureOr<LValueDescriptor>
ComptimeEvalContext::getComptimeLValue(Value v) const {
  // Check local storage first
  auto it = comptimeValues.find(v);
  if (it != comptimeValues.end()) {
    if (it->second.hasLValue()) {
      return it->second.getLValue();
    }
    // Value evaluated but has no result, or failed
    return failure();
  }

  // Check parent context if available
  if (parentContext) {
    return parentContext->getComptimeLValue(v);
  }

  // Value is not comptime-known
  return failure();
}

FailureOr<ComptimeError>
ComptimeEvalContext::getComptimeEvalFailure(Value v) const {
  // Check local storage first
  auto it = comptimeValues.find(v);
  if (it != comptimeValues.end()) {
    if (it->second.isError()) {
      return it->second.getError();
    }
    // Value succeeded, no error to return
    return failure();
  }

  // Check parent context if available
  if (parentContext) {
    return parentContext->getComptimeEvalFailure(v);
  }

  // Value has no recorded failure
  return failure();
}

//===----------------------------------------------------------------------===//
// Type Queries
//===----------------------------------------------------------------------===//

FailureOr<Type> ComptimeEvalContext::getInferredType(Value v) const {
  // Check local storage first
  auto it = inferredTypes.find(v);
  if (it != inferredTypes.end()) {
    return it->second;
  }

  // Check parent context if available
  if (parentContext) {
    return parentContext->getInferredType(v);
  }

  // No inferred type available
  return failure();
}

FailureOr<Type>
ComptimeEvalContext::getGlobalValueType(GlobalOp globalOp) const {
  // Try to get the type from the explicit type operand
  if (Value typeOperand = globalOp.getDynamicType()) {
    FailureOr<TypedAttr> typeValue = getComptimeRValue(typeOperand);
    if (succeeded(typeValue)) {
      if (auto typeAttr = mlir::dyn_cast<csl::TypeAttr>(*typeValue))
        return typeAttr.getValue();
    }
  }

  // Fall back to the initializer's inferred type
  if (auto initAttr = globalOp.getInferredInitializer()) {
    return initAttr->getType();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Variable Lookup
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ComptimeEvalContext::loadVariable(Operation *varOp) const {
  // Check local storage first
  auto it = comptimeVarValues.find(varOp);
  if (it != comptimeVarValues.end()) {
    return it->second;
  }

  // Check parent context if available
  if (parentContext) {
    return parentContext->loadVariable(varOp);
  }

  // Variable is not comptime-known
  return failure();
}

bool ComptimeEvalContext::isLValueComptime(
    const LValueDescriptor &descriptor) const {
  // Get the base operation from the descriptor
  Operation *baseOp = descriptor.getBaseOperation();

  // Check if the base variable is declared in this context
  if (declaredInThisContext_.contains(baseOp)) {
    return true;
  }

  // Recurse to parent context if available
  if (parentContext)
    return parentContext->isLValueComptime(descriptor);

  // Not declared in any context
  return false;
}

FailureOr<TypedAttr>
ComptimeEvalContext::loadLValue(const LValueDescriptor &descriptor) const {
  if (descriptor.isModuleBased()) {
    // Return a reference to the module itself
    ModuleOp moduleOp = descriptor.getBaseModule();
    return (TypedAttr)ModuleRefAttr::get(moduleOp->getContext(),
                                         moduleOp.getModuleIdentifier());
  }

  // Get the base value: either from comptimeParamValues (for function params)
  // or from the variable table (for VariableOp / GlobalOp).
  FailureOr<TypedAttr> baseValue;
  if (descriptor.isParamBased()) {
    BlockArgument blockArg = descriptor.getBlockArgument();
    auto it = comptimeParamValues.find(blockArg);
    if (it == comptimeParamValues.end()) {
      if (parentContext)
        return parentContext->loadLValue(descriptor);
      return failure();
    }
    baseValue = it->second;
  } else {
    baseValue = loadVariable(descriptor.getBaseOperation());
  }
  if (failed(baseValue))
    return failure();

  // Apply the access path to navigate through struct/array fields.
  TypedAttr currentValue = *baseValue;
  for (const AccessPathElement &element : descriptor.getAccessPath()) {
    if (auto *fieldAccess = std::get_if<FieldAccess>(&element)) {
      auto structValue = dyn_cast<csl::StructValueAttr>(currentValue);
      if (!structValue)
        return failure();
      Attribute fieldValue = structValue.getField(fieldAccess->fieldName);
      if (!fieldValue)
        return failure();
      currentValue = cast<TypedAttr>(fieldValue);
    } else if (auto *indexAccess = std::get_if<IndexAccess>(&element)) {
      if (auto structValue = dyn_cast<csl::StructValueAttr>(currentValue)) {
        Attribute fieldValue = structValue.getField(indexAccess->index);
        if (!fieldValue)
          return failure();
        currentValue = cast<TypedAttr>(fieldValue);
      } else if (auto arrayValue =
                     dyn_cast<csl::ArrayValueAttr>(currentValue)) {
        TypedAttr elemValue = arrayValue.getElement(indexAccess->index);
        if (!elemValue)
          return failure();
        currentValue = elemValue;
      } else {
        return failure();
      }
    }
  }
  return currentValue;
}

FailureOr<TypedAttr>
ComptimeEvalContext::getOrLoadComptimeRValue(Value v) const {
  // First check if there's a direct rvalue stored for this value.
  FailureOr<TypedAttr> rvalue = getComptimeRValue(v);
  if (succeeded(rvalue)) {
    return rvalue;
  }

  // Try to get as lvalue and load it
  // Note: In the new model, lvalue/rvalue distinction is tracked via
  // LValueDescriptor, not via the static SSA type.
  FailureOr<LValueDescriptor> lvalueDescr = getComptimeLValue(v);
  if (succeeded(lvalueDescr)) {
    return loadLValue(*lvalueDescr);
  }

  return failure();
}

LogicalResult
ComptimeEvalContext::storeLValue(const LValueDescriptor &descriptor,
                                 TypedAttr value) {
  assert(!descriptor.isModuleBased() && "Cannot store to module-based lvalue");

  // Handle param-based descriptors: allow initial store at binding time
  if (descriptor.isParamBased()) {
    assert(descriptor.isDirectReference() &&
           "Cannot store to param-based lvalue with access path");
    comptimeParamValues[descriptor.getBlockArgument()] = value;
    return success();
  }

  // Get the base operation from the descriptor
  Operation *baseOp = descriptor.getBaseOperation();

  // If the access path is empty, just store directly to the variable
  if (descriptor.isDirectReference()) {
    return storeVariable(baseOp, value);
  }

  // Otherwise, load the base variable and apply copy-on-write updates
  auto baseValue = loadVariable(baseOp);
  if (failed(baseValue)) {
    return failure(); // Base variable is not comptime-known
  }

  // Apply copy-on-write updates recursively along the access path
  TypedAttr updatedValue =
      applyAccessPathUpdate(*baseValue, descriptor.getAccessPath(), value, 0);
  if (!updatedValue) {
    return failure(); // Failed to apply updates
  }

  // Store the updated value back to the base variable
  return storeVariable(baseOp, updatedValue);
}

// Helper function to recursively apply copy-on-write updates along an access
// path
TypedAttr ComptimeEvalContext::applyAccessPathUpdate(
    TypedAttr currentValue, ArrayRef<AccessPathElement> accessPath,
    TypedAttr newValue, size_t pathIndex) const {
  // Base case: we've reached the end of the path, return the new value
  if (pathIndex >= accessPath.size()) {
    return newValue;
  }

  const AccessPathElement &element = accessPath[pathIndex];

  // Field access (struct only)
  if (auto *fieldAccess = std::get_if<FieldAccess>(&element)) {
    auto structValue = dyn_cast<csl::StructValueAttr>(currentValue);
    if (!structValue) {
      return nullptr; // Expected struct value
    }

    // Get the field index
    auto fieldIndex = structValue.getFieldIndex(fieldAccess->fieldName);
    if (failed(fieldIndex)) {
      return nullptr; // Field not found
    }

    // Recursively update the field value
    TypedAttr oldFieldValue =
        cast<TypedAttr>(structValue.getField(*fieldIndex));
    TypedAttr newFieldValue = applyAccessPathUpdate(oldFieldValue, accessPath,
                                                    newValue, pathIndex + 1);
    if (!newFieldValue) {
      return nullptr; // Failed to update
    }

    // Return a new struct with the updated field
    return csl::StructValueAttr::updateField(structValue, *fieldIndex,
                                             newFieldValue);
  }

  // Index access (struct or array)
  if (auto *indexAccess = std::get_if<IndexAccess>(&element)) {
    // Try StructValueAttr first
    if (auto structValue = dyn_cast<csl::StructValueAttr>(currentValue)) {
      // Recursively update the field value
      TypedAttr oldFieldValue =
          cast<TypedAttr>(structValue.getField(indexAccess->index));
      TypedAttr newFieldValue = applyAccessPathUpdate(oldFieldValue, accessPath,
                                                      newValue, pathIndex + 1);
      if (!newFieldValue) {
        return nullptr; // Failed to update
      }

      // Return a new struct with the updated field
      return csl::StructValueAttr::updateField(structValue, indexAccess->index,
                                               newFieldValue);
    }

    // Try ArrayValueAttr
    if (auto arrayValue = dyn_cast<csl::ArrayValueAttr>(currentValue)) {
      // Recursively update the element value
      TypedAttr oldElemValue = arrayValue.getElement(indexAccess->index);
      if (!oldElemValue) {
        return nullptr; // Index out of bounds
      }
      TypedAttr newElemValue = applyAccessPathUpdate(oldElemValue, accessPath,
                                                     newValue, pathIndex + 1);
      if (!newElemValue) {
        return nullptr; // Failed to update
      }

      // Return a new array with the updated element
      return csl::ArrayValueAttr::updateElement(arrayValue, indexAccess->index,
                                                newElemValue);
    }

    return nullptr; // Expected struct or array value
  }

  // Invalid accessor
  return nullptr;
}

LogicalResult ComptimeEvalContext::storeVariable(Operation *varOp,
                                                 TypedAttr value) {
  // Check write barrier first
  if (failed(checkWriteBarrier(varOp))) {
    return failure();
  }

  // Store in local storage
  comptimeVarValues[varOp] = value;
  return success();
}

void ComptimeEvalContext::declareComptimeVariable(Operation *varOp) {
  declaredInThisContext_.insert(varOp);
}

bool ComptimeEvalContext::isComptimeVariableDeclaredHere(
    Operation *varOp) const {
  return declaredInThisContext_.contains(varOp);
}

bool ComptimeEvalContext::isOpAlive(Operation *op) const {
  if (aliveOps_.contains(op))
    return true;
  if (parentContext)
    return parentContext->isOpAlive(op);
  return false;
}

//===----------------------------------------------------------------------===//
// Comptime Evaluation
//===----------------------------------------------------------------------===//

void ComptimeEvalContext::reportComptimeEvalFailure(
    MLIRContext *ctx, const ComptimeError &failure) const {
  // Track error count on the global context for LSP mode
  if (auto *global = getGlobalEvalContext())
    global->incrementErrorCount();

  // Emit the main diagnostic
  failure.emit(ctx);

  // Recursively report failures from dependencies
  for (Value dep : failure.getDependencies()) {
    auto maybeFailure = getComptimeEvalFailure(dep);
    if (succeeded(maybeFailure))
      reportComptimeEvalFailure(ctx, *maybeFailure);
  }
}

std::variant<ComptimeError, TypedAttr>
ComptimeEvalContext::evaluateComptimeCall(
    FuncOp func, llvm::ArrayRef<TypedAttr> args) const {
  assert(funcEvaluator_ &&
         "ComptimeEvalContext has no function evaluator assigned");

  return funcEvaluator_->evaluateCall(
      func, args,
      parentContext ? parentContext : const_cast<ComptimeEvalContext *>(this));
}

std::variant<ComptimeError, TypedAttr>
ComptimeEvalContext::evaluateComptimeCallViaTemp(
    GenericFuncOp generic, FunctionType funcType,
    ArrayRef<TypedAttr> comptimeArgValues, ArrayRef<Type> anytypeArgTypes,
    ArrayRef<TypedAttr> runtimeArgValues) const {
  assert(funcEvaluator_ &&
         "ComptimeEvalContext has no function evaluator assigned");

  return funcEvaluator_->evaluateComptimeCallViaTemp(
      generic, funcType, comptimeArgValues, anytypeArgTypes, runtimeArgValues,
      parentContext ? parentContext : const_cast<ComptimeEvalContext *>(this));
}

/// Extract FuncRefAttr from a comptime value (FuncRefAttr or FuncAddrAttr).
static std::optional<csl::FuncRefAttr> tryExtractFuncRef(TypedAttr value) {
  if (auto funcRef = mlir::dyn_cast<csl::FuncRefAttr>(value))
    return funcRef;
  if (auto funcAddr = mlir::dyn_cast<csl::FuncAddrAttr>(value))
    return funcAddr.getFuncRef();
  return std::nullopt;
}

/// Look up a function by FuncRefAttr, handling cross-module lookups.
/// Helper to look up a function by FuncRefAttr from an arbitrary operation,
/// finding the CompilationUnitOp automatically.
static Operation *lookupFuncByRef(csl::FuncRefAttr funcRef, Operation *from) {
  auto cu = from->getParentOfType<csl::CompilationUnitOp>();
  if (!cu)
    return nullptr;
  auto result = lookupFuncByRefAttr(funcRef, cu);
  if (failed(result))
    return nullptr;
  return std::visit([](auto op) -> Operation * { return op; }, *result);
}

std::variant<ComptimeError, ComptimeEvalContext::InstantiationResult>
ComptimeEvalContext::getOrInstantiateCallee(CallOp callOp) {
  // First check if already instantiated
  FailureOr<InstantiationResult> existing = getInstantiatedCallee(callOp);
  if (succeeded(existing)) {
    return *existing;
  }

  // Try to get the callee as a comptime-known function reference
  FailureOr<TypedAttr> calleeValue =
      getOrLoadComptimeRValue(callOp.getCallee());
  std::optional<csl::FuncRefAttr> funcRef;
  if (succeeded(calleeValue))
    funcRef = tryExtractFuncRef(*calleeValue);

  if (!funcRef) {
    // Indirect call - callee is not a comptime-known function reference.
    // Cannot instantiate anything for indirect calls.
    return ComptimeError::comptimeFailure(callOp, *this)
           << "indirect call - cannot instantiate callee";
  }

  // Direct call - look up the function
  Operation *op = lookupFuncByRef(*funcRef, callOp);
  if (!op) {
    return ComptimeError::semanticError(callOp, *this)
           << "function '" << funcRef->getSymbolName().getValue()
           << "' not found";
  }

  // If it's already a FuncOp (non-generic), return it
  if (auto instantiatedFunc = mlir::dyn_cast<FuncOp>(op)) {
    return InstantiationResult{instantiatedFunc, {}};
  }

  // It's a GenericFuncOp - need to instantiate
  auto genericFuncOp = mlir::dyn_cast<GenericFuncOp>(op);
  if (!genericFuncOp) {
    return ComptimeError::semanticError(callOp, *this)
           << "'" << funcRef->getSymbolName().getValue()
           << "' is not a function";
  }

  // Validate argument count
  if (callOp.getArgs().size() != genericFuncOp.getNumParams()) {
    return ComptimeError::semanticError(callOp, *this)
           << "expected " << genericFuncOp.getNumParams()
           << " argument(s), but got " << callOp.getArgs().size();
  }

  // Collect comptime arg values and anytype arg types
  SmallVector<Type> anytypeArgTypes;
  SmallVector<TypedAttr> comptimeArgValues;

  for (auto [idx, arg] : llvm::enumerate(callOp.getArgs())) {
    if (llvm::is_contained(genericFuncOp.getComptimeParams(), idx)) {
      FailureOr<TypedAttr> argValue = getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        return ComptimeError::semanticError(callOp, *this).withDependency(arg)
               << "comptime argument is not comptime-known";
      }
      comptimeArgValues.push_back(*argValue);
    }

    if (llvm::is_contained(genericFuncOp.getAnytypeParams(), idx)) {
      FailureOr<Type> argType = getInferredRValueType(arg);
      if (failed(argType)) {
        return ComptimeError::semanticError(callOp, *this)
               << "cannot infer type of argument";
      }
      anytypeArgTypes.push_back(*argType);
    }
  }

  // Save comptime param indices before potential erase of GenericFuncOp
  SmallVector<int32_t> comptimeParamIndices(
      genericFuncOp.getComptimeParams().begin(),
      genericFuncOp.getComptimeParams().end());

  auto funcOpResult =
      getOrInstantiateFunc(genericFuncOp, comptimeArgValues, anytypeArgTypes);
  if (std::holds_alternative<ComptimeError>(funcOpResult)) {
    return std::get<ComptimeError>(funcOpResult);
  }
  FuncOp instantiatedFuncOp = std::get<FuncOp>(funcOpResult);

  return InstantiationResult{instantiatedFuncOp,
                             std::move(comptimeParamIndices)};
}

FailureOr<ComptimeEvalContext::InstantiationResult>
ComptimeEvalContext::getInstantiatedCallee(TypedAttr funcRefOrAddr,
                                           csl::CompilationUnitOp cu) const {
  std::optional<csl::FuncRefAttr> funcRef = tryExtractFuncRef(funcRefOrAddr);
  if (!funcRef) {
    return failure();
  }

  // Look up the function
  Operation *op = lookupFuncByRef(*funcRef, cu);
  if (!op) {
    return failure();
  }

  // If it's already a FuncOp, return it
  if (auto instantiatedFunc = mlir::dyn_cast<FuncOp>(op)) {
    return InstantiationResult{instantiatedFunc, {}};

    // If it's a GenericFuncOp, make sure it has no comptime/anytype params
  }

  auto genericFuncOp = mlir::dyn_cast<GenericFuncOp>(op);
  if (!genericFuncOp)
    return failure();

  if (genericFuncOp.isActuallyGeneric())
    return failure();

  // Get the instantiated function
  // Look up in the instantiation cache (via GlobalEvalContext)
  const GlobalEvalContext *globalCtx = getGlobalEvalContext();
  if (!globalCtx)
    return failure();

  FuncOp instantiatedFunc =
      globalCtx->lookupInstantiatedFunc(genericFuncOp, {}, {});
  if (!instantiatedFunc)
    return failure();
  return InstantiationResult{instantiatedFunc, {}};
}

FailureOr<ComptimeEvalContext::InstantiationResult>
ComptimeEvalContext::getInstantiatedCallee(CallOp callOp) const {
  // Try to get the callee as a comptime-known function reference
  FailureOr<TypedAttr> calleeValue = getComptimeRValue(callOp.getCallee());
  std::optional<csl::FuncRefAttr> funcRef;
  if (succeeded(calleeValue))
    funcRef = tryExtractFuncRef(*calleeValue);

  // If we don't have a funcRef, this is an indirect call
  if (!funcRef)
    return failure();

  // Direct call - look up the function
  Operation *op = lookupFuncByRef(*funcRef, callOp);
  if (!op)
    return failure();

  // If it's already a FuncOp, return it
  if (auto instantiatedFunc = mlir::dyn_cast<FuncOp>(op)) {
    return InstantiationResult{instantiatedFunc, {}};
  }

  // It's a GenericFuncOp - look up cached instantiation
  auto genericFuncOp = mlir::dyn_cast<GenericFuncOp>(op);
  if (!genericFuncOp)
    return failure();

  // Collect comptime arg values and anytype arg types to form cache key
  SmallVector<TypedAttr> comptimeArgValues;
  SmallVector<Type> anytypeArgTypes;
  for (auto [idx, arg] : llvm::enumerate(callOp.getArgs())) {
    if (llvm::is_contained(genericFuncOp.getComptimeParams(), idx)) {
      FailureOr<TypedAttr> argValue = getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        llvm::report_fatal_error(
            "getInstantiatedCallee: comptime argument value not found. "
            "This likely means getInstantiatedCallee was called on the wrong "
            "context. Use ComptimeEvaluator::getInstantiatedCallee() instead.");
      }
      comptimeArgValues.push_back(*argValue);
    }
    if (llvm::is_contained(genericFuncOp.getAnytypeParams(), idx)) {
      FailureOr<Type> argType = getInferredRValueType(arg);
      if (failed(argType)) {
        llvm::report_fatal_error(
            "getInstantiatedCallee: anytype argument type not found. "
            "This likely means getInstantiatedCallee was called on the wrong "
            "context. Use ComptimeEvaluator::getInstantiatedCallee() instead.");
      }
      anytypeArgTypes.push_back(*argType);
    }
  }

  // Look up in the instantiation cache (via GlobalEvalContext)
  const GlobalEvalContext *globalCtx = getGlobalEvalContext();
  if (!globalCtx)
    return failure();

  FuncOp instantiatedFunc = globalCtx->lookupInstantiatedFunc(
      genericFuncOp, comptimeArgValues, anytypeArgTypes);
  if (!instantiatedFunc)
    return failure();

  SmallVector<int32_t> comptimeParamIndices(
      genericFuncOp.getComptimeParams().begin(),
      genericFuncOp.getComptimeParams().end());

  return InstantiationResult{instantiatedFunc, std::move(comptimeParamIndices)};
}

ControlFlowOutcomes ComptimeEvalContext::evaluateModule(ModuleOp module) {
  // Default implementation returns a fatal error - modules can only be
  // imported at global scope, not within functions.
  return ControlFlowOutcomes::fatal(
      ComptimeError::semanticError(module.getOperation(), *this)
      << "modules can only be imported at global scope");
}

ComptimeEvalResult ComptimeEvalContext::evaluateOperation(Operation *op) {
  // Use different debug type for verbose logging within this method
#undef DEBUG_TYPE
#define DEBUG_TYPE "comptime-eval-verbose"

  // Mark this op as alive (visited during evaluation)
  aliveOps_.insert(op);

  // Use the ComptimeEvaluableOpInterface to evaluate the operation
  auto evalInterface = dyn_cast<ComptimeEvaluableOpInterface>(op);
  if (!evalInterface) {
    op->emitError() << "does not implement ComptimeEvaluableOpInterface";
    llvm_unreachable(
        "operation does not implement ComptimeEvaluableOpInterface");
  }

  LLVM_DEBUG(
      {
        logger().startLine()
            << "----------------------------------------------------\n";
        logger().startLine() << "Evaluating operation at comptime: ";
        op->print(logger().getOStream(), OpPrintingFlags().skipRegions());
        logger().getOStream() << "\n";
        logger().startLine()
            << "- Context: " << getContextDescription() << "\n";
      });

  // Determine if this operation must evaluate at comptime
  bool mustEval = opMustEvaluateAtComptime(op);

  LLVM_DEBUG(logger().startLine()
             << "- Operation " << (mustEval ? "must" : "may not")
             << " evaluate at comptime\n");

  // Call the unified evaluate() method
  ComptimeEvalResult result = evalInterface.evaluate(*this);

  // Handle replacement/erasure: the op was erased or replaced by new ops.
  // Remove the dangling pointer from aliveOps_ and return immediately.
  if (result.opWasReplaced) {
    aliveOps_.erase(op);
    return result;
  }

  // Get the inferred type from the result
  Type resultType = result.getType();

  // Cache the inferred type if available
  if (op->getNumResults() > 0 && resultType) {
    inferredTypes[op->getResult(0)] = resultType;
  }

  LLVM_DEBUG(logger().startLine()
             << "- Inferred result type: " << (resultType ? resultType : Type())
             << "\n");

  // Handle fatal error case (semantic, execution, required failure)
  if (result.isError()) {
    LLVM_DEBUG(logger().startLine()
               << "- Fatal error: " << result.getError() << "\n");
    // Fatal errors are always propagated — no policy decision needed
    return result;
  }

  // Handle comptime failure case
  if (result.isComptimeFailure()) {
    LLVM_DEBUG(logger().startLine() << "- Comptime failure: "
                                    << result.getComptimeFailure() << "\n");

    // Verify that comptime failure has a type if op has results
    if (op->getNumResults() > 0 && !resultType) {
      llvm::report_fatal_error(
          "ComptimeFailure for operation with results must have inferred type "
          "attached via withType() for runtime fallback");
    }

    // Apply requirement policy: fatalize if comptime is required
    if (mustEval) {
      ComptimeError fatal = result.getComptimeFailure().fatalize();
      return ComptimeEvalResult(std::move(fatal));
    }

    // Optional context: record the failure for dependency tracking
    if (op->getNumResults() > 0) {
      comptimeValues.emplace_or_assign(op->getResult(0), result);
    }

    return result;
  }

  // Handle success case
  LLVM_DEBUG(logger().startLine() << "- Evaluation succeeded\n");

  // Record the successful result if the operation has results
  bool isNoValueResult =
      result.hasRValue() && mlir::isa<csl::NoValueAttr>(result.getRValue());
  if (op->getNumResults() > 0 && (result.hasLValue() || result.hasRValue()) &&
      !isNoValueResult) {
    comptimeValues.emplace_or_assign(op->getResult(0), result);
  }

  return result;

#undef DEBUG_TYPE
#define DEBUG_TYPE "comptime-eval"
}

ControlFlowOutcomes ComptimeEvalContext::evaluateRegion(Region &region) {
  // Use different debug type for verbose logging within this method
#undef DEBUG_TYPE
#define DEBUG_TYPE "comptime-eval-verbose"
  LLVM_DEBUG(
      {
        logger().startLine()
            << "====================================================\n";
        logger().startLine() << "Evaluating region at comptime\n";
        logger().indent(2);
      });

  ControlFlowOutcomes accumulated = ControlFlowOutcomes::fallthrough();
  assert(region.getBlocks().size() <= 1 &&
         "multiple blocks in region are not supported");
  if (region.getBlocks().empty()) {
    LLVM_DEBUG(logger().startLine() << "Region is empty, adding empty block\n");
    region.emplaceBlock();
  }
  Block &block = region.front();

  auto it = block.begin();
  while (it != block.end()) {
    // If no paths can reach this operation, stop. Erase all ops after this
    // op, and place an unreachable op if needed
    if (!accumulated.mayFallthrough) {
      if (this->isComptimeContext()) {
        // In comptime context, regions may be re-evaluated (e.g. for-loop
        // unrolling), so we must not mutate the IR. Just stop evaluating.
        break;
      }
      LLVM_DEBUG({
        logger().startLine() << "Erasing unreachable operation: ";
        it->print(logger().getOStream(), OpPrintingFlags().skipRegions());
        logger().getOStream() << "\n";
      });
      it->dropAllUses();
      it = block.getOperations().erase(it);
      continue;
    }

    Operation &op = *it;

    // Save the previous position so we can revisit it if the op is
    // replaced/erased. If it == block.begin(), there is no predecessor;
    // we'll restart from block.begin() in that case.
    bool isFirst = (it == block.begin());
    auto prevIt = isFirst ? it : std::prev(it);

    // Evaluate the operation
    ComptimeEvalResult result = evaluateOperation(&op);

    // Handle replacement/erasure: the op was erased or replaced by new
    // ops at the same position. Revisit from where the original op was.
    if (result.opWasReplaced) {
      it = isFirst ? block.begin() : std::next(prevIt);
      continue;
    }

    ++it;

    // Fatal errors propagate immediately (unless in LSP mode)
    if (result.isError()) {
      if (getGlobalEvalContext() &&
          getGlobalEvalContext()->getOptions().lspMode) {
        // In LSP mode, skip fatal errors and continue analyzing.
        // The op's result has no inferred type, so any dependent op
        // will also fail, propagating naturally.
        reportComptimeEvalFailure(op.getContext(), result.getError());
        continue;
      }
      LLVM_DEBUG(logger().unindent(2));
      return ControlFlowOutcomes::fatal(std::move(result.getError()));
    }

    // Compose this operation's control flow with the accumulated outcomes.
    // Only fallthrough paths get this op's outcomes (sequential composition).
    accumulated = accumulated.followedBy(result.controlFlow);
  }

  // Add an unreachable or fallthrough terminator if block is not already
  // properly terminated. Function body regions are handled by their callers
  // (analyzeFunctionBody inserts ReturnOp/UnreachableOp).
  if (!isa<FuncOp>(region.getParentOp())) {
    bool needsTerminator =
        block.empty() || !block.back().hasTrait<OpTrait::PseudoTerminator>();
    if (needsTerminator && !this->isComptimeContext()) {
      OpBuilder builder(&block, block.end());
      Operation *terminator;
      if (accumulated.mayFallthrough) {
        terminator = YieldOp::create(builder, region.getLoc(), Value());
      } else {
        terminator = UnreachableOp::create(builder, region.getLoc());
      }
      markOpAlive(terminator);
    }
  }
  LLVM_DEBUG(logger().unindent(2));
  return accumulated;

#undef DEBUG_TYPE
#define DEBUG_TYPE "comptime-eval"
}

FailureOr<bool> ComptimeEvalContext::isComptimeOnlyFunction(FuncOp func) const {

  return llvm::any_of(
             func.getFunctionType().getInputs(),
             [](Type type) { return csl::isComptimeOnlyType(type); }) ||
         llvm::any_of(func.getFunctionType().getResults(),
                      [](Type type) { return csl::isComptimeOnlyType(type); });
}

bool ComptimeEvalContext::opMustEvaluateAtComptime(Operation *op) const {
  auto evalInterface = dyn_cast<ComptimeEvaluableOpInterface>(op);
  if (!evalInterface) {
    return false;
  }

  return isComptimeContext() ||
         evalInterface.requiresComptimeBySemantics(*this);
}
