//===- ParsedCSLDialect.cpp - ParsedCSL Dialect ----------------*- C++ -*-===//
//
// Implementation of the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "parsedcsl-dialect"

using namespace mlir;
using namespace mlir::parsed_csl;

//===----------------------------------------------------------------------===//
// ParsedCSL Dialect
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLDialect.cpp.inc"

void ParsedCSLDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLOps.cpp.inc"
      >();
}

void ParsedCSLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLTypes.cpp.inc"
      >();
}

void ParsedCSLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLAttrs.cpp.inc"
      >();
}

// Type parsing and printing are generated automatically by tablegen

//===----------------------------------------------------------------------===//
// Constant Materializer
//===----------------------------------------------------------------------===//

Operation *ParsedCSLDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {

  // For TypedAttr (value attributes)
  if (auto typedAttr = mlir::dyn_cast<TypedAttr>(value)) {
    // Special case: TypeAttr needs BuiltinTypeOp
    if (auto typeAttr = dyn_cast<csl::TypeAttr>(typedAttr)) {
      return parsed_csl::BuiltinTypeOp::create(builder, loc, typeAttr);
    }

    // FuncRefAttr and GlobalRefAttr are materialized as ConstantOp.
    // This allows the AttrTypeReplacer to find and update them during
    // InlineModulesPass.
    if (mlir::isa<csl::FuncRefAttr, csl::GlobalRefAttr>(typedAttr)) {
      return parsed_csl::ConstantOp::create(builder, loc, typedAttr);
    }

    // If the result type is LValueType, use parsed_csl.lvalue
    if (mlir::isa<csl::LValueType>(type)) {
      llvm::report_fatal_error(
          "ParsedCSLDialect::materializeConstant does not support LValueType "
          "constants");
    }

    // Otherwise, use parsed_csl.constant for rvalues
    return parsed_csl::ConstantOp::create(builder, loc, typedAttr);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "ParsedCSLDialect::materializeConstant failed for value: "
                 << value << " and type: " << type << "\n";
  });

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Note: Custom format directives moved to CSL dialect
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLTypes.cpp.inc"

// void ParsedCSLDialect::printType(::mlir::Type type,
//                                  ::mlir::DialectAsmPrinter &printer) const {
//   llvm_unreachable("ParsedCSLDialect::printType should not be called");
// }

// ::mlir::Type
// ParsedCSLDialect::parseType(::mlir::DialectAsmParser &parser) const {
//   return {};
// }

//===----------------------------------------------------------------------===//
// Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLAttrs.cpp.inc"

// void ParsedCSLDialect::printAttribute(
//     ::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const {
//   llvm_unreachable("ParsedCSLDialect::printAttribute should not be called");
// }

// ::mlir::Attribute
// ParsedCSLDialect::parseAttribute(::mlir::DialectAsmParser &parser,
//                                  ::mlir::Type type) const {
//   return {};
// }

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

// Type constraint implementations
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLTypeConstraints.cpp.inc"

} // namespace parsed_csl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Symbol Resolution
//===----------------------------------------------------------------------===//

parsed_csl::ModuleOp parsed_csl::findModuleById(csl::CompilationUnitOp compUnit,
                                                uint64_t moduleId) {
  ModuleOp result;
  walkTopLevelDecls<ModuleOp>(compUnit, [&](ModuleOp mod) -> WalkResult {
    if (static_cast<uint64_t>(mod.getModuleIdentifier()) == moduleId) {
      result = mod;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Helper: find a module op by its identifier and look up a symbol inside it.
static Operation *lookupSymbolInModule(uint64_t moduleId,
                                       FlatSymbolRefAttr symbolName,
                                       csl::CompilationUnitOp unit) {
  auto targetModule = parsed_csl::findModuleById(unit, moduleId);
  if (!targetModule)
    return nullptr;
  return SymbolTable::lookupSymbolIn(targetModule, symbolName.getValue());
}

FailureOr<parsed_csl::GlobalOp>
parsed_csl::lookupGlobalByRefAttr(csl::GlobalRefAttr attr,
                                  csl::CompilationUnitOp cu) {
  Operation *op;
  if (attr.getModuleIdentifier())
    op = lookupSymbolInModule(*attr.getModuleIdentifier(), attr.getSymbolName(),
                              cu);
  else
    op = SymbolTable::lookupSymbolIn(cu, attr.getSymbolName());
  if (!op)
    return failure();
  auto globalOp = mlir::dyn_cast<parsed_csl::GlobalOp>(op);
  if (!globalOp)
    return failure();
  return globalOp;
}

FailureOr<std::variant<parsed_csl::FuncOp, parsed_csl::GenericFuncOp>>
parsed_csl::lookupFuncByRefAttr(csl::FuncRefAttr attr,
                                csl::CompilationUnitOp cu) {
  Operation *op;
  if (attr.getModuleIdentifier())
    op = lookupSymbolInModule(*attr.getModuleIdentifier(), attr.getSymbolName(),
                              cu);
  else
    op = SymbolTable::lookupSymbolIn(cu, attr.getSymbolName());
  if (!op)
    return failure();
  if (auto funcOp = mlir::dyn_cast<parsed_csl::FuncOp>(op))
    return std::variant<parsed_csl::FuncOp, parsed_csl::GenericFuncOp>(funcOp);
  if (auto genericFuncOp = mlir::dyn_cast<parsed_csl::GenericFuncOp>(op))
    return std::variant<parsed_csl::FuncOp, parsed_csl::GenericFuncOp>(
        genericFuncOp);
  return failure();
}

FailureOr<parsed_csl::FuncOp>
parsed_csl::lookupFuncByCoroFrame(csl::CoroFrameType coroFrameTy,
                                  csl::CompilationUnitOp cu) {
  Operation *op;
  if (coroFrameTy.getModuleIdentifier())
    op = lookupSymbolInModule(*coroFrameTy.getModuleIdentifier(),
                              coroFrameTy.getSymbolName(), cu);
  else
    op = SymbolTable::lookupSymbolIn(cu, coroFrameTy.getSymbolName());
  if (!op)
    return failure();
  if (auto funcOp = mlir::dyn_cast<parsed_csl::FuncOp>(op))
    return funcOp;
  return failure();
}

FailureOr<parsed_csl::FuncOp>
parsed_csl::lookupFuncByFuture(csl::FutureType futureTy,
                               csl::CompilationUnitOp cu) {
  Operation *op;
  if (futureTy.getModuleIdentifier())
    op = lookupSymbolInModule(*futureTy.getModuleIdentifier(),
                              futureTy.getSymbolName(), cu);
  else
    op = SymbolTable::lookupSymbolIn(cu, futureTy.getSymbolName());
  if (!op)
    return failure();
  if (auto funcOp = mlir::dyn_cast<parsed_csl::FuncOp>(op))
    return funcOp;
  return failure();
}

bool parsed_csl::isUnreadOperand(Operation *op, unsigned operandIdx) {
  // The source of a read or load, the target of an assignment, the operand of
  // `&`, and the operand of the transparent location wrapper.
  if (isa<ReadOp, LoadOp, AssignOp, AddressOfOp, LocOp>(op))
    return operandIdx == 0;

  // The base of an access: a field or element of a variable is itself a
  // location, e.g. the target of `a[i] = x`.
  if (isa<FieldAccessOp, IndexAccessOp, TensorAccessOp>(op))
    return operandIdx == 0;

  if (auto builtinCallOp = dyn_cast<BuiltinCallOp>(op)) {
    std::optional<csl::BuiltinKind> kind = builtinCallOp.resolveBuiltin();
    return kind && !csl::BuiltinDescriptor::get(*kind, op->getContext())
                        .readsArgument(operandIdx);
  }

  if (auto asmOp = dyn_cast<InlineAsmOp>(op)) {
    OperandRange outputs = asmOp.getOutputs();
    unsigned begin = outputs.getBeginOperandIndex();
    return operandIdx >= begin && operandIdx < begin + outputs.size();
  }

  return false;
}

bool parsed_csl::mayDenoteLocation(Value value) {
  if (mlir::isa<BlockArgument>(value))
    return true;
  Operation *def = value.getDefiningOp();
  if (isa<VariableOp, GetSymbolOp, FieldAccessOp, IndexAccessOp,
          DereferenceOp, LocOp, DeferredErrorOp>(def))
    return true;
  if (auto builtinCallOp = dyn_cast<BuiltinCallOp>(def)) {
    std::optional<csl::BuiltinKind> kind = builtinCallOp.resolveBuiltin();
    return kind && csl::BuiltinDescriptor::get(*kind, def->getContext())
                       .mayReturnLocation();
  }
  return false;
}

LogicalResult parsed_csl::verifyExplicitReads(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      if (!mayDenoteLocation(operand.get()) ||
          isUnreadOperand(op, operand.getOperandNumber()))
        continue;
      op->emitError() << "operand " << operand.getOperandNumber()
                      << " may be a location that is used as a value "
                         "without a read (internal error)";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

bool parsed_csl::operandExpectsLValue(
    Operation *op, unsigned operandIdx,
    function_ref<FailureOr<Type>(Value)> getType) {
  if (!isUnreadOperand(op, operandIdx))
    return false;

  // The base of an index access is a location if it is an array, whose
  // element is accessed in place, but a pointer variable is read and the
  // pointer indexed.
  if (isa<IndexAccessOp>(op)) {
    FailureOr<Type> baseType = getType(op->getOperand(0));
    assert(succeeded(baseType) &&
           "IndexAccessOp base should have inferred type");
    if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(*baseType))
      if (mlir::isa<csl::PointerType>(lvalueType.getValueType()))
        return false;
  }
  return true;
}