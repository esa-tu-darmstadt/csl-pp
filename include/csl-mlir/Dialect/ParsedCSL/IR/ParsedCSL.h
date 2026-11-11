//===- ParsedCSL.h - ParsedCSL Dialect --------------------------*- C++ -*-===//
//
// Main header for the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_PARSEDCSL_IR_PARSEDCSL_H
#define CSL_MLIR_DIALECT_PARSEDCSL_IR_PARSEDCSL_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <variant>

//===----------------------------------------------------------------------===//
// ParsedCSL Dialect
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLDialect.h.inc"

//===----------------------------------------------------------------------===//
// ParsedCSL Op Traits
//===----------------------------------------------------------------------===//
namespace mlir::parsed_csl {
class VariableOp;
namespace OpTrait {
template <typename ConcreteType>
class Expression : public ::mlir::OpTrait::TraitBase<ConcreteType, Expression> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Expression ops must have a single result
    if (op->getNumResults() != 1) {
      return op->emitError()
             << "operations with the ParsedCSL::Expression trait must have "
                "exactly one result";
    }
    return success();
  }
};
template <typename ConcreteType>
class Statement : public ::mlir::OpTrait::TraitBase<ConcreteType, Statement> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Statement ops must have no results
    if (op->getNumResults() != 0 && !mlir::isa<VariableOp>(op)) {
      return op->emitError()
             << "operations with the ParsedCSL::Statement trait must have no "
                "results";
    }
    return success();
  }
};

template <typename ConcreteType>
class ComptimeScope
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ComptimeScope> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getParentWithTrait<ComptimeScope>()) {
      // Ensure comptime scopes are not nested
      return op->emitError() << "'comptime' is redundant in comptime scope";
    }

    return success();
  }
};
template <typename ConcreteType>
class PseudoTerminator
    : public ::mlir::OpTrait::TraitBase<ConcreteType, PseudoTerminator> {};
} // namespace OpTrait
} // namespace mlir::parsed_csl

//===----------------------------------------------------------------------===//
// ParsedCSL Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLTypes.h.inc"

//===----------------------------------------------------------------------===//
// ParsedCSL Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLAttrs.h.inc"

//===----------------------------------------------------------------------===//
// ParsedCSL Interfaces
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLInterfaces.h.inc"

namespace mlir::parsed_csl {
/// Get the inferred type for a value using the InferredTypeInterface if
/// present. Returns failure if no inferred type is set or the operation does
/// not implement the interface.
FailureOr<Type> getInferredType(Value value);

/// Unwrap an lvalue type to get the underlying value type.
/// If the type is not an lvalue, returns it unchanged.
/// For example: !csl.lvalue<const si32> -> si32
inline Type unwrapLValueType(Type type) {
  if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(type))
    return lvalueType.getValueType();
  return type;
}

/// Get the inferred type for a value, automatically unwrapping lvalue types.
/// This is the rvalue version - use when you need the value type, not the
/// lvalue type.
inline FailureOr<Type> getInferredRValueType(Value value) {
  auto type = getInferredType(value);
  if (failed(type))
    return failure();
  return unwrapLValueType(*type);
}
} // namespace mlir::parsed_csl

//===----------------------------------------------------------------------===//
// ParsedCSL Operations
//===----------------------------------------------------------------------===//

// Bring CSL types and attributes into parsed_csl namespace for
// TableGen-generated code
namespace mlir {
namespace parsed_csl {
// Types from CSL dialect
using csl::ArrayType;
using csl::ColorType;
using csl::ComptimeFloatType;
using csl::ComptimeIntType;
using csl::ComptimeStringType;
using csl::PointerType;
using csl::TypeType;
using csl::VoidType;

// Attributes from CSL dialect
using csl::ComptimeFloatAttr;
using csl::ComptimeIntAttr;
using csl::LValueAttr;
using csl::TypeAttr;

// Runtime types and attributes now use MLIR builtins:
// - Integer types: mlir::IntegerType (si8, ui16, si32, etc.)
// - Float types: mlir::Float16Type, mlir::Float32Type, mlir::BFloat16Type
// - Bool type: mlir::IntegerType with width 1 (i1)
// - Integer attributes: mlir::IntegerAttr
// - Float attributes: mlir::FloatAttr
// - Bool attributes: mlir::IntegerAttr for i1, mlir::BoolAttr for constants

// Enums from CSL dialect
using csl::BinaryOpKind;
using csl::BinaryOpKindAttr;
using csl::Direction;

/// Whether `op` uses operand `operandIdx` without reading its value. Most
/// such operands take a location: the target of an assignment, the operand of
/// `&`, the base of a field, index or tensor access, the source of a read or
/// load, an output of inline asm, and builtin arguments such as the first one
/// of @field. The argument of @type_of is used only for its type. Every other
/// operand takes a value, which the parser reads with a ReadOp where the
/// source evaluates it. This depends only on the op, not on types, so it
/// holds right after parsing.
bool isUnreadOperand(Operation *op, unsigned operandIdx);

/// Whether `value` may denote a location, judged by the op that defines it
/// without knowing types: a variable, a parameter or other block argument, a
/// global symbol, a field, element or tensor access, a dereferenced pointer,
/// or a builtin result such as that of @field. Only such values need a
/// ReadOp where they are used as a value.
bool mayDenoteLocation(Value value);

/// Check that every operand that may denote a location is either read or
/// used without reading its value (see isUnreadOperand), i.e. that each
/// read of a location is explicit. Reports an internal error otherwise.
LogicalResult verifyExplicitReads(Operation *root);

/// Whether materialization keeps an lvalue operand instead of replacing it
/// by the value of a const variable. These are the location operands, except
/// the base of an index access that is a pointer variable: the pointer, not
/// the variable, is indexed, and the lowering reads it at the access.
/// `getType` provides the inferred type of a value. By default, it is read
/// from the `inferred_type` attributes, which are only populated once the
/// comptime evaluation is materialized.
bool operandExpectsLValue(
    Operation *op, unsigned operandIdx,
    function_ref<FailureOr<Type>(Value)> getType = getInferredType);
} // namespace parsed_csl
} // namespace mlir

#define GET_OP_CLASSES
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLOps.h.inc"

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

// Type constraint implementations
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLTypeConstraints.h.inc"

} // namespace parsed_csl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Top-Level Declaration Walking
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

/// Walk top-level declarations in a region, visiting only direct children
/// and recursing into nested ModuleOps. Unlike walk(), this does not descend
/// into function bodies, global init regions, etc.
///
/// The callback takes a specific op type and optionally returns WalkResult
/// for early termination. Modifying the IR during the walk is not supported;
/// collect ops into a container first if modification is needed.
///
/// Usage:
///   walkTopLevelDecls<FuncOp>(unit, [&](FuncOp f) { ... });
///   walkTopLevelDecls<FuncOp>(unit, [&](FuncOp f) -> WalkResult { ... });
template <typename OpT, typename FnT>
WalkResult walkTopLevelDecls(Region &region, FnT &&callback) {
  for (Operation &op : region.getOps()) {
    if (auto matched = mlir::dyn_cast<OpT>(op)) {
      if constexpr (std::is_convertible_v<decltype(callback(matched)),
                                          WalkResult>) {
        WalkResult result = callback(matched);
        if (result.wasInterrupted())
          return result;
      } else {
        callback(matched);
      }
    }
    if (auto modOp = mlir::dyn_cast<ModuleOp>(op)) {
      WalkResult result = walkTopLevelDecls<OpT>(modOp.getRegion(), callback);
      if (result.wasInterrupted())
        return result;
    }
  }
  return WalkResult::advance();
}

template <typename OpT, typename FnT>
WalkResult walkTopLevelDecls(csl::CompilationUnitOp unit, FnT &&callback) {
  return walkTopLevelDecls<OpT>(unit.getRegion(), callback);
}

/// Find a ModuleOp by its module identifier, searching only direct children
/// of the CompilationUnitOp and recursing into nested ModuleOps.
ModuleOp findModuleById(csl::CompilationUnitOp compUnit, uint64_t moduleId);

} // namespace parsed_csl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Symbol Resolution
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

/// Lookup a global variable by its GlobalRefAttr within the given compilation
/// unit
FailureOr<GlobalOp> lookupGlobalByRefAttr(csl::GlobalRefAttr attr,
                                          csl::CompilationUnitOp cu);

/// Lookup a function by its FuncRefAttr within the given compilation unit
FailureOr<std::variant<FuncOp, GenericFuncOp>>
lookupFuncByRefAttr(csl::FuncRefAttr attr, csl::CompilationUnitOp cu);

/// Lookup a function by the module identifier and symbol name stored in a
/// CoroFrameType. Returns the FuncOp if found, failure otherwise.
FailureOr<FuncOp> lookupFuncByCoroFrame(csl::CoroFrameType coroFrameTy,
                                        csl::CompilationUnitOp cu);

/// Lookup a function by the module identifier and symbol name stored in a
/// FutureType. Returns the FuncOp if found, failure otherwise.
FailureOr<FuncOp> lookupFuncByFuture(csl::FutureType futureTy,
                                     csl::CompilationUnitOp cu);

} // namespace parsed_csl
} // namespace mlir
#endif // CSL_MLIR_DIALECT_PARSEDCSL_IR_PARSEDCSL_H
