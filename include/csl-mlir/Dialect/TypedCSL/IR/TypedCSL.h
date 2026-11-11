//===- TypedCSL.h - TypedCSL Dialect ----------------------------*- C++ -*-===//
//
// Main header for the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_IR_TYPEDCSL_H
#define CSL_MLIR_DIALECT_TYPEDCSL_IR_TYPEDCSL_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// TypedCSL Dialect
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLDialect.h.inc"

// Note: Enums (BinaryOpKind, FloatKind, Direction) are now in CSL dialect
// They are available through the included CSL.h header

//===----------------------------------------------------------------------===//
// TypedCSL Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLTypes.h.inc"

//===----------------------------------------------------------------------===//
// TypedCSL Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLAttrs.h.inc"

//===----------------------------------------------------------------------===//
// TypedCSL Operations
//===----------------------------------------------------------------------===//

// Bring CSL enums and types into typed_csl namespace for TableGen-generated
// code
namespace mlir {
namespace typed_csl {
// Enums
using csl::BinaryOpKind;
using csl::BinaryOpKindAttr;
using csl::Direction;

// Types
using csl::ArrayType;
using csl::ColorType;
using csl::LValueType;
using csl::PointerType;
using csl::VoidType;

// Attributes
using csl::LValueAttr;

// Runtime types and attributes now use MLIR builtins:
// - Integer types: mlir::IntegerType (si8, ui16, si32, etc.)
// - Float types: mlir::Float16Type, mlir::Float32Type, mlir::BFloat16Type
// - Bool type: mlir::IntegerType with width 1 (i1)
// - Integer attributes: mlir::IntegerAttr
// - Float attributes: mlir::FloatAttr
// - Bool attributes: mlir::IntegerAttr for i1, mlir::BoolAttr for constants
} // namespace typed_csl
} // namespace mlir

//===----------------------------------------------------------------------===//
// TypedCSL Interfaces
//===----------------------------------------------------------------------===//

namespace mlir {
namespace typed_csl {
namespace detail {
/// Verify that a BreakingTerminatorOpInterface op addresses a valid target:
/// intermediate ops propagate breaking control flow and the target accepts
/// the terminator.
LogicalResult verifyBreakingTerminator(Operation *op);
} // namespace detail
} // namespace typed_csl
} // namespace mlir

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLInterfaces.h.inc"

#define GET_OP_CLASSES
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLOps.h.inc"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace typed_csl {

// Type constraint helpers
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLTypeConstraints.h.inc"

//===----------------------------------------------------------------------===//
// Region exit helpers
//
// A region exit is a terminator implementing RegionExitTerminatorOpInterface
// with a single target op: typed_csl.yield exits to its parent op, and
// BreakingTerminatorOpInterface ops exit to the ancestor they address.
//===----------------------------------------------------------------------===//

/// Return the op that receives control when `terminator` exits its region, or
/// nullptr if `terminator` is not a region exit.
Operation *getRegionExitTarget(Operation *terminator);

/// Invoke `callback` for every region exit nested in `op` that exits to `op`
/// or to one of its ancestors. The callback receives the terminator and its
/// 1-based nesting level relative to `op`'s regions.
void visitNestedRegionExits(
    Operation *op,
    function_ref<void(Operation *terminator, int level)> callback);

/// Collect the region exits nested in `op` that exit to `op` itself.
void collectNestedRegionExits(Operation *op,
                              SmallVectorImpl<Operation *> &exits);

//===----------------------------------------------------------------------===//
// Symbol reference helpers
//
// A reference to a function or a global is a `typed_csl.constant` holding a
// `#csl.func_ref` or `#csl.global_ref`.
//===----------------------------------------------------------------------===//

/// Create a reference to the function `name` of type `type`.
Value createFuncRef(OpBuilder &builder, Location loc, FunctionType type,
                    FlatSymbolRefAttr name);
Value createFuncRef(OpBuilder &builder, Location loc, FunctionType type,
                    StringRef name);

/// Create the lvalue of the global `name`, typed `type`.
Value createGlobalRef(OpBuilder &builder, Location loc, csl::LValueType type,
                      FlatSymbolRefAttr name);
Value createGlobalRef(OpBuilder &builder, Location loc, csl::LValueType type,
                      StringRef name);

/// The function reference `value` is, or null if it is none.
csl::FuncRefAttr getFuncRef(Value value);

/// The global reference `value` is, or null if it is none.
csl::GlobalRefAttr getGlobalRef(Value value);

} // namespace typed_csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_TYPEDCSL_IR_TYPEDCSL_H
