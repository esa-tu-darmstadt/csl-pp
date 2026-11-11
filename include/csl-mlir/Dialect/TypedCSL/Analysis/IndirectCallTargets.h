//===- IndirectCallTargets.h - Possible callees of indirect calls *- C++ -*-===//
//
// Conservative call-target analysis for `typed_csl.indirect_call`, in the
// spirit of clang's kcfi: a function pointer can only point to a function
// whose address is taken somewhere in the compilation unit, and since CSL
// cannot cast function pointers, the function's type must equal the type of
// the call.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_ANALYSIS_INDIRECTCALLTARGETS_H
#define CSL_MLIR_DIALECT_TYPEDCSL_ANALYSIS_INDIRECTCALLTARGETS_H

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::typed_csl {

/// Possible callees of every indirect call in a compilation unit.
///
/// A function's address is taken if a `#csl.func_ref` to it appears in any
/// attribute of the unit (constants, aggregates, global initializers),
/// except for a reference that is only passed to `@bind_*_task` or
/// `@export_symbol`: binding a task or exporting a symbol does not give the
/// program a pointer to the function.
///
/// An indirect call through a constant function address can only call that
/// function. Any other indirect call may call every address-taken function
/// whose type equals the call's function type.
class IndirectCallTargets {
public:
  /// Analyze `op`, a `csl.compilation_unit` or an op containing one.
  explicit IndirectCallTargets(Operation *op);

  /// The functions `call` may call, in source order. Empty if no function of
  /// the call's type has its address taken or `call` is outside the analyzed
  /// op.
  ArrayRef<FuncOp> getPossibleCallees(IndirectCallOp call) const;

  /// Whether the address of `func` is taken.
  bool isAddressTaken(FuncOp func) const;

private:
  /// Functions whose address is taken.
  DenseSet<Operation *> addressTaken;

  /// Address-taken functions grouped by function type, in source order.
  DenseMap<FunctionType, SmallVector<FuncOp>> addressTakenByType;

  /// Possible callees per indirect call.
  DenseMap<Operation *, SmallVector<FuncOp>> callees;
};

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_ANALYSIS_INDIRECTCALLTARGETS_H
