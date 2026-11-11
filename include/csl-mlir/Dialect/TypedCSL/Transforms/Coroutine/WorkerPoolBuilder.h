//===- WorkerPoolBuilder.h - Worker pool generation -*- C++ -*-===//
//
// Generates the shared worker pool used by the stackless coroutine
// (trampoline) backend's dynamic task allocation.
//
// === Overview ===
//
// At each `@await_parallel` site, the dispatch lowering claims fresh
// local task IDs via `@claim_local_task()`, runtime-binds the shared
// `__coro_worker` task to each claimed ID, fills the per-slot pool fields
// (callee frame ptr, callee fn ptr, pointer to caller's counter, task ID
// to activate on join), and `@activate`s the workers. The first callee
// runs inline on the caller's task — no worker needed for it. The caller
// emits a local `csl.variable u16` for the join counter; CoroFrameRewrite
// later spills it into the caller's coro frame, and workers receive its
// address to decrement on completion.
//
// === Pool Layout (struct of arrays) ===
//
//   var __coro_pool_frame:        [POOL_SIZE]([*]u8);
//   var __coro_pool_callee:       [POOL_SIZE](*const fn([*]u8) void);
//   var __coro_pool_counter_ptr:  [POOL_SIZE](*u16);
//   var __coro_pool_activate:     [POOL_SIZE]u16;
//
// Indexed by local_task_id. Pool size is fixed (covers the claimable
// range on both archs).
//
// === Shared worker trampoline ===
//
//   task __coro_worker() void {
//       const my_id    = @get_int(@current_task());
//       const frame    = __coro_pool_frame[my_id];
//       const callee   = __coro_pool_callee[my_id];
//       callee(frame);                       // suspend = nop.term
//       const ctr_ptr  = __coro_pool_counter_ptr[my_id];
//       const act_task = __coro_pool_activate[my_id];
//       @release_local_task(@current_task());
//       const new_count = ctr_ptr.* - 1;
//       ctr_ptr.* = new_count;
//       if (new_count == 0) {
//           @activate(@get_local_task_id(act_task));
//       }
//   }
//
// The worker trampoline is NOT bound at comptime — every binding happens
// at runtime at a dispatch site.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_WORKERPOOLBUILDER_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_WORKERPOOLBUILDER_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::typed_csl {

/// Information about a single async task entry point (user `@bind_*_task`
/// on an async function). Kept only for diagnostics — no IR transformation
/// is performed on these bindings; they pass through verbatim.
struct AsyncTaskBinding {
  FuncOp funcOp;
  BuiltinCallOp bindOp;
  uint16_t taskId;
  csl::BuiltinKind bindKind;
};

/// Information about the generated worker pool. Per-field SoA: each field
/// has its own top-level global; field metadata is keyed by field name.
struct WorkerPoolInfo {
  struct Field {
    FlatSymbolRefAttr globalName; // symbol ref to `__coro_pool_<name>`
    csl::ArrayType arrayTy;       // `[poolSize]<elementTy>`
    Type elementTy;               // element type of the field
  };

  llvm::StringMap<Field> fields; // keyed by "frame", "callee", "counter_ptr",
                                 // "activate"
  uint32_t poolSize = 0;

  /// Convenience accessors.
  Type frameElemTy() const { return fields.lookup("frame").elementTy; }
  Type calleeElemTy() const { return fields.lookup("callee").elementTy; }
  Type counterPtrElemTy() const {
    return fields.lookup("counter_ptr").elementTy;
  }
  Type activateElemTy() const { return fields.lookup("activate").elementTy; }
};

/// Analyzes async task bindings and emits the worker pool + shared
/// `__coro_worker` trampoline.
class WorkerPoolBuilder {
public:
  /// Walk the unit: collect async-function `@bind_*_task` bindings (for
  /// diagnostics) and determine whether any `@await_parallel` or offloaded
  /// `@await` exists in the unit.
  LogicalResult analyze(csl::CompilationUnitOp unit);

  /// Emit the SoA pool globals and the shared `__coro_worker` task.
  /// No-op if `!hasAsyncDispatch()`.
  ///
  /// \p slotAlign controls the frame pointer element type (1=u8, 2=u16).
  LogicalResult generate(csl::CompilationUnitOp unit, IRRewriter &rewriter,
                         unsigned slotAlign = 1);

  /// Whether the unit contains any async-dispatch site (`@await_parallel`
  /// or offloaded `@await`). If false, no pool/trampoline is emitted and
  /// no dispatch lowering is needed.
  bool hasAsyncDispatch() const { return hasAsyncDispatchFlag; }

  /// Get the pool info (valid after generate()).
  const WorkerPoolInfo &getPoolInfo() const { return pool; }

private:
  SmallVector<AsyncTaskBinding> bindings;
  WorkerPoolInfo pool;
  bool hasAsyncDispatchFlag = false;
};

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_WORKERPOOLBUILDER_H
