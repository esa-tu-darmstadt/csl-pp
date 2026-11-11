//===- FiberValidation.h - Reachability + validation for fiber sources ----===//
//
// Two related concerns kept together because the second consumes the first:
//
//   1. **Reachability.**  Which FuncOps in this `CompilationUnitOp` will end
//      up executing on a fiber stack?  Seed: every `async fn`/`async task`
//      (SymbolDCE has already removed any unused async fn by the time this
//      runs, so the seed set matches the user's actual async closure).
//      Frontier expansion through every direct `typed_csl.call` edge —
//      sync helpers, sync calls into extern fns, and the call ops that
//      produce futures for `@await`/`@await_parallel` are all the same op
//      from this analysis's perspective — and through every
//      `typed_csl.indirect_call` into its possible callees (see
//      IndirectCallTargets).
//
//   2. **Source validation.**  Errors at the user's source location if any
//      reachable function contains a construct whose stack reach we cannot
//      compute from the post-link ELF + the MLIR call graph:
//
//        * `is_extern` FuncOp in the reachable set — we have no body to
//          walk, so `.stack_sizes` only reveals the extern's own frame and
//          says nothing about anything it calls.
//        * Recursion — direct or indirect cycle through call edges with
//          neither `max_recursion_depth(N)` nor `stack_size(N)` to bound
//          it.
//        * Recursion through a callee of a multi-callee `@await_parallel`,
//          whose stack region has a fixed size.  No annotation helps.
//
//      The first two cases are silenced by an annotation at the offending
//      node (the user takes responsibility for the bound), which the
//      diagnostic names.  An indirect call without any possible callee is
//      only a warning: it cannot call anything that needs stack.  So is
//      `max_recursion_depth(N)` on a function that is not recursive.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERVALIDATION_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERVALIDATION_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/TypedCSL/Analysis/IndirectCallTargets.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::typed_csl {

/// Funcs that may execute on a fiber stack at runtime.
struct FiberReachableFuncs {
  /// Every function reachable from an async FuncOp.  Includes the async
  /// FuncOps themselves, the transitive sync-call closure, and any extern
  /// FuncOps named by a call site inside that closure (extern fns get
  /// added so the validation pass can diagnose them — the resolver can't
  /// see past their declaration).
  llvm::DenseSet<mlir::StringAttr> funcs;

  bool contains(mlir::StringAttr name) const { return funcs.contains(name); }
};

/// Walk `unit`, find every `async` FuncOp, and expand the reachable
/// function closure through direct `typed_csl.call` edges and through the
/// possible callees of every `typed_csl.indirect_call`.
FiberReachableFuncs
computeFiberReachable(csl::CompilationUnitOp unit,
                      const IndirectCallTargets &indirectTargets);

/// Diagnose constructs that make a reachable function's stack reach
/// statically unknowable.  Returns failure if any error was emitted.
///
/// The check matches the resolver's view of the world: a `stack_size(N)`
/// override on a node makes that node opaque (its body is not walked by
/// the resolver), which is also our cue to skip the node here — the user
/// has vouched for the bound.  A `max_recursion_depth(N)` node bounds the
/// cycles through it.
mlir::LogicalResult
validateFiberSources(csl::CompilationUnitOp unit,
                     const FiberReachableFuncs &reachable,
                     const IndirectCallTargets &indirectTargets);

/// Warn unless the local task entries left to the runtime allocator provably
/// suffice for the workers of `@await_parallel`.  Each multi-callee
/// `@await_parallel` claims one worker task per callee after the first and
/// releases it when the callee returns.  The most workers active at the same
/// time is computed over the call graph and summed over `rootTasks`, the
/// async functions bound as tasks, which can all be suspended at once.  If
/// the program also claims task entries at runtime, nothing can be proven.
/// Must run before the lowering adds the worker claims.
void checkWorkerTasks(csl::CompilationUnitOp unit,
                      const FiberReachableFuncs &reachable,
                      const IndirectCallTargets &indirectTargets,
                      ArrayRef<FuncOp> rootTasks, csl::TargetArch arch);

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERVALIDATION_H
