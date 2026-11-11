//===- FiberSizingContext.h - Two-round fiber stack-size context *- C++ -*-===//
//
// Carries the data the post-cslc fiber stack-size resolver needs from
// the round-1 PE pipeline (when @await_parallel + async fns are still
// visible) to the resolver (which runs against the per-PE ELFs produced
// by the round-1 cslc invocation).
//
// Per-PE: a clone of the PE module taken between stage 1 and stage 2
// (i.e. BEFORE fiber lowering — so round 2 can re-run stages 2 + 3 on
// the clone with a ResolvedSizeSource), the captured MLIR call graph,
// the PE source path, the round-1 emitted-CSL path that the round-2
// emit will overwrite, and the (x, y) tile coordinates of the
// @set_tile_code sites bound to this PE source.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSIZINGCONTEXT_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSIZINGCONTEXT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::typed_csl {

//===----------------------------------------------------------------------===//
// Fiber-stack layout constants (shared between FiberLoweringPass and the
// FiberStackResolver, so the resolver's max_stack math mirrors exactly what
// the pass bakes into the partition chain.  See `lowerSingleAwaitParallel`
// for how each constant participates in the layout.)
//===----------------------------------------------------------------------===//

/// Number of u16 slots in each worker's arg-area at the top of its
/// stack region (r2, r3, r4, r5, r15 + one padding slot for ld32
/// alignment).
inline constexpr unsigned kArgAreaU16Slots = 6; // = 12 B

/// Symbol of the runtime `caller_join` function.  Its measured stack
/// reach (`contrib`) is the post-site reserve at a multi-callee
/// @await_parallel: the `caller_join` + `suspend` chain runs AFTER the
/// sync callee g0 returns and reuses g0's (then-dead) region, so the
/// sync slot is `max(contrib(g0), contrib(caller_join))`.  When cslc
/// inlines both into the awaiting function (the common case) their
/// frames land above sp_at_site and contrib(caller_join) measures ~0.
inline constexpr llvm::StringLiteral kCallerJoinRuntimeSym =
    "__fiber_runtime__caller_join";

//===----------------------------------------------------------------------===//
// CapturedCallGraph
//===----------------------------------------------------------------------===//

/// Per-function call info captured from the pre-fiber-lowering IR.  Symbol
/// names are stored as `std::string` — `StringRef` would dangle the moment
/// the source module is rewritten or destroyed.
struct FuncCallInfo {
  /// `stack_size(N)` override on the FuncOp, if any. Bytes.
  std::optional<uint32_t> stackSizeOverride;

  /// `max_recursion_depth(N)` on the FuncOp, if any: the maximum number of
  /// activations of the function on one stack.
  std::optional<uint32_t> maxRecursionDepth;

  /// Sync direct calls inside this function, in source order.
  std::vector<std::string> syncCallees;

  /// One entry per `@await_parallel` site in this function (in source
  /// order); each entry lists the parallel callees in source order.
  std::vector<std::vector<std::string>> parallelSites;
};

/// Snapshot of a PE program's call graph, taken BEFORE FiberLoweringPass
/// rewrites @await_parallel / @await away.  The resolver later traverses
/// it from `rootTasks`, looks up each function's frame size in the per-PE
/// ELF's `.stack_sizes` section, and computes max_stack per node.
struct CapturedCallGraph {
  llvm::StringMap<FuncCallInfo> funcs;
  llvm::StringSet<> rootTasks;
};

//===----------------------------------------------------------------------===//
// FiberSizingEntry / FiberSizingContext
//===----------------------------------------------------------------------===//

/// Everything needed to resolve fiber sizes for one PE program and
/// re-run stages 2 + 3 on its pre-fiber clone before the round-2 emit.
struct FiberSizingEntry {
  /// Snapshot of the PE module taken AFTER stage 1 but BEFORE stage 2
  /// (fiber lowering).  Round 2 wraps a ResolvedSizes in a
  /// ResolvedSizeSource and re-runs stages 2 + 3 on this clone, then
  /// re-emits over `emittedCSLPath` for the round-2 cslc invocation.
  mlir::OwningOpRef<mlir::ModuleOp> clone;

  /// Original PE source path (e.g. "/.../pe.csl") and the path where the
  /// round-1 emit wrote the preprocessed CSL (the round-2 emit overwrites
  /// this same path).
  std::string peSourceFile;
  std::string emittedCSLPath;

  /// All (x, y) tile coordinates this PE source is bound to via
  /// @set_tile_code.  The resolver reads `.stack_sizes` from any one of
  /// the produced ELFs — they are byte-identical for the same PE source
  /// + params combo.
  llvm::SmallVector<std::pair<int, int>, 4> tileCoords;

  /// MLIR call graph captured before fiber lowering rewrote it.
  CapturedCallGraph callGraph;
};

/// Thread-safe container holding one FiberSizingEntry per PE program
/// that needs resolver-driven sizing.  ProcessTileCodePass populates it
/// during compileTasksInParallel; csl-pp consumes it after round-1 cslc
/// completes.
struct FiberSizingContext {
  /// Add an entry (called from per-PE worker threads).  Returns a
  /// reference to the inserted entry — caller may mutate fields after
  /// the call as long as it does not concurrently call other mutating
  /// methods.
  FiberSizingEntry &add(FiberSizingEntry &&entry) {
    std::lock_guard<std::mutex> lock(mutex);
    entries.push_back(std::move(entry));
    return entries.back();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex);
    return entries.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return entries.size();
  }

  /// Move all entries out of the context (consumes the context).  Use
  /// from the single-threaded post-PE-compile phase only.
  std::vector<FiberSizingEntry> takeEntries() {
    std::lock_guard<std::mutex> lock(mutex);
    return std::move(entries);
  }

private:
  mutable std::mutex mutex;
  std::vector<FiberSizingEntry> entries;
};

//===----------------------------------------------------------------------===//
// IR-attribute names for the captured call graph
//===----------------------------------------------------------------------===//

/// Discardable attributes FiberLoweringPass attaches to each typed_csl.func
/// BEFORE rewriting @await_parallel / @await, so the captured graph can be
/// extracted from the post-stage-A IR (when the original calls are gone).
///
/// - `csl.fiber.captured_sync_callees` : ArrayAttr<StringAttr>
/// - `csl.fiber.captured_parallel_sites` : ArrayAttr<ArrayAttr<StringAttr>>
/// - `csl.fiber.is_async_root_task` : UnitAttr (only on bound task fns)
inline constexpr llvm::StringLiteral kCapturedSyncCalleesAttrName =
    "csl.fiber.captured_sync_callees";
inline constexpr llvm::StringLiteral kCapturedParallelSitesAttrName =
    "csl.fiber.captured_parallel_sites";
inline constexpr llvm::StringLiteral kCapturedRootTaskAttrName =
    "csl.fiber.is_async_root_task";

//===----------------------------------------------------------------------===//
// Extraction helper
//===----------------------------------------------------------------------===//

/// Walk a post-stage-A PE module and extract its CapturedCallGraph from
/// the per-FuncOp attributes FiberLoweringPass attached.  Returns an
/// empty graph if no captured attributes are present (i.e. the PE has no
/// fiber lowering applied).
CapturedCallGraph extractCapturedCallGraph(mlir::ModuleOp peModule);

/// Whether a pre-fiber (post-stage-1) PE module needs round-2 resolver
/// involvement.
///
/// Returns true iff at least one of the symbols that FiberLoweringPass
/// will query the FiberSizeSource about lacks a `stack_size(N)` override.
/// Those queries are made for:
///
///   * **Root tasks** — async FuncOps referenced from a `@bind_*_task`
///     call (the pass asks `getRootStackU32Words`).
///   * **Parallel-await callees** — async FuncOps referenced from an
///     `@await_parallel` site by their `FutureType` symbol (the pass
///     asks `getCalleeStackU16Words`).
///
/// Sync callees of either are NOT queries — they enter the resolver's
/// `contrib()` recursion later but the pass doesn't ask the source
/// about them.  So a function with `stack_size(N)` set and no
/// `@await_parallel` in its body never triggers round 2, regardless of
/// what its sync callees do.
///
/// Returns false for non-fiber PE programs (no bindings + no
/// @await_parallel).
bool peNeedsFiberSizeResolution(mlir::ModuleOp peModule);

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSIZINGCONTEXT_H
