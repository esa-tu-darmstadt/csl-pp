//===- FiberStackResolver.h - Compute max_stack per fn from PE ELF data
//-*-C++-*-===//
//
// Given a PE's captured MLIR call graph and its per-PE linked ELF (with
// `.stack_sizes` + DWARF), compute `max_stack(F)` in bytes for every
// function the chained `__fiber_size_*` runtime arithmetic depends on,
// plus the per-root-task fiber-stack array dimension.
//
// The formula, expressed as a memoized recursion on the captured graph:
//
//   contrib(F) = (F has stack_size(N))
//                ? N
//                : own_framesize(F) + max over sites in F of
//                    sync site → g          : contrib(g)
//                    parallel(g1..gn) site  : Σ contrib(gi)
//
// A recursion cycle is unrolled: within a cycle, the reach is computed for a
// budget of remaining activations of the functions with
// `max_recursion_depth(N)`, starting at the sum of their N.
//
//   max_stack(root_task) = contrib(root_task)
//   max_stack(parallel callee X) = contrib(X)
//
// For a function whose `.stack_sizes` entry is absent because LLVM
// inlined it everywhere, own_framesize is 0 — the inlining caller's
// `.stack_sizes` entry already covers F's frame, and F's body's calls
// to deeper functions still contribute via `contrib(F)`.  A function
// in the captured graph that has no entry and was *not* inlined per
// DWARF is a hard error.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSTACKRESOLVER_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSTACKRESOLVER_H

#include "csl-mlir/Dialect/TypedCSL/Transforms/DwarfInlineReader.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizingContext.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::typed_csl {

/// Per-PE output of the resolver.  Round 2 wraps this in a
/// ResolvedSizeSource and re-runs FiberLoweringPass against the
/// pre-fiber clone of the PE module.
struct ResolvedSizes {
  /// Per-callee partition size, u16 words.  Covers every function the
  /// captured call graph reaches as a parallel callee.  Functions with
  /// a `stack_size(N)` override resolve to that value.
  llvm::StringMap<uint16_t> calleeSizeWords;

  /// Per-root-task fiber stack array dim, u32 words.  Used as the dim
  /// of `__fiber_stack_<root>: [N]u32` emitted by the re-run pass.
  llvm::StringMap<uint32_t> rootStackU32Words;
};

/// Resolve fiber sizes for one PE program.
///
/// \param elfPath Path to the round-1 linked ELF for this PE
///                (out/bin/out_X_Y.elf).
/// \param cg      Captured call graph from FiberSizingEntry.
///
/// On hard failure (function in the captured graph has no `.stack_sizes`
/// entry and is not inlined-only, or a user-supplied `stack_size(N)` is
/// provably too small) the resolver emits a diagnostic to stderr and
/// returns failure.
FailureOr<ResolvedSizes> resolveFiberSizes(llvm::StringRef elfPath,
                                           const CapturedCallGraph &cg);

/// Resolve fiber sizes from already-read frame sizes (`.stack_sizes`) and
/// DWARF inline information.  `context` names the program in diagnostics.
FailureOr<ResolvedSizes>
resolveFiberSizes(const llvm::StringMap<uint32_t> &stackSizes,
                  const DwarfInlineInfo &inlineInfo,
                  const CapturedCallGraph &cg, llvm::StringRef context);

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSTACKRESOLVER_H
