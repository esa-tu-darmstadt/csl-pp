//===- FiberSizeSource.h - Per-callee fiber stack size oracle ---*- C++ -*-===//
//
// Indirection between FiberLoweringPass and the source of per-callee /
// per-root-task stack sizes.  FiberLoweringPass runs twice in the
// two-round compile flow:
//
//   Round 1: PlaceholderSizeSource — returns the user's `stack_size(N)`
//            override if present, else a fixed placeholder.  cslc produces
//            a per-PE ELF whose `.stack_sizes` we read for round 2.
//
//   Round 2: ResolvedSizeSource — looks values up in a `ResolvedSizes`
//            map produced by FiberStackResolver from the round-1 ELF.
//            Hard-errors on a missing symbol (missing data is a bug).
//
// The two impls deliberately return identical values for any function with
// a `stack_size(N)` override — the resolver bakes the override into its
// output, so pass 2 sees the same number as pass 1 for such functions.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSIZESOURCE_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSIZESOURCE_H

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberStackResolver.h"

#include <cstdint>

namespace mlir::typed_csl {

/// Per-callee / per-root-task stack-size oracle queried by
/// FiberLoweringPass at lowering time.  Implementations differ only in
/// what they return for functions without a user-supplied `stack_size`.
struct FiberSizeSource {
  virtual ~FiberSizeSource() = default;

  /// Per-callee partition size in u16 words (= bytes/2).  Must be even so
  /// chained worker SPs remain 32-bit aligned.  Used in the partition
  /// chain `top[i+1] = top[i] - X` of @await_parallel lowering.
  virtual uint16_t getCalleeStackU16Words(FuncOp callee) = 0;

  /// Per-root-task fiber stack array dim in u32 words.  Used as the dim
  /// of `__fiber_stack_<root>: [N]u32` and to compute the
  /// `argAreaWordOffset = N*2 - kArgAreaSlots` literal baked into the
  /// init / reset functions.
  virtual uint32_t getRootStackU32Words(FuncOp root) = 0;
};

/// Round-1 source.  Honors `stack_size(N)` when present, returns fixed
/// placeholder values otherwise.  The placeholders only need to be large
/// enough for cslc to accept the IR — round-1 ELFs are never executed,
/// only inspected for their `.stack_sizes` section.
class PlaceholderSizeSource final : public FiberSizeSource {
public:
  uint16_t getCalleeStackU16Words(FuncOp callee) override;
  uint32_t getRootStackU32Words(FuncOp root) override;
};

/// Round-2 source.  Wraps a ResolvedSizes produced by
/// `resolveFiberSizes(round1ElfPath, capturedGraph)`.  A missing symbol
/// is a hard error — the resolver is expected to cover every function
/// FiberLoweringPass queries.
class ResolvedSizeSource final : public FiberSizeSource {
public:
  explicit ResolvedSizeSource(const ResolvedSizes &sizes) : sizes(sizes) {}
  uint16_t getCalleeStackU16Words(FuncOp callee) override;
  uint32_t getRootStackU32Words(FuncOp root) override;

private:
  const ResolvedSizes &sizes;
};

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_FIBERSIZESOURCE_H
