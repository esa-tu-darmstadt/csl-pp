//===- ParsedCSLCanonicalize.cpp - ParsedCSL Canonicalization Patterns ----===//
//
// Canonicalization patterns for ParsedCSL operations.
// ParsedCSL contains only canonicalization patterns that are required due to
// differences in IR structure from TypedCSL. For example, SelectOp in ParsedCSL
// allows for either of the branches to be missing if the condition is
// comptime-known, while TypedCSL requires both branches to always be present.
// The canonicalizer handles this by folding away SelectOps with constant
// conditions.
//
// All other simplifications (e.g., algebraic simplifications, folding of
// trivial selects/switches, etc.) are handled by canonicalization patterns in
// TypedCSL.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::parsed_csl;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Compute the trip count for a range with given start, stop, step.
/// Returns the number of iterations, or std::nullopt if it can't be computed.
static std::optional<int64_t> computeTripCount(int64_t start, int64_t stop,
                                               int64_t step) {
  if (step == 0)
    return std::nullopt;

  if (step > 0) {
    if (stop <= start)
      return 0;
    return (stop - start + step - 1) / step;
  } else {
    // step < 0
    if (start <= stop)
      return 0;
    return (start - stop + (-step) - 1) / (-step);
  }
}

//===----------------------------------------------------------------------===//
// SelectOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify select operations with constant conditions.
/// - If condition is constant true: replace with true_value
/// - If condition is constant false and has else region: inline region and use
///   yielded value
/// - If condition is constant false and no else: assertion failure (compiler
///   bug)
///
/// This canonicalizer is particularly important for select ops without an else
/// branch, where the condition must be comptime-known true. After comptime
/// evaluation replaces the condition with a constant, this pattern folds the
/// select to just the true branch value.
struct SimplifySelectWithConstantCondition : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Check if condition is defined by a ConstantOp
    auto condDefOp = op.getCondition().getDefiningOp<ConstantOp>();
    if (!condDefOp)
      return failure();

    // Get the constant value
    TypedAttr condAttr = condDefOp.getValue();

    // Try to extract boolean value
    auto intAttr = dyn_cast<IntegerAttr>(condAttr);
    if (!intAttr)
      return failure();

    // Check if it's a boolean type (i1)
    if (!intAttr.getType().isInteger(1))
      return failure();

    bool conditionValue = intAttr.getValue().getBoolValue();

    if (conditionValue) {
      // Condition is true: inline the then region and use the yielded value
      Region &thenRegion = op.getThenRegion();
      Block &thenBlock = thenRegion.front();

      auto yieldOp = cast<YieldOp>(thenBlock.getTerminator());
      Value yieldedValue = yieldOp.getValue();

      rewriter.inlineBlockBefore(&thenBlock, op);
      rewriter.eraseOp(yieldOp);
      rewriter.replaceOp(op, yieldedValue);
    } else if (op.hasElseBranch()) {
      // Condition is false with else region: inline the region and use the
      // yielded value
      Region &elseRegion = op.getElseRegion();
      Block &elseBlock = elseRegion.front();

      auto yieldOp = cast<YieldOp>(elseBlock.getTerminator());
      Value yieldedValue = yieldOp.getValue();

      rewriter.inlineBlockBefore(&elseBlock, op);
      rewriter.eraseOp(yieldOp);
      rewriter.replaceOp(op, yieldedValue);
    } else {
      // Condition is false without else branch: this should have been caught
      // by comptime validation. If we reach here, it's a compiler bug.
      llvm_unreachable("select without else has constant false condition - "
                       "should have been caught by comptime validation");
    }
    return success();
  }
};
} // namespace

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<SimplifySelectWithConstantCondition>(context);
}

//===----------------------------------------------------------------------===//
// SwitchOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify switch operations with constant input.
/// When the input is a compile-time constant, fold the switch to just the
/// matching case branch (or else branch if no case matches).
struct SimplifySwitchWithConstantInput : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;

  // Helper to extract integer value from either IntegerAttr or EnumValueAttr
  static std::optional<int64_t> getIntegerValue(TypedAttr attr) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().getSExtValue();
    if (auto enumAttr = mlir::dyn_cast<csl::EnumValueAttr>(attr))
      return enumAttr.getIntValue();
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(SwitchOp op,
                                PatternRewriter &rewriter) const override {
    // Check if input is defined by a ConstantOp
    auto inputDefOp = op.getInput().getDefiningOp<ConstantOp>();
    if (!inputDefOp)
      return failure();

    // Get the constant value
    TypedAttr inputAttr = inputDefOp.getValue();
    auto inputValue = getIntegerValue(inputAttr);
    if (!inputValue)
      return failure();

    // Get case values attribute (set by ComptimeEvalPass)
    auto caseValuesAttr = op.getCaseValues();
    if (!caseValuesAttr)
      return failure();

    // Find matching case
    std::optional<size_t> matchingCaseIdx;
    for (size_t i = 0; i < caseValuesAttr->size(); ++i) {
      auto thisCaseValues = mlir::cast<ArrayAttr>((*caseValuesAttr)[i]);
      for (Attribute caseVal : thisCaseValues) {
        auto caseIntValue = getIntegerValue(mlir::cast<TypedAttr>(caseVal));
        if (caseIntValue && *caseIntValue == *inputValue) {
          matchingCaseIdx = i;
          break;
        }
      }
      if (matchingCaseIdx)
        break;
    }

    // Get the matching region
    Region *matchingRegion;
    if (matchingCaseIdx) {
      matchingRegion = &op.getCaseRegions()[*matchingCaseIdx];
    } else if (op.hasElseBranch()) {
      matchingRegion = &op.getElseRegion();
    } else {
      // No match and no else - this should have been caught by comptime eval
      return failure();
    }

    // Region might be empty if it wasn't evaluated during comptime
    if (matchingRegion->empty())
      return failure();

    Block &matchingBlock = matchingRegion->front();

    // Get the yielded value (switch expression)
    // Check mightHaveTerminator() first to avoid assertion in getTerminator()
    YieldOp yieldOp = nullptr;
    if (!matchingBlock.empty() && matchingBlock.mightHaveTerminator())
      yieldOp = dyn_cast<YieldOp>(matchingBlock.getTerminator());
    if (!yieldOp) {
      // No yield - might be a statement switch or ends with return/break
      // Just inline the block and erase the switch
      rewriter.inlineBlockBefore(&matchingBlock, op);
      rewriter.eraseOp(op);
      return success();
    }

    Value yieldedValue = yieldOp.getValue();

    // Inline the matching block before the switch
    rewriter.inlineBlockBefore(&matchingBlock, op);
    rewriter.eraseOp(yieldOp);
    rewriter.replaceOp(op, yieldedValue);

    return success();
  }
};
/// Erase all operations after an op that acts as a logical terminator
/// (ReturnOp, UnreachableOp) in the same block. ParsedCSL doesn't mark these
/// as MLIR Terminators (FuncOp has NoTerminator), so dead code after them
/// isn't cleaned up automatically. This can happen when the canonicalizer
/// folds a comptime-known switch whose matching branch contains a return.
template <typename OpTy>
struct EraseOpsAfterTerminator : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    if (&block->back() == op.getOperation())
      return failure();

    while (&block->back() != op.getOperation()) {
      Operation &deadOp = block->back();
      deadOp.dropAllUses();
      rewriter.eraseOp(&deadOp);
    }
    return success();
  }
};
} // namespace

void SwitchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<SimplifySwitchWithConstantInput>(context);
}

void ReturnOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EraseOpsAfterTerminator<ReturnOp>>(context);
}

void UnreachableOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<EraseOpsAfterTerminator<UnreachableOp>>(context);
}
