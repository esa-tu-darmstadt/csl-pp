//===- NormalizeCasts.cpp - Validate and Normalize Casts --------*- C++ -*-===//
//
// This file implements a pass that validates and normalizes cast operations.
//
// Uses a pattern-based approach where all normalization patterns run to
// fixed-point in a single greedy rewrite pass. This handles cascading
// transformations like: insert cast → push into select → fold to constant.
//
// This makes all type conversions explicit and validates them before lowering
// to TypedCSL, catching overflow and invalid casts early.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "normalize-casts"

using namespace mlir;
using namespace mlir::parsed_csl;

namespace mlir {
namespace parsed_csl {

#define GEN_PASS_DEF_NORMALIZECASTSPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

} // namespace parsed_csl
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Error Collector
//===----------------------------------------------------------------------===//

/// Collects error diagnostics during pattern rewrites.
class ErrorCollector {
  SmallVector<InFlightDiagnostic> diagnostics_;
  bool hasErrors_ = false;

public:
  /// Record an error at the given location and return a reference to the
  /// diagnostic.
  InFlightDiagnostic &recordError(Location loc) {
    hasErrors_ = true;
    diagnostics_.push_back(mlir::emitError(loc));
    return diagnostics_.back();
  }

  /// Take ownership of an existing InFlightDiagnostic and mark it as an error.
  InFlightDiagnostic &takeError(InFlightDiagnostic &&diag) {
    hasErrors_ = true;
    diagnostics_.push_back(std::move(diag));
    return diagnostics_.back();
  }

  /// Check if any errors were recorded
  bool hasErrors() const { return hasErrors_; }

  /// Emit all collected errors
  void emitAll() {
    diagnostics_.clear();
    hasErrors_ = false;
  }
};

//===----------------------------------------------------------------------===//
// Pattern Base Class
//===----------------------------------------------------------------------===//

/// Base class for normalization patterns that need access to analysis and error
/// collector
template <typename OpTy>
class NormalizationPattern : public OpRewritePattern<OpTy> {
protected:
  ErrorCollector &errorCollector;

  /// Helper to insert an implicit cast and register its type with the analysis.
  /// Validates that the cast is allowed as an implicit cast before creating it.
  /// If multiple target types are provided, the first that is compatible will
  /// be used. Creates a CastOp marked with the 'implicit' attribute. Returns
  /// failure if the implicit cast is not allowed.
  FailureOr<Value> insertImplicitCast(OpBuilder &builder, Location loc,
                                      Value source, ArrayRef<Type> targetTypes,
                                      bool allowExplicitCast = false) const {
    auto maybeSourceType = getInferredType(source);
    if (failed(maybeSourceType)) {
      errorCollector.recordError(loc)
          << "cannot insert implicit cast: failed to query source type";
      return failure();
    }

    // If types already match, no cast needed - just return the source
    if (llvm::is_contained(targetTypes, *maybeSourceType)) {
      return source;
    }

    // Validate that this cast is allowed as an implicit cast
    // A value is comptime-known if it comes from a ConstantOp
    bool comptimeKnown = source.getDefiningOp<ConstantOp>() != nullptr;

    Type targetType;
    bool requiresExplicitCast = false;
    // Try without explicit cast first
    for (Type candidateType : targetTypes) {
      if (csl::isImplicitCastAllowed(*maybeSourceType, candidateType,
                                     comptimeKnown)) {
        targetType = candidateType;
        break;
      }
    }

    if (!targetType && allowExplicitCast) {
      // Try again allowing explicit casts
      for (Type candidateType : targetTypes) {
        if (csl::areCastCompatible(*maybeSourceType, candidateType)) {
          targetType = candidateType;
          requiresExplicitCast = true;
          break;
        }
      }
    }

    if (!targetType) {
      std::string targetTypeList;
      for (Type t : targetTypes) {
        if (!targetTypeList.empty())
          targetTypeList += " or ";
        targetTypeList += csl::prettyPrintType(t);
      }
      errorCollector.recordError(loc)
          << "implicit cast from " << csl::prettyPrintType(*maybeSourceType)
          << " to " << targetTypeList << " is not allowed";
      return failure();
    }

    // Create the cast operation with implicit attribute
    Value targetTypeValue = BuiltinTypeOp::create(
        builder, loc, parsed_csl::TypeAttr::get(targetType));
    Value castValue = CastOp::create(
        builder, loc, source, targetTypeValue, mlir::TypeAttr::get(targetType),
        requiresExplicitCast ? nullptr : builder.getUnitAttr());

    return castValue;
  }

  FailureOr<Value> insertImplicitCast(OpBuilder &builder, Location loc,
                                      Value source, Type targetType,
                                      bool allowExplicitCast = false) const {
    return insertImplicitCast(builder, loc, source, ArrayRef<Type>{targetType},
                              allowExplicitCast);
  }

public:
  NormalizationPattern(MLIRContext *context, ErrorCollector &errorCollector,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit),
        errorCollector(errorCollector) {}
};

//===----------------------------------------------------------------------===//
// Normalization Patterns
//===----------------------------------------------------------------------===//

/// Fold CastOp when source is comptime-known
struct FoldCastOpPattern : public NormalizationPattern<CastOp> {
  FoldCastOpPattern(MLIRContext *context, ErrorCollector &errorCollector)
      : NormalizationPattern(context, errorCollector, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    Value source = op.getSource();

    auto constantOp = source.getDefiningOp<ConstantOp>();
    if (!constantOp)
      return rewriter.notifyMatchFailure(op, "source not comptime-known");

    auto maybeTargetType = getInferredType(op.getResult());
    if (failed(maybeTargetType))
      return rewriter.notifyMatchFailure(op, "target type not inferred");

    InFlightDiagnostic diag = mlir::emitError(op.getLoc());
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(constantOp.getValue(), *maybeTargetType,
                          diag.getUnderlyingDiagnostic());

    if (failed(castedValue)) {
      // Pointer-to-pointer casts (@ptrcast) that reinterpret the element
      // type are valid but not foldable — they remain as runtime operations.
      if (isa<csl::PointerType>(constantOp.getValue().getType()) &&
          isa<csl::PointerType>(*maybeTargetType)) {
        diag.abandon();
        return rewriter.notifyMatchFailure(op, "pointer cast not foldable");
      }
      return errorCollector.takeError(std::move(diag))
             << " (invalid cast of comptime-known value)";
    }

    diag.abandon();
    rewriter.replaceOpWithNewOp<ConstantOp>(op, *castedValue);

    return success();
  }
};

/// Push CastOp into SelectOp branches:
/// cast(select(c, then{t}, else{f})) -> select(c, then{cast(t)}, else{cast(f)})
/// This inserts casts inside both regions before the yields.
struct PushCastIntoSelectPattern : public NormalizationPattern<CastOp> {
  PushCastIntoSelectPattern(MLIRContext *context,
                            ErrorCollector &errorCollector)
      : NormalizationPattern(context, errorCollector, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    auto selectOp = op.getSource().getDefiningOp<SelectOp>();
    if (!selectOp)
      return rewriter.notifyMatchFailure(op, "source not a select");

    // Select without else should have been folded during comptime evaluation
    if (!selectOp.hasThenBranch() || !selectOp.hasElseBranch())
      llvm_unreachable(
          "select without then or else should have been folded during "
          "comptime evaluation");

    FailureOr<Type> targetType = getInferredType(op.getResult());
    if (failed(targetType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred target type");

    // Insert cast inside the then region before the yield
    Region &thenRegion = selectOp.getThenRegion();
    auto thenYieldOp = cast<YieldOp>(thenRegion.front().getTerminator());
    Value trueValue = thenYieldOp.getValue();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(thenYieldOp);
      FailureOr<Value> castedTrue = insertImplicitCast(
          rewriter, thenYieldOp.getLoc(), trueValue, *targetType);
      if (failed(castedTrue))
        return failure();
      rewriter.modifyOpInPlace(thenYieldOp, [&]() {
        thenYieldOp.getValueMutable().assign(*castedTrue);
      });
    }

    // Insert cast inside the else region before the yield
    Region &elseRegion = selectOp.getElseRegion();
    auto elseYieldOp = cast<YieldOp>(elseRegion.front().getTerminator());
    Value falseValue = elseYieldOp.getValue();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(elseYieldOp);
      FailureOr<Value> castedFalse = insertImplicitCast(
          rewriter, elseYieldOp.getLoc(), falseValue, *targetType);
      if (failed(castedFalse))
        return failure();
      rewriter.modifyOpInPlace(elseYieldOp, [&]() {
        elseYieldOp.getValueMutable().assign(*castedFalse);
      });
    }

    // Update the select op's inferred type
    rewriter.modifyOpInPlace(selectOp, [&]() {
      selectOp.setInferredTypeAttr(mlir::TypeAttr::get(*targetType));
    });

    // Replace the cast with the updated select
    rewriter.replaceOp(op, selectOp.getResult());
    return success();
  }
};

/// Push CastOp into SwitchOp branches:
/// cast(switch(x) { ... }) -> switch(x) { ..., yield cast(v), ... }
/// This inserts casts inside each case region before the yields.
struct PushCastIntoSwitchPattern : public NormalizationPattern<CastOp> {
  PushCastIntoSwitchPattern(MLIRContext *context,
                            ErrorCollector &errorCollector)
      : NormalizationPattern(context, errorCollector, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    auto switchOp = op.getSource().getDefiningOp<SwitchOp>();
    if (!switchOp)
      return rewriter.notifyMatchFailure(op, "source not a switch");

    // Switch without result is a statement, not an expression
    if (!switchOp.getResult())
      return rewriter.notifyMatchFailure(op, "switch has no result");

    FailureOr<Type> targetType = getInferredType(op.getResult());
    if (failed(targetType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred target type");

    // Helper to get YieldOp from a region (may not exist if region ends with
    // return/break/continue)
    auto getRegionYield = [](Region &region) -> YieldOp {
      if (region.empty() || region.front().empty())
        return nullptr;
      return mlir::dyn_cast<YieldOp>(&region.front().back());
    };

    // Insert casts inside each case region before the yield
    for (Region &caseRegion : switchOp.getCaseRegions()) {
      auto yieldOp = getRegionYield(caseRegion);
      if (!yieldOp)
        continue; // Region ends with control flow, no yield to cast

      Value yieldedValue = yieldOp.getValue();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      FailureOr<Value> castedValue = insertImplicitCast(
          rewriter, yieldOp.getLoc(), yieldedValue, *targetType);
      if (failed(castedValue))
        return failure();
      rewriter.modifyOpInPlace(
          yieldOp, [&]() { yieldOp.getValueMutable().assign(*castedValue); });
    }

    // Insert cast inside the else region if present
    if (!switchOp.getElseRegion().empty()) {
      auto elseYieldOp = getRegionYield(switchOp.getElseRegion());
      if (elseYieldOp) {
        Value elseValue = elseYieldOp.getValue();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(elseYieldOp);
        FailureOr<Value> castedElse = insertImplicitCast(
            rewriter, elseYieldOp.getLoc(), elseValue, *targetType);
        if (failed(castedElse))
          return failure();
        rewriter.modifyOpInPlace(elseYieldOp, [&]() {
          elseYieldOp.getValueMutable().assign(*castedElse);
        });
      }
    }

    // Update the switch op's inferred type
    rewriter.modifyOpInPlace(switchOp, [&]() {
      switchOp.setInferredTypeAttr(mlir::TypeAttr::get(*targetType));
    });

    // Replace the cast with the updated switch
    rewriter.replaceOp(op, switchOp.getResult());
    return success();
  }
};

/// Push CastOp into LabeledBlockOp breaks:
/// cast(labeled_block "lbl" { break :lbl v }) ->
///     labeled_block "lbl" { break :lbl cast(v) }
/// This inserts casts before each break targeting the labeled block.
struct PushCastIntoLabeledBlockPattern : public NormalizationPattern<CastOp> {
  PushCastIntoLabeledBlockPattern(MLIRContext *context,
                                  ErrorCollector &errorCollector)
      : NormalizationPattern(context, errorCollector, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    auto labeledBlockOp = op.getSource().getDefiningOp<LabeledBlockOp>();
    if (!labeledBlockOp)
      return rewriter.notifyMatchFailure(op, "source not a labeled block");

    FailureOr<Type> targetType = getInferredType(op.getResult());
    if (failed(targetType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred target type");

    StringRef blockLabel = labeledBlockOp.getLabel();

    // Walk the body to find all BreakOps targeting this labeled block.
    // BreakOps can be nested inside control flow (IfOp, etc.).
    SmallVector<BreakOp> targetingBreaks;
    labeledBlockOp.getBody().walk([&](BreakOp breakOp) {
      if (breakOp.getTargetLabel() &&
          breakOp.getTargetLabel().value() == blockLabel)
        targetingBreaks.push_back(breakOp);
    });

    // Insert casts before each break's value
    for (BreakOp breakOp : targetingBreaks) {
      if (!breakOp.getValue())
        continue; // Void break - nothing to cast

      Value breakValue = breakOp.getValue();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(breakOp);
      FailureOr<Value> castedValue = insertImplicitCast(
          rewriter, breakOp.getLoc(), breakValue, *targetType);
      if (failed(castedValue))
        return failure();
      rewriter.modifyOpInPlace(
          breakOp, [&]() { breakOp.getValueMutable().assign(*castedValue); });
    }

    // Update the labeled block's inferred type
    rewriter.modifyOpInPlace(labeledBlockOp, [&]() {
      labeledBlockOp.setInferredTypeAttr(mlir::TypeAttr::get(*targetType));
    });

    // Replace the cast with the updated labeled block
    rewriter.replaceOp(op, labeledBlockOp.getResult());
    return success();
  }
};

/// Normalize LabeledBlockOp break values to the block's inferred type.
/// When breaks have different-but-compatible types (e.g., comptime_int and
/// i32), the block's inferred type is their common type. This pattern casts
/// each break value to match.
struct NormalizeLabeledBlockOpPattern
    : public NormalizationPattern<LabeledBlockOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(LabeledBlockOp op,
                                PatternRewriter &rewriter) const override {
    if (errorCollector.hasErrors())
      return failure();

    FailureOr<Type> blockType = getInferredType(op.getResult());
    if (failed(blockType))
      return rewriter.notifyMatchFailure(op, "block type not inferred");

    // Void blocks have no values to normalize
    if (mlir::isa<csl::VoidType>(*blockType))
      return rewriter.notifyMatchFailure(op, "void block");

    StringRef blockLabel = op.getLabel();
    bool modified = false;

    // Walk the body to find all BreakOps targeting this block
    op.getBody().walk([&](BreakOp breakOp) {
      if (!breakOp.getTargetLabel() ||
          breakOp.getTargetLabel().value() != blockLabel)
        return;
      if (!breakOp.getValue())
        return;

      FailureOr<Type> breakType = getInferredType(breakOp.getValue());
      if (failed(breakType) || *breakType == *blockType)
        return;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(breakOp);
      FailureOr<Value> castedValue = insertImplicitCast(
          rewriter, breakOp.getLoc(), breakOp.getValue(), *blockType);
      if (failed(castedValue))
        return;
      rewriter.modifyOpInPlace(
          breakOp, [&]() { breakOp.getValueMutable().assign(*castedValue); });
      modified = true;
    });

    return modified ? success()
                    : rewriter.notifyMatchFailure(op, "types already match");
  }
};

/// Make implicit casts explicit in BinaryOp operands
struct NormalizeBinaryOpPattern : public NormalizationPattern<BinaryOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    FailureOr<Type> lhsType = getInferredType(op.getLhs());
    FailureOr<Type> rhsType = getInferredType(op.getRhs());
    if (failed(lhsType) || failed(rhsType))
      return errorCollector.recordError(op.getLoc())
             << "operand types not inferred";

    // Build TypeOrAttr operands: use TypedAttr when comptime-known.
    csl::TypeOrAttr lhsOp(*lhsType), rhsOp(*rhsType);
    if (auto constOp = op.getLhs().getDefiningOp<ConstantOp>())
      lhsOp = constOp.getValue();
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>())
      rhsOp = constOp.getValue();

    // Use inferBinaryOp to determine target types for each operand.
    auto inferResult = csl::inferBinaryOp(op.getKind(), lhsOp, rhsOp);
    if (failed(inferResult))
      return errorCollector.recordError(op.getLoc())
             << "incompatible operand types for binary operation "
             << stringifyBinaryOpKind(op.getKind()) << ": "
             << csl::prettyPrintType(*lhsType) << " and "
             << csl::prettyPrintType(*rhsType);

    Type lhsTarget = inferResult->lhsTarget;
    Type rhsTarget = inferResult->rhsTarget;

    if (*lhsType == lhsTarget && *rhsType == rhsTarget)
      return rewriter.notifyMatchFailure(op, "types already match targets");

    Value newLhs = op.getLhs();
    Value newRhs = op.getRhs();

    if (*lhsType != lhsTarget) {
      FailureOr<Value> castedLhs =
          insertImplicitCast(rewriter, op.getLoc(), op.getLhs(), lhsTarget);
      if (failed(castedLhs))
        return failure();
      newLhs = *castedLhs;
    }

    if (*rhsType != rhsTarget) {
      FailureOr<Value> castedRhs =
          insertImplicitCast(rewriter, op.getLoc(), op.getRhs(), rhsTarget);
      if (failed(castedRhs))
        return failure();
      newRhs = *castedRhs;
    }

    // Create new binary operation with casted operands
    FailureOr<Type> resultType = getInferredType(op.getResult());
    if (failed(resultType))
      return errorCollector.recordError(op.getLoc())
             << "failed to get inferred result type";

    auto newBinaryOp =
        BinaryOp::create(rewriter, op.getLoc(), op.getKind(), newLhs, newRhs,
                         /*inferred_type=*/mlir::TypeAttr::get(*resultType));

    rewriter.replaceOp(op, newBinaryOp);

    return success();
  }
};

/// Cast branches of SelectOp to common type.
/// Both branches are now regions. This pattern:
/// 1. Normalizes the condition to i1
/// 2. Inserts casts inside both regions before the yield if needed
struct NormalizeSelectOpPattern : public NormalizationPattern<SelectOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Select without else should have been folded during comptime evaluation
    if (!op.hasThenBranch() || !op.hasElseBranch())
      llvm_unreachable(
          "select without then or else should have been folded during "
          "comptime evaluation");

    // Get the yielded values from both regions
    Region &thenRegion = op.getThenRegion();
    auto thenYieldOp = cast<YieldOp>(thenRegion.front().getTerminator());
    Value trueValue = thenYieldOp.getValue();

    Region &elseRegion = op.getElseRegion();
    auto elseYieldOp = cast<YieldOp>(elseRegion.front().getTerminator());
    Value falseValue = elseYieldOp.getValue();

    FailureOr<Type> trueType = getInferredType(trueValue);
    FailureOr<Type> falseType = getInferredType(falseValue);
    if (failed(trueType) || failed(falseType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer operand types";

    FailureOr<Type> conditionType = getInferredType(op.getCondition());
    if (failed(conditionType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer condition type";

    FailureOr<Type> commonType = csl::getCommonType(*trueType, *falseType);
    if (failed(commonType))
      return errorCollector.recordError(op.getLoc())
             << "incompatible branch types for select operation";

    bool modified = false;

    // Normalize condition to i1
    Value newCondition = op.getCondition();
    if (*conditionType != rewriter.getI1Type()) {
      FailureOr<Value> castedCondition = insertImplicitCast(
          rewriter, op.getLoc(), op.getCondition(), rewriter.getI1Type());
      if (failed(castedCondition))
        return failure();
      newCondition = *castedCondition;
      modified = true;
    }

    // Cast the yielded value inside the then region if needed
    if (*trueType != *commonType) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(thenYieldOp);
      FailureOr<Value> castedTrue = insertImplicitCast(
          rewriter, thenYieldOp.getLoc(), trueValue, *commonType);
      if (failed(castedTrue))
        return failure();
      rewriter.modifyOpInPlace(thenYieldOp, [&]() {
        thenYieldOp.getValueMutable().assign(*castedTrue);
      });
      modified = true;
    }

    // Cast the yielded value inside the else region if needed
    if (*falseType != *commonType) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(elseYieldOp);
      FailureOr<Value> castedFalse = insertImplicitCast(
          rewriter, elseYieldOp.getLoc(), falseValue, *commonType);
      if (failed(castedFalse))
        return failure();
      rewriter.modifyOpInPlace(elseYieldOp, [&]() {
        elseYieldOp.getValueMutable().assign(*castedFalse);
      });
      modified = true;
    }

    if (!modified)
      return rewriter.notifyMatchFailure(
          op, "types already match and condition is i1");

    // Update condition if changed
    if (newCondition != op.getCondition()) {
      rewriter.modifyOpInPlace(
          op, [&]() { op.getConditionMutable().assign(newCondition); });
    }

    // Update inferred type to the common type
    if (op.getInferredType() != *commonType) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setInferredTypeAttr(mlir::TypeAttr::get(*commonType));
      });
    }

    return success();
  }
};

/// Normalize WhileOp condition region yielded value to i1.
struct NormalizeWhileOpPattern : public NormalizationPattern<WhileOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Get the yielded value from the condition region
    Region &condRegion = op.getCondRegion();
    auto yieldOp = cast<YieldOp>(condRegion.front().getTerminator());
    Value condValue = yieldOp.getValue();

    FailureOr<Type> condType = getInferredType(condValue);
    if (failed(condType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer condition type";

    // Condition should be normalized to i1
    Type boolType = rewriter.getI1Type();
    if (*condType == boolType)
      return rewriter.notifyMatchFailure(op, "condition already i1");

    // Insert cast inside the condition region before the yield
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    FailureOr<Value> castedCond =
        insertImplicitCast(rewriter, yieldOp.getLoc(), condValue, boolType);
    if (failed(castedCond))
      return failure();

    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldOp.getValueMutable().assign(*castedCond); });
    return success();
  }
};

/// Normalize VariableOp initializer type
struct NormalizeVariableOpPattern : public NormalizationPattern<VariableOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(VariableOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    if (!op.getInitializer())
      return rewriter.notifyMatchFailure(op, "no initializer");

    FailureOr<Type> varType = getInferredType(op.getResult());
    FailureOr<Type> initType = getInferredType(op.getInitializer());
    if (failed(varType) || failed(initType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer types for variable";

    auto lvalueCslType = mlir::cast<csl::LValueType>(*varType);
    Type elementType = lvalueCslType.getValueType();

    if (*initType == elementType)
      return rewriter.notifyMatchFailure(op, "types already match");

    FailureOr<Value> castedInit = insertImplicitCast(
        rewriter, op.getLoc(), op.getInitializer(), elementType);
    if (failed(castedInit))
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getInitializerMutable().assign(*castedInit); });
    return success();
  }
};

/// Normalize AssignOp value type
struct NormalizeAssignOpPattern : public NormalizationPattern<AssignOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    FailureOr<Type> lvalueType = getInferredType(op.getLvalue());
    FailureOr<Type> valueType = getInferredType(op.getValue());
    if (failed(lvalueType) || failed(valueType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer types for assignment";

    auto lvalueCslType = mlir::cast<csl::LValueType>(*lvalueType);
    Type elementType = lvalueCslType.getValueType();

    if (*valueType == elementType)
      return rewriter.notifyMatchFailure(op, "types already match");

    FailureOr<Value> castedValue =
        insertImplicitCast(rewriter, op.getLoc(), op.getValue(), elementType);
    if (failed(castedValue))
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getValueMutable().assign(*castedValue); });
    return success();
  }
};

/// Normalize CallOp callee and argument types
struct NormalizeCallOpPattern : public NormalizationPattern<CallOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Get the callee's inferred type to extract parameter types
    FailureOr<Type> calleeType = getInferredType(op.getCallee());
    if (failed(calleeType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer callee type";

    // Extract FunctionType from callee type (may be wrapped in PointerType)
    FunctionType funcType;
    bool calleeIsFuncRef = false;
    if (auto ft = mlir::dyn_cast<FunctionType>(*calleeType)) {
      funcType = ft;
      calleeIsFuncRef = true;
    } else if (auto ptrType = mlir::dyn_cast<csl::PointerType>(*calleeType)) {
      funcType = mlir::dyn_cast<FunctionType>(ptrType.getPointeeType());
    }
    if (!funcType)
      return errorCollector.recordError(op.getLoc())
             << "callee does not have function type";

    ArrayRef<Type> paramTypes = funcType.getInputs();
    if (paramTypes.size() != op.getArgs().size())
      return errorCollector.recordError(op.getLoc())
             << "argument count mismatch";

    bool modified = false;

    // For indirect calls (callee is not GetSymbolOp or ConstantOp with
    // FuncRefAttr), cast fn(...) to *const fn(...) if needed
    if (calleeIsFuncRef) {
      Value callee = op.getCallee();
      bool isDirectCall = false;

      // Check if this is a direct call (GetSymbolOp or ConstantOp with
      // FuncRef)
      if (callee.getDefiningOp<GetSymbolOp>()) {
        isDirectCall = true;
      } else if (auto constantOp = callee.getDefiningOp<ConstantOp>()) {
        if (mlir::isa<csl::FuncRefAttr, csl::FuncAddrAttr>(
                constantOp.getValue()))
          isDirectCall = true;
      }

      if (!isDirectCall) {
        // Indirect call with fn(...) type - cast to *const fn(...)
        Type ptrType = csl::PointerType::get(funcType, /*isConst=*/true);
        FailureOr<Value> castedCallee =
            insertImplicitCast(rewriter, op.getLoc(), callee, ptrType);
        if (failed(castedCallee))
          return failure();
        rewriter.modifyOpInPlace(
            op, [&]() { op.getCalleeMutable().assign(*castedCallee); });
        modified = true;
      }
    }

    // Normalize argument types
    for (auto [idx, arg] : llvm::enumerate(op.getArgs())) {
      FailureOr<Type> argType = getInferredType(arg);
      Type paramType = paramTypes[idx];

      if (failed(argType))
        return errorCollector.recordError(op.getLoc())
               << "failed to infer types for call argument";

      if (*argType != paramType) {
        FailureOr<Value> castedArg =
            insertImplicitCast(rewriter, op.getLoc(), arg, paramType);
        if (failed(castedArg))
          return failure();
        rewriter.modifyOpInPlace(
            op, [&]() { op.getArgsMutable()[idx].assign(*castedArg); });
        modified = true;
      }
    }

    return modified ? success()
                    : rewriter.notifyMatchFailure(op, "types already match");
  }
};

/// Normalize ReturnOp value type
struct NormalizeReturnOpPattern : public NormalizationPattern<ReturnOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    if (op.getOperands().empty())
      return rewriter.notifyMatchFailure(op, "no return value");

    auto funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp)
      return errorCollector.recordError(op.getLoc())
             << "return not within an instantiated function";

    Type expectedType = funcOp.getBodyReturnType();

    FailureOr<Type> returnValueType = getInferredType(op.getOperands()[0]);
    if (failed(returnValueType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer return value type";

    if (*returnValueType == expectedType)
      return rewriter.notifyMatchFailure(op, "types already match");

    FailureOr<Value> castedValue = insertImplicitCast(
        rewriter, op.getLoc(), op.getOperands()[0], expectedType);
    if (failed(castedValue))
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperandMutable().assign(*castedValue); });
    return success();
  }
};

/// Normalize ConstructTupleOp field value types
struct NormalizeConstructTupleOpPattern
    : public NormalizationPattern<ConstructTupleOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(ConstructTupleOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Get the inferred tuple type
    FailureOr<Type> tupleType = getInferredType(op.getResult());
    if (failed(tupleType))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer tuple type";

    auto structType = mlir::dyn_cast<csl::StructType>(*tupleType);
    if (!structType)
      return errorCollector.recordError(op.getLoc())
             << "tuple type is not a struct type";

    ArrayRef<Type> expectedFieldTypes = structType.getFieldTypes();
    OperandRange fieldValues = op.getFieldValues();

    if (expectedFieldTypes.size() != fieldValues.size())
      return errorCollector.recordError(op.getLoc())
             << "field count mismatch: expected " << expectedFieldTypes.size()
             << " fields, got " << fieldValues.size();

    bool modified = false;

    // Check each field value
    for (auto [idx, fieldValue] : llvm::enumerate(fieldValues)) {
      FailureOr<Type> actualType = getInferredType(fieldValue);
      if (failed(actualType))
        return errorCollector.recordError(op.getLoc())
               << "failed to infer type of field " << idx;

      Type expectedType = expectedFieldTypes[idx];

      if (*actualType != expectedType) {
        FailureOr<Value> castedField =
            insertImplicitCast(rewriter, op.getLoc(), fieldValue, expectedType);
        if (failed(castedField))
          return failure();

        rewriter.modifyOpInPlace(op, [&]() {
          op.getFieldValuesMutable()[idx].assign(*castedField);
        });
        modified = true;
      }
    }

    return modified ? success()
                    : rewriter.notifyMatchFailure(op, "types already match");
  }
};

/// Normalize ConstructStructOp field value types
struct NormalizeConstructStructOpPattern
    : public NormalizationPattern<ConstructStructOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(ConstructStructOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Get the inferred struct type
    FailureOr<Type> structTypeResult = getInferredType(op.getResult());
    if (failed(structTypeResult))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer struct type";

    auto structType = mlir::dyn_cast<csl::StructType>(*structTypeResult);
    if (!structType)
      return errorCollector.recordError(op.getLoc())
             << "struct type is not a struct type";

    ArrayRef<Attribute> fieldNamesAttr = op.getFieldNames().getValue();
    OperandRange fieldValues = op.getFieldValues();

    if (fieldNamesAttr.size() != fieldValues.size())
      return errorCollector.recordError(op.getLoc())
             << "field names count does not match field values count";

    bool modified = false;

    // Check each field value
    // Note: field order in field_values matches field_names, but might differ
    // from the order in the struct type
    for (auto [idx, fieldValue] : llvm::enumerate(fieldValues)) {
      // Get the field name for this value
      auto fieldNameAttr = mlir::dyn_cast<StringAttr>(fieldNamesAttr[idx]);
      if (!fieldNameAttr)
        return errorCollector.recordError(op.getLoc())
               << "field name " << idx << " is not a string attribute";

      StringRef fieldName = fieldNameAttr.getValue();

      // Look up the expected type for this field name in the struct type
      FailureOr<Type> expectedType = structType.getFieldType(fieldName);
      if (failed(expectedType))
        return errorCollector.recordError(op.getLoc())
               << "field '" << fieldName << "' not found in struct type";

      FailureOr<Type> actualType = getInferredType(fieldValue);
      if (failed(actualType))
        return errorCollector.recordError(op.getLoc())
               << "failed to infer type of field '" << fieldName << "'";

      if (*actualType != *expectedType) {
        FailureOr<Value> castedField = insertImplicitCast(
            rewriter, op.getLoc(), fieldValue, *expectedType);
        if (failed(castedField))
          return failure();

        rewriter.modifyOpInPlace(op, [&]() {
          op.getFieldValuesMutable()[idx].assign(*castedField);
        });
        modified = true;
      }
    }

    return modified ? success()
                    : rewriter.notifyMatchFailure(op, "types already match");
  }
};

/// Normalize ConstructArrayOp element types
struct NormalizeConstructArrayOpPattern
    : public NormalizationPattern<ConstructArrayOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(ConstructArrayOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Get the inferred array type
    FailureOr<Type> arrayTypeResult = getInferredType(op.getResult());
    if (failed(arrayTypeResult))
      return errorCollector.recordError(op.getLoc())
             << "failed to infer array type";

    auto arrayType = mlir::dyn_cast<csl::ArrayType>(*arrayTypeResult);
    if (!arrayType)
      return errorCollector.recordError(op.getLoc())
             << "result type is not an array type";

    Type expectedElemType = arrayType.getElementType();
    OperandRange elements = op.getElements();

    bool modified = false;

    for (auto [idx, elemValue] : llvm::enumerate(elements)) {
      FailureOr<Type> actualType = getInferredType(elemValue);
      if (failed(actualType))
        return errorCollector.recordError(op.getLoc())
               << "failed to infer type of element " << idx;

      if (*actualType != expectedElemType) {
        FailureOr<Value> castedElem = insertImplicitCast(
            rewriter, op.getLoc(), elemValue, expectedElemType);
        if (failed(castedElem))
          return failure();

        rewriter.modifyOpInPlace(
            op, [&]() { op.getElementsMutable()[idx].assign(*castedElem); });
        modified = true;
      }
    }

    return modified ? success()
                    : rewriter.notifyMatchFailure(op, "types already match");
  }
};

/// Normalize IndexAccessOp indices:
/// - For array access: indices are normalized to u16
/// - For struct/tuple access: index is normalized to comptime_int
struct NormalizeIndexAccessOpPattern
    : public NormalizationPattern<IndexAccessOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(IndexAccessOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Get base type to determine array vs struct/tuple
    FailureOr<Type> baseTypeResult = getInferredType(op.getBase());
    if (failed(baseTypeResult))
      return rewriter.notifyMatchFailure(op, "base type not inferred");

    Type valueType = *baseTypeResult;
    if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(valueType))
      valueType = lvalueType.getValueType();

    // Determine target type based on base type
    Type targetIndexType;
    if (mlir::isa<csl::ArrayType>(valueType)) {
      // Array indices should be u16
      targetIndexType =
          IntegerType::get(rewriter.getContext(), 16, IntegerType::Unsigned);
    } else if (mlir::isa<csl::StructType>(valueType)) {
      // Struct/tuple indices should be comptime_int
      targetIndexType = csl::ComptimeIntType::get(rewriter.getContext());
    } else {
      return rewriter.notifyMatchFailure(op, "unknown base type for indexing");
    }

    bool modified = false;

    for (auto [idx, indexOperand] : llvm::enumerate(op.getIndices())) {
      FailureOr<Type> indexType = getInferredType(indexOperand);
      if (failed(indexType))
        return errorCollector.recordError(op.getLoc())
               << "failed to infer type of index " << idx;

      // Check that index is an integer type
      if (!csl::isAnyIntegerType(*indexType)) {
        return errorCollector.recordError(op.getLoc())
               << "index must be an integer type, got "
               << csl::prettyPrintType(*indexType);
      }

      // If index is not the target type, insert a cast
      if (*indexType != targetIndexType) {
        // We already made sure index is an integer type, so allow explicit cast
        // to allow for signedness and bitwidth changes
        FailureOr<Value> castedIndex =
            insertImplicitCast(rewriter, op.getLoc(), indexOperand,
                               targetIndexType, /*allowExplicitCast=*/true);
        if (failed(castedIndex))
          return failure();

        rewriter.modifyOpInPlace(
            op, [&]() { op.getIndicesMutable()[idx].assign(*castedIndex); });
        modified = true;
      }
    }

    return modified
               ? success()
               : rewriter.notifyMatchFailure(op, "indices already normalized");
  }
};

/// Normalize BuiltinCallOp argument types using signature matching
struct NormalizeBuiltinCallOpPattern
    : public NormalizationPattern<BuiltinCallOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(BuiltinCallOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Gather inferred argument types
    SmallVector<TypedAttr> argValues;
    for (Value arg : op.getArgs()) {
      if (auto constOp = arg.getDefiningOp<ConstantOp>()) {
        auto typedAttr = constOp.getValue();
        argValues.push_back(typedAttr);
        continue;
      } else if (auto builtinTypeOp = arg.getDefiningOp<BuiltinTypeOp>()) {
        argValues.push_back(builtinTypeOp.getValue());
        continue;
      }
      FailureOr<Type> argType = getInferredType(arg);
      if (failed(argType))
        return errorCollector.recordError(op.getLoc())
               << "failed to infer type of builtin argument";
      argValues.push_back(csl::NoValueAttr::get(*argType));
    }

    // Resolve builtin name to enum; skip if unknown (comptime eval will error)
    auto builtinKind = op.resolveBuiltin();
    if (!builtinKind)
      return rewriter.notifyMatchFailure(op, "unknown builtin");

    // Use signature matcher to get required casts
    csl::SignatureMatchResult matchResult = csl::BuiltinSignatureMatcher::match(
        *builtinKind, argValues, op->getContext());

    if (!matchResult.isValid)
      return errorCollector.recordError(op.getLoc())
             << matchResult.errorMessage;

    // No casts needed
    if (matchResult.requiredCasts.empty())
      return rewriter.notifyMatchFailure(op, "no casts needed");

    // Insert required implicit casts
    for (auto [argIdx, targetType] : matchResult.requiredCasts) {
      Value arg = op.getArgs()[argIdx];
      FailureOr<Value> castedArg =
          insertImplicitCast(rewriter, op.getLoc(), arg, targetType);
      if (failed(castedArg))
        return failure();

      rewriter.modifyOpInPlace(
          op, [&]() { op.getArgsMutable()[argIdx].assign(*castedArg); });
    }

    return success();
  }
};

/// Normalize struct-to-struct cast by extracting fields and reconstructing
/// When casting a struct, we extract each field, cast it individually,
/// and construct a new struct with the casted fields.
struct NormalizeStructCastPattern : public NormalizationPattern<CastOp> {
  using NormalizationPattern::NormalizationPattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    // Stop if there are already errors to prevent duplicate diagnostics
    if (errorCollector.hasErrors())
      return failure();

    // Only handle implicit casts
    if (!op.isImplicit())
      return rewriter.notifyMatchFailure(op, "not an implicit cast");

    // Get source and target types
    FailureOr<Type> sourceTypeResult = getInferredType(op.getSource());
    FailureOr<Type> targetTypeResult = getInferredType(op.getResult());
    if (failed(sourceTypeResult) || failed(targetTypeResult))
      return rewriter.notifyMatchFailure(op, "types not inferred");

    auto sourceStructType = mlir::dyn_cast<csl::StructType>(*sourceTypeResult);
    auto targetStructType = mlir::dyn_cast<csl::StructType>(*targetTypeResult);

    // Only match struct-to-struct casts
    if (!sourceStructType || !targetStructType)
      return rewriter.notifyMatchFailure(op, "not a struct-to-struct cast");

    // Don't process if types are identical (would be a no-op)
    if (sourceStructType == targetStructType)
      return rewriter.notifyMatchFailure(op, "source and target types match");

    // Field counts must match
    if (sourceStructType.getNumFields() != targetStructType.getNumFields())
      return errorCollector.recordError(op.getLoc())
             << "struct field count mismatch: source has "
             << sourceStructType.getNumFields() << " fields, target has "
             << targetStructType.getNumFields();

    Location loc = op.getLoc();
    Value source = op.getSource();

    SmallVector<Value> castedFieldValues;
    SmallVector<Attribute> fieldNames;

    // Fields of a tuple match by position, fields of a struct by name.
    bool targetIsTuple = targetStructType.isTuple();
    if (targetIsTuple && !sourceStructType.isTuple())
      return errorCollector.recordError(loc)
             << "cannot cast struct '" << csl::prettyPrintType(sourceStructType)
             << "' to tuple '" << csl::prettyPrintType(targetStructType)
             << "', because its fields have names";

    // Process each field
    for (size_t i = 0; i < targetStructType.getNumFields(); ++i) {
      StringRef fieldName;
      if (!targetIsTuple)
        fieldName = *targetStructType.getFieldName(i);

      size_t sourceIndex = i;
      if (!sourceStructType.isTuple()) {
        FailureOr<size_t> matchingIndex =
            sourceStructType.getFieldIndex(fieldName);
        if (failed(matchingIndex))
          return errorCollector.recordError(loc)
                 << "source struct has no field named '" << fieldName << "'";
        sourceIndex = *matchingIndex;
      }

      Type sourceFieldType = sourceStructType.getFieldType(sourceIndex);
      Type targetFieldType = targetStructType.getFieldType(i);

      // Create field access operation to extract the field from source
      Value fieldAccess;
      if (sourceStructType.isTuple()) {
        fieldAccess = IndexAccessOp::create(
            rewriter, loc, source, ValueRange{},
            /*inferred_type=*/mlir::TypeAttr::get(sourceFieldType),
            /*constant_indices=*/
            rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(sourceIndex)}));
      } else {
        fieldAccess = FieldAccessOp::create(
            rewriter, loc, source, fieldName,
            /*inferred_type=*/mlir::TypeAttr::get(sourceFieldType),
            /*inferred_base_type=*/mlir::TypeAttr::get(sourceStructType));
      }

      Value fieldValue = fieldAccess;

      // Cast the field if types differ
      if (sourceFieldType != targetFieldType) {
        FailureOr<Value> castedField =
            insertImplicitCast(rewriter, loc, fieldValue, targetFieldType);
        if (failed(castedField))
          return failure();
        fieldValue = *castedField;
      }

      castedFieldValues.push_back(fieldValue);
      if (!targetIsTuple)
        fieldNames.push_back(rewriter.getStringAttr(fieldName));
    }

    if (targetIsTuple) {
      rewriter.replaceOp(op, ConstructTupleOp::create(
                                 rewriter, loc, castedFieldValues,
                                 mlir::TypeAttr::get(targetStructType)));
      return success();
    }

    // Create new struct with casted fields
    // Use the target type SSA value if it's a named struct
    Value newStruct = ConstructStructOp::create(
        rewriter, loc, rewriter.getArrayAttr(fieldNames),
        /*struct_type=*/op.getTargetType(), castedFieldValues,
        /*inferred_type=*/mlir::TypeAttr::get(targetStructType));

    rewriter.replaceOp(op, newStruct);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Normalize Casts Pass
//===----------------------------------------------------------------------===//

struct NormalizeCastsPass
    : public parsed_csl::impl::NormalizeCastsPassBase<NormalizeCastsPass> {

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();

    MLIRContext *context = &getContext();
    ErrorCollector errorCollector;

    //===------------------------------------------------------------------===//
    // Single-Phase: All normalization patterns to fixed-point
    //===------------------------------------------------------------------===//
    // The greedy rewriter handles cascading transformations automatically:
    // 1. NormalizeVariableOp inserts: var x = cast(select(...), type)
    // 2. PushCastIntoSelectPattern creates: var x = select(c, cast(t),
    // cast(f))
    // 3. FoldCastOpPattern folds each cast to constants
    // 4. Repeats until no patterns match (fixed-point)
    //===------------------------------------------------------------------===//

    RewritePatternSet patterns(context);

    // Add folding/pushing patterns with higher benefit (run first when
    // possible)
    patterns.add<FoldCastOpPattern, PushCastIntoSelectPattern,
                 PushCastIntoSwitchPattern, PushCastIntoLabeledBlockPattern>(
        context, errorCollector);

    // Add implicit cast insertion patterns with default benefit
    patterns.add<NormalizeBinaryOpPattern, NormalizeSelectOpPattern,
                 NormalizeWhileOpPattern, NormalizeVariableOpPattern,
                 NormalizeAssignOpPattern, NormalizeCallOpPattern,
                 NormalizeReturnOpPattern, NormalizeLabeledBlockOpPattern,
                 NormalizeConstructTupleOpPattern,
                 NormalizeConstructStructOpPattern,
                 NormalizeConstructArrayOpPattern, NormalizeStructCastPattern,
                 NormalizeBuiltinCallOpPattern, NormalizeIndexAccessOpPattern>(
        context, errorCollector);

    GreedyRewriteConfig config;
    config.enableFolding();

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPatternsGreedily(unit, frozenPatterns, config))) {
      // Either hit max iterations or a pattern failed
      signalPassFailure();
      return;
    }

    // Check if any errors were recorded during pattern application
    if (errorCollector.hasErrors()) {
      errorCollector.emitAll();
      signalPassFailure();
      return;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

std::unique_ptr<Pass> createNormalizeCastsPass() {
  return std::make_unique<NormalizeCastsPass>();
}

} // namespace parsed_csl
} // namespace mlir
