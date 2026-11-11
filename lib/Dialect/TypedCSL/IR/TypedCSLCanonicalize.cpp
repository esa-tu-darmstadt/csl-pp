//===- TypedCSLCanonicalize.cpp - TypedCSL Canonicalization Patterns ------===//
//
// Canonicalization patterns for TypedCSL control flow operations.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::typed_csl;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Returns true if all non-terminator ops in the block would be trivially dead
/// (i.e., have no side effects that prevent removal). This is used to check
/// whether a region branch can be safely discarded.
static bool blockIsSideEffectFree(Block &block) {
  for (Operation &op : block.without_terminator())
    if (!wouldOpBeTriviallyDead(&op))
      return false;
  return true;
}

/// Push a transformation into all terminators of a RegionBranchOpInterface op.
/// For each terminator found by collectNestedRegionExits that yields a
/// value to the parent, calls `createOp(rewriter, loc, yieldedValue)` to
/// produce a replacement value, then updates the terminator and the parent
/// op's result type.
///
/// Returns failure if `base` is not the single-use result of a
/// RegionBranchOpInterface op, or if no predecessors are found.
static LogicalResult pushOpIntoRegionTerminators(
    PatternRewriter &rewriter, Operation *consumerOp, Value base,
    Type newResultType,
    function_ref<Value(PatternRewriter &, Location, Value)> createOp) {
  auto defOp = base.getDefiningOp();
  if (!defOp || !mlir::isa<RegionBranchOpInterface>(defOp))
    return rewriter.notifyMatchFailure(
        consumerOp, "not defined by a RegionBranchOpInterface op");

  if (!base.hasOneUse())
    return rewriter.notifyMatchFailure(consumerOp, "base has multiple uses");

  // Find all terminators that yield a value to the parent op.
  SmallVector<Operation *> predecessors;
  collectNestedRegionExits(defOp, predecessors);

  if (predecessors.empty())
    return rewriter.notifyMatchFailure(consumerOp, "no predecessors found");

  for (Operation *pred : predecessors) {
    auto branchTerminator = mlir::cast<RegionBranchTerminatorOpInterface>(pred);
    MutableOperandRange operands =
        branchTerminator.getMutableSuccessorOperands(RegionSuccessor(defOp));
    if (operands.size() == 0)
      continue;

    Value yielded = static_cast<OperandRange>(operands)[0];
    rewriter.setInsertionPoint(pred);
    Value newValue = createOp(rewriter, consumerOp->getLoc(), yielded);
    operands.assign(newValue);
  }

  // Update the parent op's result type and replace the consumer op.
  rewriter.modifyOpInPlace(defOp, [&] { base.setType(newResultType); });
  rewriter.replaceOp(consumerOp, base);
  return success();
}

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

/// Replace a structured control flow op with the contents of one of its
/// regions, following the same pattern as SCF's replaceOpWithRegion.
///
/// The region terminator must be the only region exit addressed to `op`:
/// a nested break or continue addressed to `op` has no target once `op` is
/// gone. Callers must reject such ops.
///
/// Before inlining, decrements the number of breaking regions of all nested
/// region exits that break through the op being removed (via
/// visitNestedRegionExits, which skips terminators caught by intermediate ops
/// like for/while/labeled_block).
///
/// After inlining, if the block terminator breaks through the removed op to
/// an enclosing op, all ops after the removed parent in its block are dead
/// and get erased.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block *block = &region.front();

  // Moving a region into its parent removes a CSL lexical scope. Keep local
  // declaration names hygienic: the emitted source may otherwise contain two
  // declarations with the same name even though the original program legally
  // declared one of them in a nested scope.
  llvm::StringSet<> usedNames;
  for (Operation &sibling : *op->getBlock()) {
    if (auto variable = dyn_cast<VariableOp>(sibling))
      usedNames.insert(variable.getName());
  }
  for (Operation &nested : block->without_terminator()) {
    auto variable = dyn_cast<VariableOp>(nested);
    if (!variable)
      continue;

    StringRef originalName = variable.getName();
    if (usedNames.insert(originalName).second)
      continue;

    unsigned suffix = 1;
    std::string uniqueName;
    do {
      uniqueName = (originalName + "__scope_" + Twine(suffix++)).str();
    } while (!usedNames.insert(uniqueName).second);
    rewriter.modifyOpInPlace(variable, [&]() { variable.setName(uniqueName); });
  }

  Operation *terminator = block->getTerminator();
  Operation *terminatorTarget = getRegionExitTarget(terminator);
  bool breaksThroughOp = terminatorTarget && terminatorTarget != op;

  visitNestedRegionExits(op, [&](Operation *exit, int) {
    if (getRegionExitTarget(exit) == op) {
      assert(exit == terminator &&
             "only the region terminator may exit to the replaced op");
      return;
    }
    auto breaking = cast<BreakingTerminatorOpInterface>(exit);
    rewriter.modifyOpInPlace(exit, [&] {
      breaking.setNumBreakingRegions(breaking.getNumBreakingRegions() - 1);
    });
  });

  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);

  if (!breaksThroughOp) {
    // Normal case: terminator was a yield/fallthrough.
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
  } else {
    // Early exit case: terminator is still breaking (e.g. return 2 → return 1).
    // Everything from the parent op onward in its block is now dead.
    Operation *toDelete = &op->getBlock()->back();
    Operation *prevOp = toDelete;
    do {
      toDelete = prevOp;
      prevOp = prevOp->getPrevNode();
      toDelete->dropAllUses();
      rewriter.eraseOp(toDelete);
    } while (toDelete != op);
  }
}

//===----------------------------------------------------------------------===//
// DereferenceOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Fold dereference(address_of(x)) -> x
struct FoldDereferenceOfAddressOf : public OpRewritePattern<DereferenceOp> {
  using OpRewritePattern<DereferenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DereferenceOp op,
                                PatternRewriter &rewriter) const override {
    auto addressOf = op.getOperand().getDefiningOp<AddressOfOp>();
    if (!addressOf)
      return failure();

    rewriter.replaceOp(op, addressOf.getOperand());
    return success();
  }
};
} // namespace

void DereferenceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<FoldDereferenceOfAddressOf>(context);
}

//===----------------------------------------------------------------------===//
// IfOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Remove if operations with constant conditions.
/// - If condition is true: inline the then region
/// - If condition is false with else: inline the else region
/// - If condition is false without else: erase the op
struct RemoveIfWithConstantCondition : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    BoolAttr condAttr;
    if (!matchPattern(op.getCondition(), m_Constant(&condAttr)))
      return failure();

    if (condAttr.getValue()) {
      replaceOpWithRegion(rewriter, op, op.getThenRegion());
    } else if (!op.getElseRegion().empty()) {
      replaceOpWithRegion(rewriter, op, op.getElseRegion());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// Simplify boolean if-expressions with constant branches to short-circuit
/// operations or simpler boolean expressions.
///
///   if(c) { yield x } else { yield false }     →  short_circuit c and { yield
///   x } if(c) { yield true } else { yield x }      →  short_circuit c or  {
///   yield x } if(c) { yield true } else { yield false }   →  c if(c) { yield
///   false } else { yield true }   →  binary ne, c, true if(c) { yield K } else
///   { yield K }           →  K
struct SimplifyBoolIfExpression : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.isExpression() || op.getElseRegion().empty())
      return failure();

    // Only applies to bool (i1) results.
    if (op.getResult().getType() != rewriter.getI1Type())
      return failure();

    auto thenYield =
        mlir::cast<YieldOp>(op.getThenRegion().front().getTerminator());
    auto elseYield =
        mlir::cast<YieldOp>(op.getElseRegion().front().getTerminator());

    BoolAttr thenAttr, elseAttr;
    bool thenIsConst =
        matchPattern(thenYield.getValue(), m_Constant(&thenAttr));
    bool elseIsConst =
        matchPattern(elseYield.getValue(), m_Constant(&elseAttr));

    if (!thenIsConst && !elseIsConst)
      return failure();

    Location loc = op.getLoc();

    // Both branches yield constants — both must be side-effect-free to fold.
    // If side-effect check fails, fall through to the single-constant cases.
    if (thenIsConst && elseIsConst &&
        blockIsSideEffectFree(op.getThenRegion().front()) &&
        blockIsSideEffectFree(op.getElseRegion().front())) {
      if (thenAttr.getValue() == elseAttr.getValue()) {
        // Same constant on both sides → just the constant.
        Value result =
            ConstantOp::create(rewriter, loc, rewriter.getI1Type(), thenAttr);
        rewriter.replaceOp(op, result);
      } else if (thenAttr.getValue()) {
        // if(c) true else false → c
        rewriter.replaceOp(op, op.getCondition());
      } else {
        // if(c) false else true → !c
        Value neg =
            UnaryOp::create(rewriter, loc, rewriter.getI1Type(),
                            csl::UnaryOpKind::LogicalNot, op.getCondition());
        rewriter.replaceOp(op, neg);
      }
      return success();
    }

    // if(c) { yield x } else { yield false } → short_circuit c and { yield x }
    // The discarded else branch must be side-effect-free.
    if (elseIsConst && !elseAttr.getValue()) {
      if (!blockIsSideEffectFree(op.getElseRegion().front()))
        return failure();
      auto sc = ShortCircuitOp::create(
          rewriter, loc, csl::ShortCircuitKind::And, op.getCondition());
      rewriter.inlineRegionBefore(op.getThenRegion(), sc.getRhsRegion(),
                                  sc.getRhsRegion().end());
      rewriter.replaceOp(op, sc.getResult());
      return success();
    }

    // if(c) { yield true } else { yield x } → short_circuit c or { yield x }
    // The discarded then branch must be side-effect-free.
    if (thenIsConst && thenAttr.getValue()) {
      if (!blockIsSideEffectFree(op.getThenRegion().front()))
        return failure();
      auto sc = ShortCircuitOp::create(rewriter, loc, csl::ShortCircuitKind::Or,
                                       op.getCondition());
      rewriter.inlineRegionBefore(op.getElseRegion(), sc.getRhsRegion(),
                                  sc.getRhsRegion().end());
      rewriter.replaceOp(op, sc.getResult());
      return success();
    }

    return failure();
  }
};

/// Remove empty else regions from statement-form if ops.
/// An else region is empty if it contains only a valueless yield.
///
///   if(c) { ... } else { yield }  →  if(c) { ... }
struct RemoveEmptyElse : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.isExpression() || op.getElseRegion().empty())
      return failure();

    Block &elseBlock = op.getElseRegion().front();
    // Empty means: single block with just a valueless yield terminator.
    if (elseBlock.getOperations().size() != 1)
      return failure();
    if (!mlir::isa<YieldOp>(elseBlock.front()))
      return failure();

    rewriter.modifyOpInPlace(op, [&] { rewriter.eraseBlock(&elseBlock); });
    return success();
  }
};
/// Swap branches when the then region is empty but the else is not.
/// Negates the condition and moves the else body into the then region.
///
///   if(c) { yield } else { ... yield }  →  if(!c) { ... yield }
struct SwapBranchesOnEmptyThen : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.isExpression() || op.getElseRegion().empty())
      return failure();

    Block &thenBlock = op.getThenRegion().front();
    Block &elseBlock = op.getElseRegion().front();

    // Then must be empty (just a valueless yield), else must not be.
    if (thenBlock.getOperations().size() != 1 ||
        !mlir::isa<YieldOp>(thenBlock.front()) ||
        elseBlock.getOperations().size() == 1)
      return failure();

    // Negate the condition.
    rewriter.setInsertionPoint(op);
    Value negCond =
        UnaryOp::create(rewriter, op.getLoc(), rewriter.getI1Type(),
                        csl::UnaryOpKind::LogicalNot, op.getCondition());

    // Create new if with negated condition, else body becomes then body.
    auto newIf = IfOp::create(rewriter, op.getLoc(), TypeRange{}, negCond);
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<RemoveIfWithConstantCondition, SimplifyBoolIfExpression,
               RemoveEmptyElse, SwapBranchesOnEmptyThen>(context);
}

//===----------------------------------------------------------------------===//
// WhileOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Remove while loops with constant false conditions.
/// While loops with constant true conditions are infinite loops and should
/// NOT be canonicalized.
struct RemoveWhileWithFalseCondition : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the condition region yields a constant false value.
    // The cond_region should contain just a constant and a yield.
    Region &condRegion = op.getCondRegion();
    if (condRegion.empty())
      return failure();

    Block &condBlock = condRegion.front();
    auto condYieldOp = dyn_cast<CondYieldOp>(condBlock.getTerminator());
    if (!condYieldOp)
      return failure();

    // Check if yielded value is a constant boolean
    BoolAttr condAttr;
    if (!matchPattern(condYieldOp.getCondition(), m_Constant(&condAttr)))
      return failure();

    // Only canonicalize constant false - constant true is infinite loop
    if (condAttr.getValue())
      return failure();

    // Condition is false: loop never executes, just erase
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<RemoveWhileWithFalseCondition>(context);
}

//===----------------------------------------------------------------------===//
// ForOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify for loops with constant range bounds.
/// - If range is empty (0 iterations): erase the loop
/// - If range has single element (1 iteration): inline the body
struct SimplifyTrivialForLoop : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    // The iterable must be defined by a constant op with RangeAttr
    auto definingOp = op.getIterable().getDefiningOp<ConstantOp>();
    if (!definingOp)
      return failure();

    auto rangeValue = mlir::dyn_cast<csl::RangeAttr>(definingOp.getValue());
    if (!rangeValue)
      return failure();

    // Compute trip count
    int64_t start = rangeValue.getStart();
    int64_t stop = rangeValue.getStop();
    int64_t step = rangeValue.getStep();

    auto tripCount = computeTripCount(start, stop, step);
    if (!tripCount)
      return failure();

    if (*tripCount == 0) {
      // Empty range: loop never executes, just erase
      rewriter.eraseOp(op);
      return success();
    }

    if (*tripCount == 1) {
      // A break or continue nested inside the body that addresses this loop
      // skips the rest of the body, which the inlined body cannot express.
      Operation *bodyTerminator = op.getBody().front().getTerminator();
      SmallVector<Operation *> exits;
      collectNestedRegionExits(op, exits);
      if (llvm::any_of(exits,
                       [&](Operation *exit) { return exit != bodyTerminator; }))
        return rewriter.notifyMatchFailure(
            op, "body has a nested break or continue addressed to the loop");

      // Single iteration: inline the body with element = start
      Type elemType = rangeValue.getElementType();
      Location loc = op.getLoc();

      // Create constant for the single element value
      TypedAttr elemAttr = rewriter.getIntegerAttr(elemType, start);
      Value elemValue = ConstantOp::create(rewriter, loc, elemAttr);

      // Build block arguments: element value, and optionally index (0)
      SmallVector<Value> blockArgs;
      blockArgs.push_back(elemValue);

      if (op.hasIndex()) {
        Type indexType = rewriter.getIntegerType(16, /*isSigned=*/false);
        TypedAttr indexAttr = rewriter.getIntegerAttr(indexType, 0);
        Value indexValue = ConstantOp::create(rewriter, loc, indexAttr);
        blockArgs.push_back(indexValue);
      }

      replaceOpWithRegion(rewriter, op, op.getBody(), blockArgs);
      return success();
    }

    // Trip count > 1: no canonicalization
    return failure();
  }
};
} // namespace

void ForOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<SimplifyTrivialForLoop>(context);
}

//===----------------------------------------------------------------------===//
// ShortCircuitOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify short-circuit operations with constant LHS.
/// For && (And):
///   - If LHS is false: result is false (don't evaluate RHS)
///   - If LHS is true: inline RHS region and use yielded value
/// For || (Or):
///   - If LHS is true: result is true (don't evaluate RHS)
///   - If LHS is false: inline RHS region and use yielded value
struct SimplifyShortCircuitWithConstantLhs
    : public OpRewritePattern<ShortCircuitOp> {
  using OpRewritePattern<ShortCircuitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShortCircuitOp op,
                                PatternRewriter &rewriter) const override {
    // Check if LHS is a constant boolean
    BoolAttr lhsAttr;
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)))
      return failure();

    bool lhsValue = lhsAttr.getValue();
    bool isAnd = op.getKind() == csl::ShortCircuitKind::And;

    // Determine if we should short-circuit (skip RHS evaluation)
    bool shouldShortCircuit = isAnd ? !lhsValue : lhsValue;

    if (shouldShortCircuit) {
      // Short-circuit: result is the LHS value (false for &&, true for ||)
      auto boolType = rewriter.getI1Type();
      auto resultAttr = rewriter.getBoolAttr(lhsValue);
      Value result =
          ConstantOp::create(rewriter, op.getLoc(), boolType, resultAttr);
      rewriter.replaceOp(op, result);
    } else {
      replaceOpWithRegion(rewriter, op, op.getRhsRegion());
    }

    return success();
  }
};
/// Simplify short-circuit operations whose RHS region just yields a constant.
///
///   x and true  → x       (identity)
///   x or  false → x       (identity)
///   x and false → false   (absorbing, RHS not needed)
///   x or  true  → true    (absorbing, RHS not needed)
struct SimplifyShortCircuitWithConstantRhs
    : public OpRewritePattern<ShortCircuitOp> {
  using OpRewritePattern<ShortCircuitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShortCircuitOp op,
                                PatternRewriter &rewriter) const override {
    Block &rhsBlock = op.getRhsRegion().front();
    auto yieldOp = mlir::cast<YieldOp>(rhsBlock.getTerminator());

    BoolAttr rhsAttr;
    if (!matchPattern(yieldOp.getValue(), m_Constant(&rhsAttr)))
      return failure();

    bool rhsValue = rhsAttr.getValue();
    bool isAnd = op.getKind() == csl::ShortCircuitKind::And;

    // The RHS region is being discarded, so it must be side-effect-free.
    if (!blockIsSideEffectFree(rhsBlock))
      return failure();

    // Identity: (x and true) or (x or false) → x
    bool isIdentity = isAnd ? rhsValue : !rhsValue;
    if (isIdentity) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // Absorbing: (x and false) → false, (x or true) → true
    Value result =
        ConstantOp::create(rewriter, op.getLoc(), rewriter.getI1Type(),
                           rewriter.getBoolAttr(rhsValue));
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void ShortCircuitOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<SimplifyShortCircuitWithConstantLhs,
               SimplifyShortCircuitWithConstantRhs>(context);
}

//===----------------------------------------------------------------------===//
// SwitchOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify switch operations with constant input.
/// Find the matching case and inline that region, or the else region if no
/// match.
struct SimplifySwitchWithConstantInput : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp op,
                                PatternRewriter &rewriter) const override {
    // Check if input is a constant integer
    IntegerAttr inputAttr;
    if (!matchPattern(op.getInput(), m_Constant(&inputAttr)))
      return failure();

    APInt inputValue = inputAttr.getValue();

    // Find matching case
    ArrayAttr caseValuesAttr = op.getCaseValues();
    std::optional<size_t> matchingCaseIdx;

    for (size_t i = 0; i < caseValuesAttr.size(); ++i) {
      auto thisCaseValues = mlir::cast<ArrayAttr>(caseValuesAttr[i]);
      for (Attribute caseVal : thisCaseValues) {
        auto caseIntAttr = mlir::cast<IntegerAttr>(caseVal);
        if (caseIntAttr.getValue() == inputValue) {
          matchingCaseIdx = i;
          break;
        }
      }
      if (matchingCaseIdx)
        break;
    }

    // Get the region to inline
    Region *regionToInline;
    if (matchingCaseIdx) {
      regionToInline = &op.getCaseRegions()[*matchingCaseIdx];
    } else if (op.hasElseBranch()) {
      regionToInline = &op.getElseRegion();
    } else {
      // No match and no else - this should be a verification error
      return failure();
    }

    replaceOpWithRegion(rewriter, op, *regionToInline);
    return success();
  }
};
} // namespace

void SwitchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<SimplifySwitchWithConstantInput>(context);
}

//===----------------------------------------------------------------------===//
// LabeledBlockOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Drop break values and result when the labeled block result is unused.
/// Replaces a valued labeled block with a void one.
struct DropUnusedLabeledBlockValue : public OpRewritePattern<LabeledBlockOp> {
  using OpRewritePattern<LabeledBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LabeledBlockOp op,
                                PatternRewriter &rewriter) const override {
    // Only applies to valued blocks whose result is unused
    if (op.getNumResults() == 0 || !op.getResult().use_empty())
      return failure();

    // Drop value from all breaks targeting this block
    StringRef myLabel = op.getLabel();
    op.getBody().walk([&](typed_csl::BreakOp breakOp) {
      if (breakOp.getTargetLabel() && *breakOp.getTargetLabel() == myLabel &&
          breakOp.getValue()) {
        rewriter.modifyOpInPlace(breakOp,
                                 [&]() { breakOp.getValueMutable().clear(); });
      }
    });

    // Create a new void labeled block (no result)
    auto newOp = LabeledBlockOp::create(rewriter, op.getLoc(), TypeRange{},
                                        op.getLabelAttr());

    // Move body region
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Inline a labeled block when either (a) no breaks target it, or (b) the only
/// break targeting it is the last op in the body. Handles both valued and
/// valueless breaks: for valued breaks, replaces the block result uses with the
/// break value before inlining.
struct InlineSimpleLabeledBlock : public OpRewritePattern<LabeledBlockOp> {
  using OpRewritePattern<LabeledBlockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LabeledBlockOp op,
                                PatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();

    // Collect all terminators that directly target this labeled block.
    SmallVector<Operation *> predecessors;
    collectNestedRegionExits(op, predecessors);

    // Case (a): no predecessors — block just falls through, inline it.
    if (predecessors.empty()) {
      if (op.getNumResults() != 0 && !op.getResult().use_empty())
        return failure();
      replaceOpWithRegion(rewriter, op, op.getBody());
      return success();
    }

    // Case (b): exactly one break, and it's the last op in the body.
    // replaceOpWithRegion extracts the break's value as the result and erases
    // it.
    if (predecessors.size() != 1)
      return failure();
    auto trailingBreak = dyn_cast<BreakOp>(predecessors[0]);
    if (!trailingBreak || &body.back() != trailingBreak.getOperation())
      return failure();
    replaceOpWithRegion(rewriter, op, op.getBody());
    return success();
  }
};
} // namespace

void LabeledBlockOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<DropUnusedLabeledBlockValue, InlineSimpleLabeledBlock>(context);
}

//===----------------------------------------------------------------------===//
// Terminator Canonicalization (ReturnOp, UnreachableOp)
//===----------------------------------------------------------------------===//

namespace {
/// Erase dead ops after a terminator within the same block.
/// When canonicalization inlines a block ending with a terminator (e.g.,
/// folding `if (true) { return x; }`), subsequent ops in the parent block
/// become dead code after the terminator, which is illegal in MLIR.
template <typename TerminatorOpTy>
struct EraseOpsAfterTerminator : public OpRewritePattern<TerminatorOpTy> {
  using OpRewritePattern<TerminatorOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(TerminatorOpTy op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    if (&block->back() == op.getOperation())
      return failure();

    // Erase all ops after the terminator in reverse order.
    while (&block->back() != op.getOperation()) {
      Operation &deadOp = block->back();
      deadOp.dropAllUses();
      rewriter.eraseOp(&deadOp);
    }
    return success();
  }
};
} // namespace

void ReturnOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EraseOpsAfterTerminator<ReturnOp>>(context);
}

void UnreachableOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<EraseOpsAfterTerminator<UnreachableOp>>(context);
}

void BreakOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<EraseOpsAfterTerminator<BreakOp>>(context);
}

void ContinueOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EraseOpsAfterTerminator<ContinueOp>>(context);
}

//===----------------------------------------------------------------------===//
// BinaryOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Push a binary op with a constant operand into the regions of a
/// RegionBranchOpInterface op that produces the other operand.
///
/// Example:
///   %val = typed_csl.if %cond {
///     typed_csl.yield %a : i32
///   } else {
///     typed_csl.yield %b : i32
///   } : i32
///   %cmp = typed_csl.binary ne, %val, %c0 : (i32, i32) -> i1
/// Becomes:
///   %cmp = typed_csl.if %cond {
///     %r = typed_csl.binary ne, %a, %c0 : (i32, i32) -> i1
///     typed_csl.yield %r : i1
///   } else {
///     %r = typed_csl.binary ne, %b, %c0 : (i32, i32) -> i1
///     typed_csl.yield %r : i1
///   } : i1
struct PushBinaryOpIntoRegions : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    csl::BinaryOpKind opcode = op.getOpcode();
    Type resultType = op.getResult().getType();

    // Determine which operand is the region result and which is the constant.
    Value regionResult;
    Value constantOperand;
    bool regionIsLhs;

    if (lhs.getDefiningOp() &&
        mlir::isa<RegionBranchOpInterface>(lhs.getDefiningOp()) &&
        rhs.getDefiningOp<ConstantOp>()) {
      regionResult = lhs;
      constantOperand = rhs;
      regionIsLhs = true;
    } else if (rhs.getDefiningOp() &&
               mlir::isa<RegionBranchOpInterface>(rhs.getDefiningOp()) &&
               lhs.getDefiningOp<ConstantOp>()) {
      regionResult = rhs;
      constantOperand = lhs;
      regionIsLhs = false;
    } else {
      return rewriter.notifyMatchFailure(
          op, "neither operand is a region result with constant other side");
    }

    return pushOpIntoRegionTerminators(
        rewriter, op, regionResult, resultType,
        [&](PatternRewriter &rw, Location loc, Value yielded) {
          Value newLhs = regionIsLhs ? yielded : constantOperand;
          Value newRhs = regionIsLhs ? constantOperand : yielded;
          return BinaryOp::create(rw, loc, resultType, opcode, newLhs, newRhs);
        });
  }
};
} // namespace

void BinaryOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<PushBinaryOpIntoRegions>(context);
}

//===----------------------------------------------------------------------===//
// FieldAccessOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify field_access of construct_struct to directly use the field value.
/// Example:
///   %struct = construct_struct ["a", "b"] %val_a, %val_b
///   %field = field_access %struct["a"]
/// Becomes:
///   %field = %val_a
struct SimplifyFieldAccessOfConstructStruct
    : public OpRewritePattern<FieldAccessOp> {
  using OpRewritePattern<FieldAccessOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FieldAccessOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle rvalue structs (not lvalues)
    Value structValue = op.getStructLvalue();
    if (mlir::isa<csl::LValueType>(structValue.getType()))
      return rewriter.notifyMatchFailure(op, "struct is an lvalue");

    // Check if the struct is defined by a ConstructStructOp
    auto constructOp = structValue.getDefiningOp<ConstructStructOp>();
    if (!constructOp)
      return rewriter.notifyMatchFailure(
          op, "struct not defined by construct_struct");

    // Find the field index
    StringRef fieldName = op.getFieldName();
    ArrayRef<Attribute> fieldNames = constructOp.getFieldNames().getValue();

    for (size_t i = 0; i < fieldNames.size(); ++i) {
      auto nameAttr = mlir::cast<StringAttr>(fieldNames[i]);
      if (nameAttr.getValue() == fieldName) {
        // Replace with the corresponding field value
        rewriter.replaceOp(op, constructOp.getFieldValues()[i]);
        return success();
      }
    }

    // Field not found (shouldn't happen if verification passed)
    return rewriter.notifyMatchFailure(op, "field not found in struct");
  }
};
} // namespace

void FieldAccessOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<SimplifyFieldAccessOfConstructStruct>(context);
}

//===----------------------------------------------------------------------===//
// TupleIndexAccessOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify tuple_index_access of construct_tuple/construct_struct to directly
/// use the element value.
struct SimplifyTupleIndexAccessOfConstruct
    : public OpRewritePattern<TupleIndexAccessOp> {
  using OpRewritePattern<TupleIndexAccessOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleIndexAccessOp op,
                                PatternRewriter &rewriter) const override {
    Value base = op.getBase();
    if (mlir::isa<csl::LValueType>(base.getType()))
      return rewriter.notifyMatchFailure(op, "base is an lvalue");

    int64_t index = op.getIndex();

    if (auto constructTupleOp = base.getDefiningOp<ConstructTupleOp>()) {
      rewriter.replaceOp(op, constructTupleOp.getFieldValues()[index]);
      return success();
    }

    if (auto constructStructOp = base.getDefiningOp<ConstructStructOp>()) {
      rewriter.replaceOp(op, constructStructOp.getFieldValues()[index]);
      return success();
    }

    return rewriter.notifyMatchFailure(
        op, "not defined by construct_tuple or construct_struct");
  }
};

/// Push tuple_index_access into the regions of a RegionBranchOpInterface op.
struct PushTupleIndexAccessIntoRegions
    : public OpRewritePattern<TupleIndexAccessOp> {
  using OpRewritePattern<TupleIndexAccessOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleIndexAccessOp op,
                                PatternRewriter &rewriter) const override {
    Value base = op.getBase();
    if (mlir::isa<csl::LValueType>(base.getType()))
      return rewriter.notifyMatchFailure(op, "base is an lvalue");

    int64_t tupleIndex = op.getIndex();
    Type elementType = op.getResult().getType();

    return pushOpIntoRegionTerminators(
        rewriter, op, base, elementType,
        [&](PatternRewriter &rw, Location loc, Value yielded) {
          return TupleIndexAccessOp::create(rw, loc, elementType, yielded,
                                            tupleIndex);
        });
  }
};
} // namespace

void TupleIndexAccessOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<SimplifyTupleIndexAccessOfConstruct,
               PushTupleIndexAccessIntoRegions>(context);
}

//===----------------------------------------------------------------------===//
// ConstructTupleOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Simplify construct_tuple that reconstructs an existing tuple by extracting
/// all elements sequentially from the same source.
///
///   construct_tuple(tuple_index_access(x, 0),
///                   tuple_index_access(x, 1),
///                   tuple_index_access(x, 2))  →  x
struct SimplifyIdentityTupleReconstruction
    : public OpRewritePattern<ConstructTupleOp> {
  using OpRewritePattern<ConstructTupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstructTupleOp op,
                                PatternRewriter &rewriter) const override {
    OperandRange fields = op.getFieldValues();
    if (fields.empty())
      return failure();

    // Check that every field is tuple_index_access from the same source,
    // with sequential indices 0..N-1.
    auto firstAccess = fields[0].getDefiningOp<TupleIndexAccessOp>();
    if (!firstAccess || firstAccess.getIndex() != 0)
      return failure();

    Value source = firstAccess.getBase();

    // The source type must match the result type (same tuple).
    if (source.getType() != op.getResult().getType())
      return failure();

    for (int64_t i = 1, e = fields.size(); i < e; ++i) {
      auto access = fields[i].getDefiningOp<TupleIndexAccessOp>();
      if (!access || access.getBase() != source || access.getIndex() != i)
        return failure();
    }

    rewriter.replaceOp(op, source);
    return success();
  }
};
} // namespace

void ConstructTupleOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<SimplifyIdentityTupleReconstruction>(context);
}

//===----------------------------------------------------------------------===//
// AssignOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {

/// Recreate a RegionBranchOpInterface op without its result, converting it
/// from expression form to statement form. Moves all regions from the old op
/// to the new one. Returns the new op, or nullptr if the op type is not
/// supported.
static Operation *recreateAsStatement(PatternRewriter &rewriter,
                                      Operation *defOp) {
  Location loc = defOp->getLoc();

  if (auto ifOp = mlir::dyn_cast<IfOp>(defOp)) {
    auto newIf = IfOp::create(rewriter, loc, TypeRange{}, ifOp.getCondition());
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().end());
    return newIf;
  }

  if (auto switchOp = mlir::dyn_cast<SwitchOp>(defOp)) {
    unsigned numCases = switchOp.getNumCases();
    auto newSwitch =
        SwitchOp::create(rewriter, loc, /*result=*/Type(), switchOp.getInput(),
                         switchOp.getCaseValues(), numCases);
    rewriter.inlineRegionBefore(switchOp.getElseRegion(),
                                newSwitch.getElseRegion(),
                                newSwitch.getElseRegion().end());
    for (unsigned i = 0; i < numCases; ++i) {
      rewriter.inlineRegionBefore(switchOp.getCaseRegions()[i],
                                  newSwitch.getCaseRegions()[i],
                                  newSwitch.getCaseRegions()[i].end());
    }
    return newSwitch;
  }

  if (auto labeledBlock = mlir::dyn_cast<LabeledBlockOp>(defOp)) {
    auto newBlock = LabeledBlockOp::create(rewriter, loc, TypeRange{},
                                           labeledBlock.getLabelAttr());
    rewriter.inlineRegionBefore(labeledBlock.getBody(), newBlock.getBody(),
                                newBlock.getBody().end());
    return newBlock;
  }

  return nullptr;
}

/// Push an assignment of a region result into the regions, converting the
/// region op from expression form to statement form.
///
/// Example:
///   %val = typed_csl.if %cond {
///     typed_csl.yield %a : i1
///   } else {
///     typed_csl.yield %b : i1
///   } : i1
///   typed_csl.assign %val to %lv : !csl.lvalue<mutable i1>
/// Becomes:
///   typed_csl.if %cond {
///     typed_csl.assign %a to %lv : !csl.lvalue<mutable i1>
///     typed_csl.yield
///   } else {
///     typed_csl.assign %b to %lv : !csl.lvalue<mutable i1>
///     typed_csl.yield
///   }
struct PushAssignIntoRegions : public OpRewritePattern<AssignOp> {
  using OpRewritePattern<AssignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    Value lvalue = op.getLvalue();

    auto defOp = value.getDefiningOp();
    if (!defOp || !mlir::isa<RegionBranchOpInterface>(defOp))
      return rewriter.notifyMatchFailure(
          op, "value not defined by a RegionBranchOpInterface op");

    if (!value.hasOneUse())
      return rewriter.notifyMatchFailure(op, "value has multiple uses");

    SmallVector<Operation *> predecessors;
    collectNestedRegionExits(defOp, predecessors);

    if (predecessors.empty())
      return rewriter.notifyMatchFailure(op, "no predecessors found");

    // Push assign before each terminator and clear the yield operand.
    for (Operation *pred : predecessors) {
      auto branchTerminator =
          mlir::cast<RegionBranchTerminatorOpInterface>(pred);
      MutableOperandRange operands =
          branchTerminator.getMutableSuccessorOperands(RegionSuccessor(defOp));
      if (operands.size() == 0)
        continue;

      Value yielded = static_cast<OperandRange>(operands)[0];
      rewriter.setInsertionPoint(pred);
      AssignOp::create(rewriter, op.getLoc(), lvalue, yielded);
      operands.clear();
    }

    // Recreate the parent op without the result (expression → statement).
    rewriter.setInsertionPoint(defOp);
    Operation *newOp = recreateAsStatement(rewriter, defOp);
    if (!newOp)
      return rewriter.notifyMatchFailure(
          op, "unsupported op type for statement conversion");

    rewriter.eraseOp(op);
    rewriter.eraseOp(defOp);
    return success();
  }
};
} // namespace

void AssignOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  // patterns.add<PushAssignIntoRegions>(context);
}

//===----------------------------------------------------------------------===//
// PtrIndexAccessOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Fold ptr_index_access(cast(address_of(array_lval)), idx) into
/// array_index_access(array_lval, idx).
///
/// This pattern eliminates the redundant cast chain produced when accessing
/// coroutine frame fields: the frame is a [N]u8 array, and the access pattern
/// takes its address (*[N]u8), casts to [*]u8, then indexes. We can index the
/// array directly instead.
struct FoldPtrIndexAccessOfCastAddressOf
    : public OpRewritePattern<PtrIndexAccessOp> {
  using OpRewritePattern<PtrIndexAccessOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PtrIndexAccessOp op,
                                PatternRewriter &rewriter) const override {
    // Check: base is a CastOp (pointer cast)
    auto castOp = op.getBase().getDefiningOp<CastOp>();
    if (!castOp)
      return failure();

    // Check: both source and dest of the cast are pointer types
    auto srcPtrTy =
        mlir::dyn_cast<csl::PointerType>(castOp.getSource().getType());
    auto dstPtrTy =
        mlir::dyn_cast<csl::PointerType>(castOp.getDest().getType());
    if (!srcPtrTy || !dstPtrTy)
      return failure();

    // Check: the cast is from *[N]T to [*]T (single-item ptr to array →
    // many-item ptr to element)
    if (srcPtrTy.getIsManyPtr() || !dstPtrTy.getIsManyPtr())
      return failure();

    auto srcArrayTy = mlir::dyn_cast<csl::ArrayType>(srcPtrTy.getPointeeType());
    if (!srcArrayTy)
      return failure();

    if (srcArrayTy.getElementType() != dstPtrTy.getPointeeType())
      return failure();

    // Check: the cast source is address_of(array_lval)
    auto addrOf = castOp.getSource().getDefiningOp<AddressOfOp>();
    if (!addrOf)
      return failure();

    // Replace with array_index_access on the original array lvalue.
    // Both produce lvalue<element_type>, so the result type is preserved.
    rewriter.replaceOpWithNewOp<ArrayIndexAccessOp>(
        op, op.getResult().getType(), addrOf.getOperand(),
        ValueRange{op.getIndex()});
    return success();
  }
};

/// Fold ptr_index_access(cast(address_of(ptr_index_access(base, outer)), *T →
/// [*]T), inner)
///   → ptr_index_access(base, outer + inner * scale)
///
/// Merges nested pointer indexing through a single-to-many pointer cast.
/// The scale factor accounts for element size differences: if the outer
/// base is [*]u8 and the inner is [*]u16, inner indices must be scaled by 2.
/// When both use the same element type (e.g., both [*]u16), scale is 1.
struct FoldNestedPtrIndexAccess : public OpRewritePattern<PtrIndexAccessOp> {
  using OpRewritePattern<PtrIndexAccessOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PtrIndexAccessOp op,
                                PatternRewriter &rewriter) const override {
    // Base must be a cast from *T to [*]T.
    auto castOp = op.getBase().getDefiningOp<CastOp>();
    if (!castOp)
      return failure();

    auto srcPtrTy =
        mlir::dyn_cast<csl::PointerType>(castOp.getSource().getType());
    auto dstPtrTy =
        mlir::dyn_cast<csl::PointerType>(castOp.getDest().getType());
    if (!srcPtrTy || !dstPtrTy)
      return failure();
    if (srcPtrTy.getIsManyPtr() || !dstPtrTy.getIsManyPtr())
      return failure();
    if (srcPtrTy.getPointeeType() != dstPtrTy.getPointeeType())
      return failure();

    // Cast source must be address_of(ptr_index_access(base, outer_idx)).
    auto addrOf = castOp.getSource().getDefiningOp<AddressOfOp>();
    if (!addrOf)
      return failure();
    auto outerIdx = addrOf.getOperand().getDefiningOp<PtrIndexAccessOp>();
    if (!outerIdx)
      return failure();

    // Determine the element types for inner and outer indexing.
    Type innerElemTy = dstPtrTy.getPointeeType();
    auto outerBasePtrTy =
        mlir::dyn_cast<csl::PointerType>(outerIdx.getBase().getType());
    if (!outerBasePtrTy)
      return failure();
    Type outerElemTy = outerBasePtrTy.getPointeeType();

    auto innerElemSize = csl::getSizeInBytes(innerElemTy);
    auto outerElemSize = csl::getSizeInBytes(outerElemTy);
    if (!innerElemSize || !outerElemSize)
      return failure();

    // The inner index is in units of innerElemTy, but the outer index is in
    // units of outerElemTy. Scale the inner index to match outer units.
    // scale = sizeof(innerElem) / sizeof(outerElem)
    // Only fold when the division is exact (no fractional element offsets).
    if (*innerElemSize % *outerElemSize != 0)
      return failure();
    uint32_t scale = *innerElemSize / *outerElemSize;

    Location loc = op.getLoc();
    Type idxTy = outerIdx.getIndex().getType();

    Value scaledInner = op.getIndex();
    if (scale != 1) {
      Value scaleVal =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(idxTy, scale));
      scaledInner =
          BinaryOp::create(rewriter, loc, idxTy, csl::BinaryOpKind::Mul,
                           op.getIndex(), scaleVal);
    }

    Value newOffset =
        BinaryOp::create(rewriter, loc, idxTy, csl::BinaryOpKind::Add,
                         outerIdx.getIndex(), scaledInner);

    rewriter.replaceOpWithNewOp<PtrIndexAccessOp>(
        op, op.getResult().getType(), outerIdx.getBase(), newOffset);
    return success();
  }
};
} // namespace

void PtrIndexAccessOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<FoldPtrIndexAccessOfCastAddressOf, FoldNestedPtrIndexAccess>(
      context);
}

//===----------------------------------------------------------------------===//
// AddressOfOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Fold address_of(array_index_access(dereference(cast(addr)), idx))
///   → addr  (when idx == 0 and types match)
///   → addr + idx * sizeof(T)  (general case, when addr is
///   address_of(ptr_index_access(base, offset)))
///
/// Matches the pattern produced by callee sub-frame access in coroutine frames:
///   &(@ptrcast(*[N]T, addr).*)[idx]
/// After FoldPtrIndexAccessOfCastAddressOf fires, the inner PtrIndexAccess
/// becomes ArrayIndexAccess, so we match that form here.
struct FoldAddressOfArrayIndexDeref : public OpRewritePattern<AddressOfOp> {
  using OpRewritePattern<AddressOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddressOfOp op,
                                PatternRewriter &rewriter) const override {
    // Match: address_of(array_index_access(deref(...), idx))
    auto arrayIdx = op.getOperand().getDefiningOp<ArrayIndexAccessOp>();
    if (!arrayIdx || arrayIdx.getIndices().size() != 1)
      return failure();

    auto derefOp = arrayIdx.getBase().getDefiningOp<DereferenceOp>();
    if (!derefOp)
      return failure();

    // The deref operand should be a cast from *T to *[N]T.
    auto castOp = derefOp.getOperand().getDefiningOp<CastOp>();
    if (!castOp)
      return failure();

    auto dstPtrTy =
        mlir::dyn_cast<csl::PointerType>(castOp.getDest().getType());
    if (!dstPtrTy || dstPtrTy.getIsManyPtr())
      return failure();
    auto arrayTy = mlir::dyn_cast<csl::ArrayType>(dstPtrTy.getPointeeType());
    if (!arrayTy)
      return failure();

    // The cast source must be a single-item pointer (*T) whose pointee
    // matches the array element type, so the address is the same byte.
    auto srcPtrTy =
        mlir::dyn_cast<csl::PointerType>(castOp.getSource().getType());
    if (!srcPtrTy || srcPtrTy.getIsManyPtr())
      return failure();
    if (srcPtrTy.getPointeeType() != arrayTy.getElementType())
      return failure();

    // Result type of this address_of must match the source pointer type
    // (both are *T pointing to the same element type).
    if (op.getResult().getType() != castOp.getSource().getType())
      return failure();

    Value idx = arrayIdx.getIndices()[0];
    Type elemTy = arrayTy.getElementType();
    auto elemSize = csl::getSizeInBytes(elemTy);
    if (!elemSize)
      return failure();

    // Check if idx is constant 0 — pure identity, no arithmetic needed.
    APInt idxVal;
    if (matchPattern(idx, m_ConstantInt(&idxVal)) && idxVal.isZero()) {
      rewriter.replaceOp(op, castOp.getSource());
      return success();
    }

    // General case: need to add idx * sizeof(T) to the source pointer's
    // offset. This requires the source to be address_of(ptr_index_access(base,
    // offset)) so we can merge the offsets.
    auto srcAddrOf = castOp.getSource().getDefiningOp<AddressOfOp>();
    if (!srcAddrOf)
      return failure();
    auto srcPtrIdx = srcAddrOf.getOperand().getDefiningOp<PtrIndexAccessOp>();
    if (!srcPtrIdx)
      return failure();

    Location loc = op.getLoc();
    Type idxTy = srcPtrIdx.getIndex().getType();

    // byteOffset = idx * sizeof(T), omit multiply when sizeof == 1.
    Value byteOffset = idx;
    if (*elemSize != 1) {
      Value scale =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(idxTy, *elemSize));
      byteOffset = BinaryOp::create(rewriter, loc, idxTy,
                                    csl::BinaryOpKind::Mul, idx, scale);
    }

    Value newOffset =
        BinaryOp::create(rewriter, loc, idxTy, csl::BinaryOpKind::Add,
                         srcPtrIdx.getIndex(), byteOffset);

    Value newAccess =
        PtrIndexAccessOp::create(rewriter, loc, srcPtrIdx.getResult().getType(),
                                 srcPtrIdx.getBase(), newOffset);
    Value newAddr =
        AddressOfOp::create(rewriter, loc, op.getResult().getType(), newAccess);

    rewriter.replaceOp(op, newAddr);
    return success();
  }
};

/// Fold address_of(dereference(ptr)) → ptr
///
/// Taking the address of a dereferenced pointer is an identity.
struct FoldAddressOfDereference : public OpRewritePattern<AddressOfOp> {
  using OpRewritePattern<AddressOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddressOfOp op,
                                PatternRewriter &rewriter) const override {
    auto derefOp = op.getOperand().getDefiningOp<DereferenceOp>();
    if (!derefOp)
      return failure();

    if (derefOp.getOperand().getType() != op.getResult().getType())
      return failure();

    rewriter.replaceOp(op, derefOp.getOperand());
    return success();
  }
};
} // namespace

void AddressOfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<FoldAddressOfArrayIndexDeref, FoldAddressOfDereference>(context);
}

//===----------------------------------------------------------------------===//
// CastOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Fold ptrcast(ptrcast(x : *A → *B) : *B → *C) → ptrcast(x : *A → *C)
///
/// Collapses chained pointer casts into a single cast. Both source and
/// destination must be pointer types.
struct FoldChainedPtrCast : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    // Both source and result must be pointer types.
    if (!mlir::isa<csl::PointerType>(op.getSource().getType()) ||
        !mlir::isa<csl::PointerType>(op.getDest().getType()))
      return failure();

    // Source must be another CastOp between pointer types.
    auto innerCast = op.getSource().getDefiningOp<CastOp>();
    if (!innerCast)
      return failure();
    if (!mlir::isa<csl::PointerType>(innerCast.getSource().getType()))
      return failure();

    // Replace with a single cast from the inner source to the outer dest type.
    rewriter.replaceOpWithNewOp<CastOp>(op, op.getDest().getType(),
                                        innerCast.getSource());
    return success();
  }
};
} // namespace

void CastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<FoldChainedPtrCast>(context);
}
