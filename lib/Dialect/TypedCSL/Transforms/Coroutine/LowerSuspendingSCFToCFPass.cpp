//===- LowerSuspendingSCFToCFPass.cpp - Lower SCF to CF ----------*- C++
//-*-===//
//
// Selectively lowers structured control flow ops that contain suspend points
// to the cf dialect. Non-suspending structured ops survive intact.
// Breaking terminators targeting decomposed regions are rewritten to cf.br.
// Returns are decremented. Yield terminators are replaced with cf.br.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "lower-suspending-scf-to-cf"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_LOWERSUSPENDINGSCFTOCFPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

/// Returns true if the operation transitively contains a SuspendOp, AwaitOp,
/// or a CallOp tagged with "coro.async_call" (sequential await lowered to a
/// single call under the nop.term convention).
static bool containsSuspend(Operation *op) {
  if (isa<SuspendOp, AwaitOp>(op))
    return true;
  if (isa<CallOp>(op) && op->hasAttr("coro.async_call"))
    return true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        if (containsSuspend(&nested))
          return true;
  return false;
}

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Decrement the number of breaking regions for all nested breaking
/// terminators that propagate through `op`. This must be called before
/// inlining regions.
static void decrementBreakingOps(Operation *op, PatternRewriter &rewriter) {
  visitNestedRegionExits(op, [&](Operation *visited, int level) {
    auto breaking = dyn_cast<BreakingTerminatorOpInterface>(visited);
    if (!breaking)
      return;
    if (int32_t n = breaking.getNumBreakingRegions(); n > level)
      rewriter.modifyOpInPlace(visited,
                               [&] { breaking.setNumBreakingRegions(n - 1); });
  });
}

/// For each block in `region`, if the block ends with YieldOp, replace it with
/// cf.br to `mergeBlock`, forwarding the yield's operands as branch args.
/// All other terminators are left in place (they were already decremented).
static void replaceYieldsWithBranch(Region &region, Block *mergeBlock,
                                    PatternRewriter &rewriter) {
  for (Block &block : region) {
    if (block.empty())
      continue;
    auto *term = block.getTerminator();
    if (auto yield = dyn_cast<YieldOp>(term)) {
      rewriter.setInsertionPoint(yield);
      SmallVector<Value> operands;
      if (yield.getValue())
        operands.push_back(yield.getValue());
      cf::BranchOp::create(rewriter, yield.getLoc(), mergeBlock, operands);
      rewriter.eraseOp(yield);
    }
  }
}

//===----------------------------------------------------------------------===//
// IfOp Lowering
//===----------------------------------------------------------------------===//

struct IfOpLowering : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!containsSuspend(ifOp))
      return failure();
    Location loc = ifOp.getLoc();
    decrementBreakingOps(ifOp, rewriter);

    Block *condBlock = rewriter.getInsertionBlock();
    Block *remainingBlock =
        rewriter.splitBlock(condBlock, Block::iterator(ifOp));

    // Expression form: create merge block with a block arg for the result.
    Block *mergeBlock = remainingBlock;
    if (ifOp.isExpression()) {
      mergeBlock =
          rewriter.createBlock(remainingBlock, ifOp.getResult().getType(), loc);
      rewriter.setInsertionPointToEnd(mergeBlock);
      cf::BranchOp::create(rewriter, loc, remainingBlock);
    }

    // Then region.
    Region &thenRegion = ifOp.getThenRegion();
    Block *thenEntry = &thenRegion.front();
    replaceYieldsWithBranch(thenRegion, mergeBlock, rewriter);
    rewriter.inlineRegionBefore(thenRegion, mergeBlock);

    // Else region (or branch directly to merge).
    Block *elseEntry = mergeBlock;
    Region &elseRegion = ifOp.getElseRegion();
    if (!elseRegion.empty()) {
      elseEntry = &elseRegion.front();
      replaceYieldsWithBranch(elseRegion, mergeBlock, rewriter);
      rewriter.inlineRegionBefore(elseRegion, mergeBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    cf::CondBranchOp::create(rewriter, loc, ifOp.getCondition(), thenEntry,
                             ValueRange(), elseEntry, ValueRange());

    if (ifOp.isExpression())
      rewriter.replaceOp(ifOp, mergeBlock->getArguments());
    else
      rewriter.eraseOp(ifOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SwitchOp Lowering
//===----------------------------------------------------------------------===//

struct SwitchOpLowering : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    if (!containsSuspend(switchOp))
      return failure();
    Location loc = switchOp.getLoc();
    decrementBreakingOps(switchOp, rewriter);

    Block *condBlock = rewriter.getInsertionBlock();
    Block *remainingBlock =
        rewriter.splitBlock(condBlock, Block::iterator(switchOp));

    // Create merge block (with optional result block arg).
    Block *mergeBlock;
    if (switchOp.getResult()) {
      mergeBlock = rewriter.createBlock(remainingBlock,
                                        switchOp.getResult().getType(), loc);
      rewriter.setInsertionPointToEnd(mergeBlock);
      cf::BranchOp::create(rewriter, loc, remainingBlock);
    } else {
      mergeBlock = remainingBlock;
    }

    // Build case values and destinations for cf.switch.
    // Each case_values entry is an ArrayAttr of constants that map to one case.
    SmallVector<APInt> flatCaseValues;
    SmallVector<Block *> caseDestinations;
    SmallVector<ValueRange> caseOperands;

    auto caseRegions = switchOp.getCaseRegions();
    ArrayAttr caseValuesAttr = switchOp.getCaseValues();
    for (auto [i, caseRegion] : llvm::enumerate(caseRegions)) {
      Block *caseEntry = &caseRegion.front();
      replaceYieldsWithBranch(caseRegion, mergeBlock, rewriter);
      rewriter.inlineRegionBefore(caseRegion, mergeBlock);

      // Each case_values[i] is an ArrayAttr of values for this case.
      auto valuesForCase = mlir::cast<ArrayAttr>(caseValuesAttr[i]);
      for (Attribute val : valuesForCase) {
        auto intVal = mlir::cast<IntegerAttr>(val);
        flatCaseValues.push_back(intVal.getValue());
        caseDestinations.push_back(caseEntry);
        caseOperands.push_back(ValueRange());
      }
    }

    // Default (else) region.
    Block *defaultDest = mergeBlock;
    Region &elseRegion = switchOp.getElseRegion();
    if (!elseRegion.empty()) {
      defaultDest = &elseRegion.front();
      replaceYieldsWithBranch(elseRegion, mergeBlock, rewriter);
      rewriter.inlineRegionBefore(elseRegion, mergeBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    cf::SwitchOp::create(rewriter, loc, switchOp.getInput(), defaultDest,
                         ValueRange(), flatCaseValues, caseDestinations,
                         caseOperands);

    if (switchOp.getResult())
      rewriter.replaceOp(switchOp, mergeBlock->getArguments());
    else
      rewriter.eraseOp(switchOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ShortCircuitOp Lowering
//===----------------------------------------------------------------------===//

struct ShortCircuitOpLowering : public OpRewritePattern<ShortCircuitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShortCircuitOp scOp,
                                PatternRewriter &rewriter) const override {
    if (!containsSuspend(scOp))
      return failure();
    Location loc = scOp.getLoc();
    decrementBreakingOps(scOp, rewriter);

    Block *condBlock = rewriter.getInsertionBlock();
    Block *remainingBlock =
        rewriter.splitBlock(condBlock, Block::iterator(scOp));

    auto i1Ty = rewriter.getI1Type();

    // Merge block receives the boolean result.
    Block *mergeBlock = rewriter.createBlock(remainingBlock, i1Ty, loc);
    rewriter.setInsertionPointToEnd(mergeBlock);
    cf::BranchOp::create(rewriter, loc, remainingBlock);

    // Short-circuit block: LHS determined the result.
    Block *scBlock = rewriter.createBlock(mergeBlock);
    rewriter.setInsertionPointToEnd(scBlock);
    bool isOr = scOp.getKind() == csl::ShortCircuitKind::Or;
    Value scVal = ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(isOr));
    cf::BranchOp::create(rewriter, loc, mergeBlock, scVal);

    // RHS region: evaluate the right-hand side.
    Region &rhsRegion = scOp.getRhsRegion();
    Block *rhsEntry = &rhsRegion.front();
    replaceYieldsWithBranch(rhsRegion, mergeBlock, rewriter);
    rewriter.inlineRegionBefore(rhsRegion, mergeBlock);

    // Condition branch: for OR, short-circuit if LHS is true.
    // For AND, short-circuit if LHS is false.
    rewriter.setInsertionPointToEnd(condBlock);
    if (isOr) {
      cf::CondBranchOp::create(rewriter, loc, scOp.getLhs(), scBlock,
                               ValueRange(), rhsEntry, ValueRange());
    } else {
      cf::CondBranchOp::create(rewriter, loc, scOp.getLhs(), rhsEntry,
                               ValueRange(), scBlock, ValueRange());
    }

    rewriter.replaceOp(scOp, mergeBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LabeledBlockOp Lowering
//===----------------------------------------------------------------------===//

struct LabeledBlockOpLowering : public OpRewritePattern<LabeledBlockOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LabeledBlockOp lblOp,
                                PatternRewriter &rewriter) const override {
    if (!containsSuspend(lblOp))
      return failure();
    Location loc = lblOp.getLoc();

    // Collect all breaks targeting this labeled block (at any depth).
    SmallVector<BreakOp> targetingBreaks;
    visitNestedRegionExits(lblOp, [&](Operation *op, int level) {
      if (level == 1)
        if (auto breakOp = mlir::dyn_cast<BreakOp>(op))
          targetingBreaks.push_back(breakOp);
    });

    decrementBreakingOps(lblOp, rewriter);

    Block *condBlock = rewriter.getInsertionBlock();
    Block *remainingBlock =
        rewriter.splitBlock(condBlock, Block::iterator(lblOp));

    // Create exit block (with optional result block arg).
    Block *exitBlock;
    if (lblOp.getResult()) {
      exitBlock = rewriter.createBlock(remainingBlock,
                                       lblOp.getResult().getType(), loc);
      rewriter.setInsertionPointToEnd(exitBlock);
      cf::BranchOp::create(rewriter, loc, remainingBlock);
    } else {
      exitBlock = remainingBlock;
    }

    // Convert breaks targeting this block to cf.br ^exit.
    for (BreakOp breakOp : targetingBreaks) {
      rewriter.setInsertionPoint(breakOp);
      SmallVector<Value> operands;
      if (breakOp.getValue())
        operands.push_back(breakOp.getValue());
      cf::BranchOp::create(rewriter, breakOp.getLoc(), exitBlock, operands);
      rewriter.eraseOp(breakOp);
    }

    // Replace yields with cf.br ^exit.
    Region &body = lblOp.getBody();
    replaceYieldsWithBranch(body, exitBlock, rewriter);

    // Branch from condBlock to body entry.
    Block *bodyEntry = &body.front();
    rewriter.inlineRegionBefore(body, exitBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    cf::BranchOp::create(rewriter, loc, bodyEntry);

    if (lblOp.getResult())
      rewriter.replaceOp(lblOp, exitBlock->getArguments());
    else
      rewriter.eraseOp(lblOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ForOp Lowering
//===----------------------------------------------------------------------===//

struct ForOpLowering : public OpRewritePattern<ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!containsSuspend(forOp))
      return failure();
    Location loc = forOp.getLoc();

    // Collect all breaks/continues targeting this loop (at any depth).
    SmallVector<BreakOp> targetingBreaks;
    SmallVector<ContinueOp> targetingContinues;
    visitNestedRegionExits(forOp, [&](Operation *op, int level) {
      if (level == 1) {
        if (auto breakOp = mlir::dyn_cast<BreakOp>(op))
          targetingBreaks.push_back(breakOp);
        else if (auto contOp = mlir::dyn_cast<ContinueOp>(op))
          targetingContinues.push_back(contOp);
      }
    });

    decrementBreakingOps(forOp, rewriter);

    MLIRContext *ctx = rewriter.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    // Determine iteration kind.
    Type iterType = forOp.getIterable().getType();
    Value iterableVal = forOp.getIterable();
    bool iterIsLvalue = mlir::isa<csl::LValueType>(iterType);
    Type rawIterType =
        iterIsLvalue ? mlir::cast<csl::LValueType>(iterType).getValueType()
                     : iterType;

    bool isRange = mlir::isa<csl::RangeType>(rawIterType);
    bool isArray = mlir::isa<csl::ArrayType>(rawIterType);
    if (!isRange && !isArray)
      return forOp.emitError("unsupported iterable type for ForOp lowering");

    // Split block at ForOp: initBlock | exitBlock.
    Block *initBlock = rewriter.getInsertionBlock();
    Block *exitBlock = rewriter.splitBlock(initBlock, Block::iterator(forOp));

    rewriter.setInsertionPointToEnd(initBlock);

    // Load iterable if it's an lvalue.
    Value iterRval = iterableVal;
    if (iterIsLvalue)
      iterRval = LoadOp::create(rewriter, loc, iterableVal);

    // Create iteration state variables.
    Value ivVar;   // mutable lvalue for current element / index
    Value stopVal; // loop bound
    Value stepVal; // step (for ranges)
    Value idxVar;  // mutable lvalue for index counter (optional)
    Type elemType;
    bool hasIndex = forOp.hasIndex();

    if (isRange) {
      auto rangeType = mlir::cast<csl::RangeType>(rawIterType);
      elemType = rangeType.getElementType();

      // Extract range parameters.
      Value startVal =
          BuiltinCallOp::create(rewriter, loc, elemType,
                                csl::BuiltinKind::RangeStart, iterRval)
              .getResult();
      stopVal = BuiltinCallOp::create(rewriter, loc, elemType,
                                      csl::BuiltinKind::RangeStop, iterRval)
                    .getResult();
      stepVal = BuiltinCallOp::create(rewriter, loc, elemType,
                                      csl::BuiltinKind::RangeStep, iterRval)
                    .getResult();

      auto ivLvalTy = csl::LValueType::get(elemType, /*isConst=*/false);
      ivVar = VariableOp::create(rewriter, loc, ivLvalTy, forOp.getElemName(),
                                 startVal,
                                 /*is_const=*/false);
    } else {
      // Array iteration.
      auto arrayType = mlir::cast<csl::ArrayType>(rawIterType);
      elemType = arrayType.getElementType();
      int64_t numElements = arrayType.getShape()[0];

      stopVal = ConstantOp::create(rewriter, loc,
                                   IntegerAttr::get(ui16Ty, numElements));

      auto idxLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
      Value zero =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 0));
      ivVar = VariableOp::create(rewriter, loc, idxLvalTy, forOp.getElemName(),
                                 zero,
                                 /*is_const=*/false);
    }

    // Create index variable if needed.
    if (hasIndex) {
      auto idxLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
      Value zero =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 0));
      idxVar = VariableOp::create(rewriter, loc, idxLvalTy,
                                  *forOp.getIndexName(), zero,
                                  /*is_const=*/false);
    }

    // Create condition block.
    Block *condBlock = rewriter.createBlock(exitBlock);
    rewriter.setInsertionPointToEnd(condBlock);

    // Compare iteration variable against stop value (works for both range and
    // array: ivVar holds elem index for range, array index for array).
    Value cmpLhs = LoadOp::create(rewriter, loc, ivVar);
    Value cmpRhs = stopVal;
    Value cmp = BinaryOp::create(rewriter, loc, rewriter.getI1Type(),
                                 csl::BinaryOpKind::Lt, cmpLhs, cmpRhs);

    // Create step block (between body and condBlock in the loop).
    Block *stepBlock = rewriter.createBlock(exitBlock);
    rewriter.setInsertionPointToEnd(stepBlock);
    if (isRange) {
      Value curIv = LoadOp::create(rewriter, loc, ivVar);
      Value nextIv = BinaryOp::create(rewriter, loc, elemType,
                                      csl::BinaryOpKind::Add, curIv, stepVal);
      AssignOp::create(rewriter, loc, ivVar, nextIv);
    } else {
      Value one =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 1));
      Value curIdx = LoadOp::create(rewriter, loc, ivVar);
      Value nextIdx = BinaryOp::create(rewriter, loc, ui16Ty,
                                       csl::BinaryOpKind::Add, curIdx, one);
      AssignOp::create(rewriter, loc, ivVar, nextIdx);
    }
    if (hasIndex) {
      Value one =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 1));
      Value curIdxVal = LoadOp::create(rewriter, loc, idxVar);
      Value nextIdxVal = BinaryOp::create(
          rewriter, loc, ui16Ty, csl::BinaryOpKind::Add, curIdxVal, one);
      AssignOp::create(rewriter, loc, idxVar, nextIdxVal);
    }
    cf::BranchOp::create(rewriter, loc, condBlock);

    // Replace uses of body block args with loads from variables.
    Block &bodyBlock = forOp.getBody().front();
    rewriter.setInsertionPointToStart(&bodyBlock);
    if (isRange) {
      // Elem arg → load from ivVar.
      Value elemVal = LoadOp::create(rewriter, loc, ivVar);
      bodyBlock.getArgument(0).replaceAllUsesWith(elemVal);
    } else {
      // Array: elem arg → load array element at current index.
      Value idxVal = LoadOp::create(rewriter, loc, ivVar);
      if (iterIsLvalue) {
        // iterable is lvalue<array> → array_index_access gives lvalue<elem>
        auto elemLvalTy = csl::LValueType::get(
            elemType, mlir::cast<csl::LValueType>(iterType).getIsConst());
        Value elemLval = ArrayIndexAccessOp::create(rewriter, loc, elemLvalTy,
                                                    iterableVal, idxVal);
        Value elemVal = LoadOp::create(rewriter, loc, elemLval);
        bodyBlock.getArgument(0).replaceAllUsesWith(elemVal);
      } else {
        // iterable is rvalue array → array_index_access gives rvalue elem
        Value elemVal = ArrayIndexAccessOp::create(rewriter, loc, elemType,
                                                   iterRval, idxVal);
        bodyBlock.getArgument(0).replaceAllUsesWith(elemVal);
      }
    }
    if (hasIndex) {
      Value idxVal = LoadOp::create(rewriter, loc, isRange ? idxVar : ivVar);
      // For arrays, the iteration variable IS the index. But the user might
      // also have a separate index name. If isRange, index is separate.
      bodyBlock.getArgument(1).replaceAllUsesWith(idxVal);
    }

    // Remove body block args (after replacing all uses).
    while (bodyBlock.getNumArguments() > 0)
      bodyBlock.eraseArgument(bodyBlock.getNumArguments() - 1);

    // Convert breaks targeting this loop → cf.br ^exit.
    for (BreakOp breakOp : targetingBreaks) {
      rewriter.setInsertionPoint(breakOp);
      cf::BranchOp::create(rewriter, breakOp.getLoc(), exitBlock);
      rewriter.eraseOp(breakOp);
    }

    // Convert continues targeting this loop → cf.br ^step.
    for (ContinueOp contOp : targetingContinues) {
      rewriter.setInsertionPoint(contOp);
      cf::BranchOp::create(rewriter, contOp.getLoc(), stepBlock);
      rewriter.eraseOp(contOp);
    }

    // Convert yields in body → cf.br ^step.
    replaceYieldsWithBranch(forOp.getBody(), stepBlock, rewriter);

    // Inline body region before step block.
    Block *bodyEntry = &forOp.getBody().front();
    rewriter.inlineRegionBefore(forOp.getBody(), stepBlock);

    // Condition: branch to body or exit.
    rewriter.setInsertionPointToEnd(condBlock);
    cf::CondBranchOp::create(rewriter, loc, cmp, bodyEntry, ValueRange(),
                             exitBlock, ValueRange());

    // Init: branch to condition block.
    rewriter.setInsertionPointToEnd(initBlock);
    cf::BranchOp::create(rewriter, loc, condBlock);

    rewriter.eraseOp(forOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// WhileOp Lowering
//===----------------------------------------------------------------------===//

struct WhileOpLowering : public OpRewritePattern<WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    if (!containsSuspend(whileOp))
      return failure();
    Location loc = whileOp.getLoc();

    // Collect ALL breaks/continues targeting this loop (at any nesting
    // depth). Level 1 = directly targets this while. These will become
    // cf.br after flattening. Returns are left as-is (decremented).
    SmallVector<BreakOp> targetingBreaks;
    SmallVector<ContinueOp> targetingContinues;
    visitNestedRegionExits(whileOp, [&](Operation *op, int level) {
      if (level == 1) {
        if (auto breakOp = mlir::dyn_cast<BreakOp>(op))
          targetingBreaks.push_back(breakOp);
        else if (auto contOp = mlir::dyn_cast<ContinueOp>(op))
          targetingContinues.push_back(contOp);
      }
    });

    decrementBreakingOps(whileOp, rewriter);

    Block *initBlock = rewriter.getInsertionBlock();
    Block *exitBlock = rewriter.splitBlock(initBlock, Block::iterator(whileOp));

    bool hasCont = !whileOp.getContinuation().empty();

    // Create continuation block (if present).
    Block *contEntry = nullptr;
    if (hasCont) {
      Region &contRegion = whileOp.getContinuation();
      contEntry = &contRegion.front();
      // Continuation yields → cf.br back to condEntry (set up after inlining).
    }

    // Inline condition region.
    Region &condRegion = whileOp.getCondRegion();
    Block *condEntry = &condRegion.front();

    // Replace cond_yield with cf.cond_br.
    Block &condBlock = condRegion.back();
    auto *condTerm = condBlock.getTerminator();
    auto condYield = mlir::cast<CondYieldOp>(condTerm);
    Value condVal = condYield.getCondition();

    // Inline body region.
    Region &bodyRegion = whileOp.getBody();
    Block *bodyEntry = &bodyRegion.front();

    // Determine where continue goes.
    Block *continueTarget = hasCont ? contEntry : condEntry;

    // Convert breaks targeting this loop → cf.br ^exit.
    for (BreakOp breakOp : targetingBreaks) {
      rewriter.setInsertionPoint(breakOp);
      cf::BranchOp::create(rewriter, breakOp.getLoc(), exitBlock);
      rewriter.eraseOp(breakOp);
    }

    // Convert continues targeting this loop → cf.br ^cont (or ^cond).
    for (ContinueOp contOp : targetingContinues) {
      rewriter.setInsertionPoint(contOp);
      cf::BranchOp::create(rewriter, contOp.getLoc(), continueTarget);
      rewriter.eraseOp(contOp);
    }

    // Replace body yields → cf.br to continue target.
    replaceYieldsWithBranch(bodyRegion, continueTarget, rewriter);

    // Replace cond_yield with cf.cond_br.
    rewriter.setInsertionPoint(condYield);
    cf::CondBranchOp::create(rewriter, loc, condVal, bodyEntry, ValueRange(),
                             exitBlock, ValueRange());
    rewriter.eraseOp(condYield);

    // Inline regions before exit block.
    if (hasCont) {
      Region &contRegion = whileOp.getContinuation();
      // Continuation yields → cf.br ^cond.
      replaceYieldsWithBranch(contRegion, condEntry, rewriter);
      rewriter.inlineRegionBefore(contRegion, exitBlock);
    }
    rewriter.inlineRegionBefore(bodyRegion, exitBlock);
    rewriter.inlineRegionBefore(condRegion, exitBlock);

    // Branch from init to condition.
    rewriter.setInsertionPointToEnd(initBlock);
    cf::BranchOp::create(rewriter, loc, condEntry);

    rewriter.eraseOp(whileOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSuspendingSCFToCFPass
    : public typed_csl::impl::LowerSuspendingSCFToCFPassBase<
          LowerSuspendingSCFToCFPass> {
  using LowerSuspendingSCFToCFPassBase::LowerSuspendingSCFToCFPassBase;

  /// Replace non-terminator SuspendOps with SuspendBrOp terminators,
  /// splitting the block at each suspend point.
  void lowerSuspendOps(FuncOp funcOp, IRRewriter &rewriter) {
    SmallVector<SuspendOp> suspends;
    funcOp.walk([&](SuspendOp s) { suspends.push_back(s); });

    for (SuspendOp op : suspends) {
      Block *block = op->getBlock();
      Block *resumeBlock =
          rewriter.splitBlock(block, Block::iterator(op->getNextNode()));
      rewriter.setInsertionPoint(op);
      SuspendBrOp::create(rewriter, op.getLoc(), resumeBlock);
      rewriter.eraseOp(op);
    }
  }

  /// Split blocks at coro.async_call-tagged CallOps. The call and everything
  /// after it goes to the successor block; the predecessor is terminated with
  /// await_br (a non-mergeable boundary). This ensures frame-init ops and the
  /// async call end up in different trampoline states.
  void splitAtAsyncCalls(FuncOp funcOp, IRRewriter &rewriter) {
    SmallVector<CallOp> asyncCalls;
    funcOp.walk([&](CallOp callOp) {
      if (callOp->hasAttr("coro.async_call"))
        asyncCalls.push_back(callOp);
    });

    for (CallOp callOp : asyncCalls) {
      Block *block = callOp->getBlock();
      Block *callBlock = rewriter.splitBlock(block, Block::iterator(callOp));
      rewriter.setInsertionPointToEnd(block);
      AwaitBrOp::create(rewriter, callOp.getLoc(), callBlock);
      callOp->removeAttr("coro.async_call");
    }
  }

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();

    unit.walk([&](FuncOp funcOp) {
      if (!funcOp->hasAttr("coro.wip"))
        return;

      // Phase 1: Selectively lower structured CF ops containing suspends.
      RewritePatternSet patterns(funcOp->getContext());
      patterns.add<IfOpLowering, SwitchOpLowering, ShortCircuitOpLowering,
                   LabeledBlockOpLowering, ForOpLowering, WhileOpLowering>(
          funcOp->getContext());

      GreedyRewriteConfig config;
      config.setStrictness(GreedyRewriteStrictness::ExistingOps);
      // Disable region simplification — its aggressive form merges blocks
      // with identical content into a single block parameterized by block
      // arguments. The downstream trampoline emitter does not handle
      // non-entry block args (it has no per-block reload logic), so we
      // must keep blocks separate. We trade some IR redundancy for a
      // shape the rest of the trampoline pipeline can consume.
      config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

      if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }

      IRRewriter rewriter(funcOp->getContext());

      // Phase 2: Replace SuspendOps with SuspendBrOp terminators.
      lowerSuspendOps(funcOp, rewriter);

      // Phase 3: Split blocks at async call sites (coro.async_call).
      // Creates await_br boundaries so the canonicalizer cannot merge
      // frame-init blocks with call blocks.
      splitAtAsyncCalls(funcOp, rewriter);
    });
  }
};

} // namespace
