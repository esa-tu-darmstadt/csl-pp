//===- InlineSmallNonResumeBlocksPass.cpp - Inline non-resume CFG blocks --===//
//
// Duplicates small non-resume blocks into their predecessors to reduce
// trampoline state count. Runs directly after CoroFrameRewritePass (which
// eliminates all cross-block SSA values) and before TrampolineEmitPass
// (which maps blocks 1:1 to switch states).
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "inline-small-non-resume-blocks"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_INLINESMALLNONRESUMEBLOCKSPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

/// Maximum number of non-terminator ops (counting nested ops in regions) for
/// a block to be eligible for inlining.
static constexpr unsigned kInlineOpThreshold = 50;

/// Count non-terminator ops in a block, including ops nested in regions.
static unsigned countBlockOps(Block *block) {
  unsigned count = 0;
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    op.walk([&](Operation *) { ++count; });
  }
  return count;
}

/// Collect all blocks that are resume targets: successors of SuspendBrOp or
/// AwaitBrOp. These blocks must remain as separate basic blocks because they
/// are re-entry points after task reactivation.
static DenseSet<Block *> collectResumeTargets(FuncOp funcOp) {
  DenseSet<Block *> targets;
  for (Block &block : funcOp.getBody()) {
    Operation *term = block.getTerminator();
    if (auto suspendBr = dyn_cast<SuspendBrOp>(term))
      targets.insert(suspendBr.getDest());
    if (auto awaitBr = dyn_cast<AwaitBrOp>(term))
      targets.insert(awaitBr.getDest());
  }
  return targets;
}

/// Clone `target`'s non-terminator ops and terminator into the current
/// insertion point of `rewriter`, using `mapping` for value remapping.
static void cloneBlockOps(Block *target, IRRewriter &rewriter,
                          IRMapping &mapping) {
  for (Operation &op : *target) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    rewriter.clone(op, mapping);
  }
  rewriter.clone(*target->getTerminator(), mapping);
}

/// Map `target`'s block arguments from `branchOperands` into `mapping`.
static void mapBlockArgs(Block *target, OperandRange branchOperands,
                         IRMapping &mapping) {
  for (auto [arg, operand] :
       llvm::zip(target->getArguments(), branchOperands)) {
    mapping.map(arg, operand);
  }
}

/// Inline the contents of `target` into `pred`, replacing `pred`'s branch
/// terminator with clones of `target`'s ops and terminator.
static void inlineBlockIntoPredecessor(Block *target, Block *pred,
                                       OperandRange branchOperands,
                                       IRRewriter &rewriter) {
  Operation *predTerm = pred->getTerminator();
  rewriter.setInsertionPoint(predTerm);

  IRMapping mapping;
  mapBlockArgs(target, branchOperands, mapping);
  cloneBlockOps(target, rewriter, mapping);

  rewriter.eraseOp(predTerm);
}

/// For a cf.cond_br where one or both targets should be inlined, create
/// intermediate blocks with the cloned ops and retarget the cond_br.
static void inlineCondBrTargets(Block *pred, cf::CondBranchOp condBr,
                                bool inlineTrue, Block *trueTarget,
                                bool inlineFalse, Block *falseTarget,
                                IRRewriter &rewriter) {
  auto inlineBranch = [&](Block *target, OperandRange branchArgs) -> Block * {
    Block *inlinedBlock =
        rewriter.createBlock(pred->getParent(), std::next(pred->getIterator()));
    rewriter.setInsertionPointToStart(inlinedBlock);

    IRMapping mapping;
    mapBlockArgs(target, branchArgs, mapping);
    cloneBlockOps(target, rewriter, mapping);
    return inlinedBlock;
  };

  Block *newTrueTarget =
      inlineTrue ? inlineBranch(trueTarget, condBr.getTrueOperands())
                 : condBr.getTrueDest();
  ValueRange newTrueArgs = inlineTrue ? ValueRange() : condBr.getTrueOperands();

  Block *newFalseTarget =
      inlineFalse ? inlineBranch(falseTarget, condBr.getFalseOperands())
                  : condBr.getFalseDest();
  ValueRange newFalseArgs =
      inlineFalse ? ValueRange() : condBr.getFalseOperands();

  rewriter.setInsertionPoint(condBr);
  cf::CondBranchOp::create(rewriter, condBr.getLoc(), condBr.getCondition(),
                           newTrueTarget, newTrueArgs, newFalseTarget,
                           newFalseArgs);
  rewriter.eraseOp(condBr);
}

/// Verify that no op result in the block has uses outside the block.
/// After CoroFrameRewritePass, all cross-block SSA values should have been
/// spilled to the frame. If this invariant is violated, something is wrong
/// in the pipeline ordering.
static LogicalResult verifyNoEscapingValues(Block *block, FuncOp funcOp) {
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getBlock() != block) {
          return funcOp.emitError(
              "InlineSmallNonResumeBlocksPass: dead block has op result with "
              "external use — pass must run after CoroFrameRewritePass");
        }
      }
    }
  }
  return success();
}

/// Process a single coroutine function: inline small non-resume blocks.
static LogicalResult processFunction(FuncOp funcOp, IRRewriter &rewriter) {
  DenseSet<Block *> resumeTargets = collectResumeTargets(funcOp);
  // The entry block is implicitly a resume target (initial entry point).
  resumeTargets.insert(&funcOp.getBody().front());

  // Find inlinable blocks: non-resume, small.
  DenseSet<Block *> inlinable;
  for (Block &block : funcOp.getBody()) {
    if (resumeTargets.contains(&block))
      continue;
    if (countBlockOps(&block) < kInlineOpThreshold)
      inlinable.insert(&block);
  }

  if (inlinable.empty())
    return success();

  LLVM_DEBUG({
    llvm::dbgs() << "  Inlinable blocks:";
    for (Block *b : inlinable)
      llvm::dbgs() << " ^bb"
                   << std::distance(funcOp.getBody().begin(), b->getIterator());
    llvm::dbgs() << "\n";
  });

  // Process predecessors. Iterate until convergence because inlining may
  // create new intermediate blocks that target other inlinable blocks.
  bool progress = true;
  while (progress) {
    progress = false;

    // Snapshot the block list — CFG modifications may add/remove blocks.
    SmallVector<Block *> worklist;
    for (Block &block : funcOp.getBody())
      worklist.push_back(&block);

    for (Block *pred : worklist) {
      if (pred->getParent() == nullptr)
        continue;

      Operation *term = pred->getTerminator();

      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        Block *target = br.getDest();
        if (!inlinable.contains(target))
          continue;

        LLVM_DEBUG(llvm::dbgs() << "  Inlining ^bb"
                                << std::distance(funcOp.getBody().begin(),
                                                 target->getIterator())
                                << " into predecessor via cf.br\n");

        inlineBlockIntoPredecessor(target, pred, br.getDestOperands(),
                                   rewriter);
        progress = true;
        continue;
      }

      if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        bool inlineTrue = inlinable.contains(condBr.getTrueDest());
        bool inlineFalse = inlinable.contains(condBr.getFalseDest());
        if (!inlineTrue && !inlineFalse)
          continue;

        LLVM_DEBUG({
          if (inlineTrue)
            llvm::dbgs() << "  Inlining true-target ^bb"
                         << std::distance(funcOp.getBody().begin(),
                                          condBr.getTrueDest()->getIterator())
                         << " into cond_br predecessor\n";
          if (inlineFalse)
            llvm::dbgs() << "  Inlining false-target ^bb"
                         << std::distance(funcOp.getBody().begin(),
                                          condBr.getFalseDest()->getIterator())
                         << " into cond_br predecessor\n";
        });

        inlineCondBrTargets(pred, condBr, inlineTrue, condBr.getTrueDest(),
                            inlineFalse, condBr.getFalseDest(), rewriter);
        progress = true;
        continue;
      }
    }
  }

  // Erase inlinable blocks that are now dead (no remaining predecessors).
  // Snapshot to a vector first — cannot iterate DenseSet while erasing.
  SmallVector<Block *> candidates(inlinable.begin(), inlinable.end());
  for (Block *block : candidates) {
    if (block->getParent() == nullptr)
      continue;
    if (!block->hasNoPredecessors())
      continue;

    // After CoroFrameRewritePass, no SSA values should cross blocks.
    // Verify this before erasing; if violated, the pipeline is misconfigured.
    if (failed(verifyNoEscapingValues(block, funcOp)))
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "  Erasing dead block ^bb"
               << std::distance(funcOp.getBody().begin(), block->getIterator())
               << "\n");
    rewriter.eraseBlock(block);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct InlineSmallNonResumeBlocksPass
    : public typed_csl::impl::InlineSmallNonResumeBlocksPassBase<
          InlineSmallNonResumeBlocksPass> {
  using InlineSmallNonResumeBlocksPassBase::InlineSmallNonResumeBlocksPassBase;

  void runOnOperation() override {
    auto unit = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<FuncOp> coroFuncs;
    unit.walk([&](FuncOp funcOp) {
      if (funcOp->hasAttr("coro.wip"))
        coroFuncs.push_back(funcOp);
    });

    for (FuncOp funcOp : coroFuncs) {
      LLVM_DEBUG(llvm::dbgs() << "=== InlineSmallNonResumeBlocks: "
                              << funcOp.getSymName() << " ===\n");
      if (failed(processFunction(funcOp, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
