//===- TrampolineEmitPass.cpp - Emit structured trampoline state machine --===//
//
// Transforms CFG-form coroutines into structured while+switch trampolines.
// Each basic block becomes a switch case. CF terminators become state
// transitions.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/CoroFrameBuilder.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "trampoline-emit"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_TRAMPOLINEEMITPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Emit `asm volatile ("nop.term")` — instantly terminates the current task.
/// Used at suspend points: the hardware scheduler takes over immediately.
static void emitNopTerm(OpBuilder &b, Location loc) {
  auto ctx = b.getContext();
  auto emptyStrArr = b.getStrArrayAttr({});
  InlineAsmOp::create(b, loc, /*outputs=*/ValueRange{}, /*inputs=*/ValueRange{},
                      b.getStringAttr("nop.term"),
                      /*output_names=*/emptyStrArr,
                      /*output_constraints=*/emptyStrArr,
                      /*input_names=*/emptyStrArr,
                      /*input_constraints=*/emptyStrArr,
                      /*clobbers=*/emptyStrArr,
                      /*is_volatile=*/UnitAttr::get(ctx));
}

/// Store a state value to the frame at offset 0 (the state field).
static void emitStoreState(OpBuilder &b, Location loc, Value framePtr,
                           uint16_t stateId, unsigned slotAlign) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  auto mutableElemLvalTy = csl::LValueType::get(elemTy, /*isConst=*/false);

  uint32_t elemOffset = byteToElementOffset(kCoroStateOffset, slotAlign);
  Value offsetVal =
      ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, elemOffset));
  Value elemLval =
      PtrIndexAccessOp::create(b, loc, mutableElemLvalTy, framePtr, offsetVal);
  auto elemPtrTy =
      csl::PointerType::get(elemTy, /*isConst=*/false, /*isArray=*/false);
  Value elemPtr = AddressOfOp::create(b, loc, elemPtrTy, elemLval);
  auto statePtrTy =
      csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/false);
  Value typedPtr = CastOp::create(b, loc, statePtrTy, elemPtr).getResult();
  auto mutableLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
  Value lval = DereferenceOp::create(b, loc, mutableLvalTy, typedPtr);
  Value stateVal =
      ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, stateId));
  AssignOp::create(b, loc, lval, stateVal);
}

/// Load the state value from the frame at offset 0.
static Value emitLoadState(OpBuilder &b, Location loc, Value framePtr,
                           unsigned slotAlign) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  auto mutableElemLvalTy = csl::LValueType::get(elemTy, /*isConst=*/false);

  uint32_t elemOffset = byteToElementOffset(kCoroStateOffset, slotAlign);
  Value offsetVal =
      ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, elemOffset));
  Value elemLval =
      PtrIndexAccessOp::create(b, loc, mutableElemLvalTy, framePtr, offsetVal);
  auto elemPtrTy =
      csl::PointerType::get(elemTy, /*isConst=*/false, /*isArray=*/false);
  Value elemPtr = AddressOfOp::create(b, loc, elemPtrTy, elemLval);
  auto statePtrTy =
      csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/false);
  Value typedPtr = CastOp::create(b, loc, statePtrTy, elemPtr).getResult();
  auto mutableLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
  Value lval = DereferenceOp::create(b, loc, mutableLvalTy, typedPtr);
  return LoadOp::create(b, loc, lval);
}

//===----------------------------------------------------------------------===//
// Pre-step: Lower FrameAddressOp
//===----------------------------------------------------------------------===//
//
// FrameAddressOp needs the task frame global (created by CoroFrameRewritePass),
// so it cannot be lowered in CoroPrepareCFPass. We lower it here,
// right before emitting the trampoline.

/// Get the frame element pointer for a coroutine function.
/// For tasks: global reference + address_of + ptrcast from the frame global.
/// For async fns: load from the first block argument.
static Value getFramePtr(FuncOp funcOp, OpBuilder &b, Location loc,
                         unsigned slotAlign) {
  if (funcOp.getIsTask()) {
    auto globalName = (funcOp.getSymName() + "__coro_frame").str();
    auto unit = funcOp->getParentOfType<csl::CompilationUnitOp>();
    auto globalOp = unit.lookupSymbol<GlobalOp>(globalName);
    assert(globalOp && "task frame global must exist by TrampolineEmitPass");
    auto frameLvalTy =
        csl::LValueType::get(globalOp.getType(), /*isConst=*/false);
    Value lval = createGlobalRef(b, loc, frameLvalTy, globalOp.getSymName());
    auto ptrTy = csl::PointerType::get(globalOp.getType(), /*isConst=*/false,
                                       /*isArray=*/false);
    Value addr = AddressOfOp::create(b, loc, ptrTy, lval);
    auto framePtrTyped = mlir::cast<TypedValue<csl::PointerType>>(addr);
    return CoroutineFrameAccess::ensureFramePtr(b, loc, framePtrTyped,
                                                slotAlign);
  }
  // Async fn: frame is the first block argument.
  Block &entry = funcOp.getBody().front();
  Value frameLvalArg = entry.getArgument(0);
  Value loaded = LoadOp::create(b, loc, frameLvalArg);
  auto loadedTyped = mlir::cast<TypedValue<csl::PointerType>>(loaded);
  return CoroutineFrameAccess::ensureFramePtr(b, loc, loadedTyped, slotAlign);
}

/// Lower FrameAddressOp in a coro function.
static LogicalResult lowerFrameAccessOps(FuncOp funcOp, IRRewriter &rewriter) {
  auto unit = funcOp->getParentOfType<csl::CompilationUnitOp>();
  unsigned slotAlign = getCoroSlotAlignment(unit);
  MLIRContext *ctx = funcOp.getContext();

  // Lower FrameAddressOp → [*]u8 pointer to the frame.
  SmallVector<FrameAddressOp> faOps;
  funcOp.walk([&](FrameAddressOp op) { faOps.push_back(op); });
  for (FrameAddressOp faOp : faOps) {
    Location loc = faOp.getLoc();
    rewriter.setInsertionPoint(faOp);

    Value framePtr = getFramePtr(funcOp, rewriter, loc, slotAlign);
    Type bytePtrTy =
        csl::PointerType::get(IntegerType::get(ctx, 8, IntegerType::Unsigned),
                              /*isConst=*/false,
                              /*isArray=*/true);
    Value bytePtr =
        CastOp::create(rewriter, loc, bytePtrTy, framePtr).getResult();
    rewriter.replaceOp(faOp, bytePtr);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Trampoline Emitter
//===----------------------------------------------------------------------===//

static LogicalResult emitTrampoline(FuncOp funcOp, IRRewriter &rewriter) {
  Location loc = funcOp.getLoc();
  MLIRContext *ctx = funcOp.getContext();
  bool isTask = funcOp.getIsTask();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto unit = funcOp->getParentOfType<csl::CompilationUnitOp>();
  unsigned slotAlign = getCoroSlotAlignment(unit);

  // Collect original blocks and assign state IDs.
  SmallVector<Block *> blocks;
  DenseMap<Block *, uint16_t> blockToState;
  for (Block &block : funcOp.getBody()) {
    uint16_t stateId = blocks.size();
    blockToState[&block] = stateId;
    blocks.push_back(&block);
  }

  LLVM_DEBUG(llvm::dbgs() << "  " << blocks.size() << " blocks\n");

  // Create a new FuncOp right after the old one. We build the trampoline
  // entirely inside the new function, then erase the old one. This avoids
  // dangling-reference issues from cloning ops that reference values in
  // the original blocks.
  rewriter.setInsertionPointAfter(funcOp);
  auto newFuncOp =
      FuncOp::create(rewriter, loc, funcOp.getSymNameAttr(),
                     funcOp.getFunctionTypeAttr(), funcOp.getIsTaskAttr(),
                     funcOp.getIsInlineAttr(), funcOp.getIsNoinlineAttr(),
                     funcOp.getIsExportAttr(), funcOp.getIsExternAttr(),
                     funcOp.getIsAsyncAttr(), funcOp.getStackSizeAttr(),
                     funcOp.getLinknameAttr(), funcOp.getArgAttrsAttr(),
                     funcOp.getResAttrsAttr(), funcOp.getAsyncResultTypeAttr(),
                     funcOp.getMaxRecursionDepthAttr());

  // Build the entry block of the new function.
  Block *entryBlock = rewriter.createBlock(&newFuncOp.getBody());

  // Set up the IRMapping from old entry block arg → new entry block arg.
  IRMapping seedMapping;
  if (!isTask) {
    BlockArgument origArg = blocks[0]->getArgument(0);
    BlockArgument newArg =
        entryBlock->addArgument(origArg.getType(), origArg.getLoc());
    seedMapping.map(origArg, newArg);
  }

  // Derive the frame element pointer in the new entry block.
  Value framePtr;
  rewriter.setInsertionPointToStart(entryBlock);
  if (isTask) {
    auto globalName = (funcOp.getSymName() + "__coro_frame").str();
    auto globalOp = unit.lookupSymbol<GlobalOp>(globalName);
    if (!globalOp)
      return funcOp.emitError("task frame global '")
             << globalName << "' not found";
    auto frameLvalTy =
        csl::LValueType::get(globalOp.getType(), /*isConst=*/false);
    Value lval =
        createGlobalRef(rewriter, loc, frameLvalTy, globalOp.getSymName());
    auto ptrTy = csl::PointerType::get(globalOp.getType(), /*isConst=*/false,
                                       /*isArray=*/false);
    Value addr = AddressOfOp::create(rewriter, loc, ptrTy, lval);
    auto framePtrTyped = mlir::cast<TypedValue<csl::PointerType>>(addr);
    framePtr = CoroutineFrameAccess::ensureFramePtr(rewriter, loc,
                                                    framePtrTyped, slotAlign);
  } else {
    BlockArgument origArg = blocks[0]->getArgument(0);
    BlockArgument newArg = entryBlock->getArgument(0);
    Value loaded = LoadOp::create(rewriter, loc, newArg);
    auto loadedTyped = mlir::cast<TypedValue<csl::PointerType>>(loaded);
    framePtr = CoroutineFrameAccess::ensureFramePtr(rewriter, loc, loadedTyped,
                                                    slotAlign);

    // Map the original load-from-arg result so cloned ops that
    // reference the loaded frame pointer get the new value.
    for (Operation &op : *blocks[0]) {
      if (auto origLoad = mlir::dyn_cast<LoadOp>(&op)) {
        if (origLoad.getLvalue() == origArg) {
          seedMapping.map(origLoad.getResult(), loaded);
          break;
        }
      }
    }
  }

  // while(true) { ... }
  rewriter.setInsertionPointToEnd(entryBlock);
  auto whileOp = WhileOp::create(rewriter, loc, /*label=*/StringAttr());

  // After while: return void (coroutine is done).
  rewriter.setInsertionPointAfter(whileOp);
  ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);

  // While condition: always true.
  {
    Block *condBlock = rewriter.createBlock(&whileOp.getCondRegion());
    rewriter.setInsertionPointToEnd(condBlock);
    Value tv = ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    CondYieldOp::create(rewriter, loc, tv);
  }

  // While body: load state, switch.
  Block *bodyBlock = rewriter.createBlock(&whileOp.getBody());
  rewriter.setInsertionPointToEnd(bodyBlock);
  Value stateVal = emitLoadState(rewriter, loc, framePtr, slotAlign);

  // Switch on state.
  unsigned numCases = blocks.size() > 1 ? blocks.size() - 1 : 0;
  SmallVector<Attribute> caseArrays;
  for (unsigned s = 1; s < blocks.size(); ++s) {
    SmallVector<Attribute> vals;
    vals.push_back(IntegerAttr::get(ui16Ty, s));
    caseArrays.push_back(rewriter.getArrayAttr(vals));
  }
  auto switchOp = SwitchOp::create(rewriter, loc, /*result=*/Type(), stateVal,
                                   rewriter.getArrayAttr(caseArrays), numCases);

  // After switch: yield to continue the while loop.
  rewriter.setInsertionPointAfter(switchOp);
  YieldOp::create(rewriter, loc, /*value=*/Value());

  // Emit each original block's ops into a switch case.
  // State transitions: cf.br/cond_br → store state + yield from case.
  // Suspend: suspend_br → store state + nop.term (kills task) + dead yield.
  // Return: clone with numBreaking adjusted for trampoline nesting (+2).
  auto emitCase = [&](Block *origBlock, Region &caseRegion) {
    Block *caseBlock = rewriter.createBlock(&caseRegion);
    rewriter.setInsertionPointToEnd(caseBlock);

    IRMapping mapping(seedMapping);
    for (Operation &op : *origBlock) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      rewriter.clone(op, mapping);
    }

    Operation *term = origBlock->getTerminator();

    if (auto suspendBr = mlir::dyn_cast<SuspendBrOp>(term)) {
      uint16_t targetState = blockToState[suspendBr.getDest()];
      emitStoreState(rewriter, loc, framePtr, targetState, slotAlign);
      emitNopTerm(rewriter, loc);
      // Dead code: nop.term kills the task. YieldOp is structurally
      // required (each switch case region needs a terminator).
      YieldOp::create(rewriter, loc, /*value=*/Value());

    } else if (auto br = mlir::dyn_cast<cf::BranchOp>(term)) {
      uint16_t targetState = blockToState[br.getDest()];
      emitStoreState(rewriter, loc, framePtr, targetState, slotAlign);
      YieldOp::create(rewriter, loc, /*value=*/Value());

    } else if (auto awaitBr = mlir::dyn_cast<AwaitBrOp>(term)) {
      // Non-mergeable block boundary at async call sites.
      // Treated identically to cf.br: store next state and yield.
      uint16_t targetState = blockToState[awaitBr.getDest()];
      emitStoreState(rewriter, loc, framePtr, targetState, slotAlign);
      YieldOp::create(rewriter, loc, /*value=*/Value());

    } else if (auto condBr = mlir::dyn_cast<cf::CondBranchOp>(term)) {
      Value cond = mapping.lookupOrDefault(condBr.getCondition());
      uint16_t trueState = blockToState[condBr.getTrueDest()];
      uint16_t falseState = blockToState[condBr.getFalseDest()];

      auto ifOp = IfOp::create(rewriter, loc, TypeRange{}, cond);
      {
        Block *thenBlock = rewriter.createBlock(&ifOp.getThenRegion());
        rewriter.setInsertionPointToEnd(thenBlock);
        emitStoreState(rewriter, loc, framePtr, trueState, slotAlign);
        YieldOp::create(rewriter, loc, /*value=*/Value());
      }
      {
        Block *elseBlock = rewriter.createBlock(&ifOp.getElseRegion());
        rewriter.setInsertionPointToEnd(elseBlock);
        emitStoreState(rewriter, loc, framePtr, falseState, slotAlign);
        YieldOp::create(rewriter, loc, /*value=*/Value());
      }
      rewriter.setInsertionPointAfter(ifOp);
      YieldOp::create(rewriter, loc, /*value=*/Value());

    } else if (auto returnOp = mlir::dyn_cast<ReturnOp>(term)) {
      // Tasks are reentrant: reset state to 0 so the next activation
      // re-enters at the initial entry point (else => case).
      if (isTask)
        emitStoreState(rewriter, loc, framePtr, 0, slotAlign);
      auto cloned = mlir::cast<ReturnOp>(rewriter.clone(*term, mapping));
      cloned.setNumBreakingRegions(returnOp.getNumBreakingRegions() + 2);

    } else {
      rewriter.clone(*term, mapping);
    }
  };

  // State 0 (entry block) → else region, states 1+ → case regions.
  emitCase(blocks[0], switchOp.getElseRegion());
  for (unsigned i = 1; i < blocks.size(); ++i)
    emitCase(blocks[i], switchOp.getCaseRegions()[i - 1]);

  // Erase the old function and we're done.
  rewriter.eraseOp(funcOp);

  LLVM_DEBUG(llvm::dbgs() << "  Emitted trampoline with " << blocks.size()
                          << " states\n");
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TrampolineEmitPass
    : public typed_csl::impl::TrampolineEmitPassBase<TrampolineEmitPass> {
  using TrampolineEmitPassBase::TrampolineEmitPassBase;

  void runOnOperation() override {
    auto unit = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<FuncOp> coroFuncs;
    unit.walk([&](FuncOp funcOp) {
      if (funcOp->hasAttr("coro.wip"))
        coroFuncs.push_back(funcOp);
    });

    for (FuncOp funcOp : coroFuncs) {
      LLVM_DEBUG(llvm::dbgs()
                 << "=== TrampolineEmit: " << funcOp.getSymName() << " ===\n");

      // Lower FrameAddressOp before building the trampoline.
      // It needs the task frame global created by CoroFrameRewritePass.
      if (failed(lowerFrameAccessOps(funcOp, rewriter))) {
        signalPassFailure();
        return;
      }

      if (failed(emitTrampoline(funcOp, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
