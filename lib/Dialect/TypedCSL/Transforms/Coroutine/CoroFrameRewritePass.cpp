//===- CoroFrameRewritePass.cpp - Frame analysis and spill/reload ---------===//
//
// Analyzes coroutine frames, performs bin-packing layout, and rewrites all
// values that cross suspend points to go through frame slots.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/CoroFrameBuilder.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "coro-frame-rewrite"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_COROFRAMEREWRITEPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create CoroutineFrameAccess for a function.
/// For tasks: creates a global frame variable first.
/// The FuncOp constructor handles per-block frame pointer re-derivation.
static CoroutineFrameAccess createFrameAccess(FuncOp funcOp,
                                              const CoroFrameLayout &layout,
                                              OpBuilder &builder,
                                              unsigned slotAlign) {
  Location loc = funcOp.getLoc();

  if (funcOp.getIsTask()) {
    uint32_t frameSize = layout.maxTotalSize;
    auto ctx = funcOp.getContext();
    auto arrayTy = getCoroFrameArrayType(ctx, frameSize, slotAlign);
    auto elemTy = arrayTy.getElementType();
    auto globalName = (funcOp.getSymName() + "__coro_frame").str();

    // Zero-initialize so the state field (u16 at offset 0) is 0 on first entry.
    auto zero = IntegerAttr::get(elemTy, 0);
    SmallVector<TypedAttr> zeros(arrayTy.getShape()[0], zero);
    auto initAttr =
        csl::ArrayValueAttr::get(ctx, arrayTy.getShape(), elemTy, zeros);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(funcOp);
    // FIXME: Properly calculate alignment based on the fields in the frame. For
    // now, just use 8-byte alignment for everything
    auto alignAttr = IntegerAttr::get(
        IntegerType::get(funcOp.getContext(), 32, IntegerType::Unsigned), 8);
    GlobalOp::create(builder, loc, arrayTy, globalName,
                     /*initializer=*/initAttr, /*is_const=*/false,
                     /*is_export=*/false, /*is_extern=*/false,
                     /*linkname=*/nullptr, /*alignment=*/alignAttr);
  }

  builder.setInsertionPointToStart(&funcOp.getBody().front());
  return CoroutineFrameAccess(builder, loc, layout, funcOp, slotAlign);
}

/// Move VariableOps to frame slots.
static LogicalResult moveVariablesToFrame(FuncOp funcOp,
                                          CoroFrameAnalysis &analysis,
                                          CoroFrameLayout &layout,
                                          CoroutineFrameAccess &frameAccess,
                                          IRRewriter &rewriter) {
  Location loc = funcOp.getLoc();

  DenseMap<Value, unsigned> varToFieldIdx;
  for (auto [idx, field] : llvm::enumerate(analysis.staticFields)) {
    if (field.kind == FieldKind::Variable)
      varToFieldIdx[field.value] = idx;
  }

  SmallVector<VariableOp> varsToMove;
  funcOp.walk([&](VariableOp varOp) {
    if (varToFieldIdx.count(varOp.getResult()))
      varsToMove.push_back(varOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "    Moving " << varsToMove.size()
                          << " VariableOps to frame\n");

  for (VariableOp varOp : varsToMove) {
    unsigned fieldIdx = varToFieldIdx[varOp.getResult()];
    Type origTy = varOp.getType();
    bool isConst = mlir::cast<csl::LValueType>(origTy).getIsConst();

    if (Value init = varOp.getInitializer()) {
      rewriter.setInsertionPoint(varOp);
      frameAccess.storeSlot(rewriter, loc, fieldIdx, init);
    }

    SmallVector<OpOperand *> uses;
    for (OpOperand &use : varOp.getResult().getUses())
      uses.push_back(&use);

    for (OpOperand *use : uses) {
      rewriter.setInsertionPoint(use->getOwner());
      Value frameLval = frameAccess.getSlotLValue(rewriter, loc, fieldIdx);
      if (isConst)
        frameLval = ConstCastOp::create(rewriter, loc, origTy, frameLval);
      use->set(frameLval);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "      " << varOp.getName() << " → field " << fieldIdx
               << " @ offset " << layout.staticSlots[fieldIdx].offset << "\n");

    rewriter.eraseOp(varOp);
  }
  return success();
}

/// SSA spill: store-after-def for values crossing suspend points.
/// Returns the set of ops created for stores (to exclude from reload).
static SmallPtrSet<Operation *, 32>
spillStoreAfterDef(FuncOp funcOp, CoroFrameAnalysis &analysis,
                   CoroutineFrameAccess &frameAccess, IRRewriter &rewriter) {
  Location loc = funcOp.getLoc();
  SmallPtrSet<Operation *, 32> storeAfterDefOps;

  for (auto [fieldIdx, field] : llvm::enumerate(analysis.staticFields)) {
    if (!field.value || field.spillUses.empty())
      continue;

    if (auto blockArg = mlir::dyn_cast<BlockArgument>(field.value)) {
      rewriter.setInsertionPointToStart(blockArg.getOwner());
    } else {
      rewriter.setInsertionPointAfter(field.value.getDefiningOp());
    }

    Value toStore = field.value;
    if (mlir::isa<csl::LValueType>(field.value.getType())) {
      toStore = LoadOp::create(rewriter, loc, field.value);
      storeAfterDefOps.insert(toStore.getDefiningOp());
    }

    Block::iterator beforeStore = rewriter.getInsertionPoint();
    frameAccess.storeSlot(rewriter, loc, fieldIdx, toStore);
    for (auto it = beforeStore; it != rewriter.getInsertionPoint(); ++it)
      storeAfterDefOps.insert(&*it);

    LLVM_DEBUG(llvm::dbgs() << "    Spill: " << field.debugName << " (field "
                            << fieldIdx << ")\n");
  }
  return storeAfterDefOps;
}

/// Verify that no SSA value is used in a different block from where it's
/// defined. After spill/reload, every block must be self-contained (except
/// for entry block arguments which are always available).
static LogicalResult verifyCrossBlockValues(FuncOp funcOp) {
  Region &body = funcOp.getBody();
  Block &entry = body.front();

  SmallPtrSet<Value, 4> exempt;
  for (BlockArgument arg : entry.getArguments())
    exempt.insert(arg);

  for (Block &block : body) {
    for (Operation &op : block) {
      for (OpOperand &operand : op.getOpOperands()) {
        Value val = operand.get();
        if (exempt.contains(val))
          continue;
        Block *defBlock = val.getParentBlock();
        if (!defBlock || defBlock == &block)
          continue;

        InFlightDiagnostic diag =
            op.emitError("value used in a different block from its definition");
        diag.attachNote(val.getLoc()) << "value defined here";
        return failure();
      }
    }
  }
  return success();
}

/// SSA reload: at the top of each block where a spilled value is used
/// across a suspend, load it from the frame and replace uses.
static void reloadPerUseBlock(FuncOp funcOp, CoroFrameAnalysis &analysis,
                              CoroutineFrameAccess &frameAccess,
                              const SmallPtrSet<Operation *, 32> &storeOps,
                              IRRewriter &rewriter) {
  Location loc = funcOp.getLoc();

  for (auto [fieldIdx, field] : llvm::enumerate(analysis.staticFields)) {
    if (!field.value || field.spillUses.empty())
      continue;

    Block *defBlock = field.value.getParentBlock();
    llvm::SmallPtrSet<Block *, 8> reloadBlocks;
    for (OpOperand &use : field.value.getUses()) {
      if (storeOps.contains(use.getOwner()))
        continue;
      if (use.getOwner()->getBlock() != defBlock)
        reloadBlocks.insert(use.getOwner()->getBlock());
    }

    DenseMap<Block *, Value> reloadedInBlock;
    for (Block *block : reloadBlocks) {
      frameAccess.positionAfterFramePtr(rewriter, loc, block);
      Value reloaded = frameAccess.loadSlot(rewriter, loc, fieldIdx);
      reloadedInBlock[block] = reloaded;
    }

    SmallVector<std::pair<OpOperand *, Value>> replacements;
    for (OpOperand &use : field.value.getUses()) {
      if (storeOps.contains(use.getOwner()))
        continue;
      auto it = reloadedInBlock.find(use.getOwner()->getBlock());
      if (it != reloadedInBlock.end())
        replacements.push_back({&use, it->second});
    }
    for (auto [use, replacement] : replacements)
      use->set(replacement);

    LLVM_DEBUG(llvm::dbgs() << "    Reload: " << field.debugName << " in "
                            << reloadBlocks.size() << " blocks\n");
  }
}

//===----------------------------------------------------------------------===//
// Topological ordering of coro functions
//===----------------------------------------------------------------------===//

/// Collect all coro.wip functions in callee-first topological order.
///
/// We cannot use MLIR's CallGraph because CoroPrepareCFPass has already
/// lowered async CallOps into VariableOps + stores for offloaded calls
/// (@await_parallel, @await with dedicated worker). The call edges are gone
/// by this point. Instead, we scan each function's VariableOps for
/// CoroFrameType references to reconstruct the dependency graph.
static void
collectCoroFuncsInTopologicalOrder(Operation *unit,
                                   SmallVectorImpl<FuncOp> &result) {
  DenseMap<StringRef, FuncOp> byName;
  for (auto funcOp : unit->getRegion(0).getOps<FuncOp>()) {
    if (funcOp->hasAttr("coro.wip"))
      byName[funcOp.getSymName()] = funcOp;
  }

  // Build dependency edges: if A contains a VariableOp of type
  // !csl.coroframe<@B>, then B must be processed before A.
  DenseMap<FuncOp, unsigned> inDegree;
  DenseMap<FuncOp, SmallVector<FuncOp>> dependents; // callee → callers
  for (auto &[name, funcOp] : byName)
    inDegree[funcOp] = 0;

  for (auto &[name, funcOp] : byName) {
    funcOp.walk([&](VariableOp varOp) {
      // Variable types are !csl.lvalue<mutable !csl.coroframe<@callee>>.
      auto lvalTy = mlir::dyn_cast<csl::LValueType>(varOp.getType());
      if (!lvalTy)
        return;
      auto frameTy = mlir::dyn_cast<csl::CoroFrameType>(lvalTy.getValueType());
      if (!frameTy)
        return;
      LLVM_DEBUG(llvm::dbgs()
                 << "  " << name << " has coroframe var: " << frameTy << "\n");
      auto it = byName.find(frameTy.getSymbolName().getValue());
      if (it == byName.end() || it->second == funcOp)
        llvm_unreachable("coroframe references unknown or self coro function");
      dependents[it->second].push_back(funcOp);
      inDegree[funcOp]++;
      LLVM_DEBUG(llvm::dbgs()
                 << "    edge: " << it->first << " → " << name << "\n");
    });
  }

  // Kahn's algorithm: emit leaves first (in-degree 0).
  SmallVector<FuncOp> worklist;
  for (auto &[func, deg] : inDegree) {
    LLVM_DEBUG(llvm::dbgs()
               << "  in-degree[" << func.getSymName() << "] = " << deg << "\n");
    if (deg == 0)
      worklist.push_back(func);
  }
  while (!worklist.empty()) {
    FuncOp func = worklist.pop_back_val();
    result.push_back(func);
    LLVM_DEBUG(llvm::dbgs() << "  emit: " << func.getSymName() << "\n");
    for (FuncOp caller : dependents[func]) {
      if (--inDegree[caller] == 0)
        worklist.push_back(caller);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "  total emitted: " << result.size() << " of "
                          << byName.size() << " coro funcs\n");
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CoroFrameRewritePass
    : public typed_csl::impl::CoroFrameRewritePassBase<CoroFrameRewritePass> {
  using CoroFrameRewritePassBase::CoroFrameRewritePassBase;

  void runOnOperation() override {
    auto unit = getOperation();
    IRRewriter rewriter(&getContext());
    unsigned slotAlign = getCoroSlotAlignment(unit);

    // Collect coro.wip functions in callee-first topological order.
    // Callee frame sizes must be known before processing callers
    // (needed for AttrTypeReplacer: coroframe → [N]u8).
    SmallVector<FuncOp> coroFuncs;
    DenseMap<FuncOp, std::unique_ptr<AsyncFuncInfo>> funcInfos;

    collectCoroFuncsInTopologicalOrder(unit, coroFuncs);

    for (FuncOp funcOp : coroFuncs) {
      LLVM_DEBUG(llvm::dbgs() << "=== CoroFrameRewrite: " << funcOp.getSymName()
                              << " ===\n");

      // Convert callee !csl.coroframe types → [N]u8 before analysis
      // so that getSizeInBytes works on callee frame VariableOps.
      // Skip self-reference (not in funcInfos yet).
      // Note: FutureType is already lowered to *CoroFrameType by
      // CoroPrepareCFPass, so only CoroFrameType needs handling here.
      {
        AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](csl::CoroFrameType type)
                -> std::optional<std::pair<Type, WalkResult>> {
              auto calleeFuncOp =
                  unit.lookupSymbol<FuncOp>(type.getSymbolName().getValue());
              if (!calleeFuncOp)
                return std::nullopt;
              auto it = funcInfos.find(calleeFuncOp);
              if (it == funcInfos.end())
                return std::nullopt;
              uint32_t frameSize = it->second->layout.maxTotalSize;
              auto arrayTy =
                  getCoroFrameArrayType(&getContext(), frameSize, slotAlign);
              return std::make_pair(arrayTy, WalkResult::advance());
            });
        replacer.recursivelyReplaceElementsIn(funcOp, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/false,
                                              /*replaceTypes=*/true);
      }

      // Run frame analysis.
      CoroFrameBuilder frameBuilder(funcOp, &funcInfos, slotAlign);
      if (failed(frameBuilder.analyze())) {
        signalPassFailure();
        return;
      }
      auto layoutResult = frameBuilder.packLayout(slotAlign);
      if (failed(layoutResult)) {
        signalPassFailure();
        return;
      }

      auto info = std::make_unique<AsyncFuncInfo>();
      info->analysis = std::move(frameBuilder.getAnalysis());
      info->layout = *layoutResult;

      LLVM_DEBUG(llvm::dbgs()
                 << "  Frame: " << info->layout.maxTotalSize << " bytes, "
                 << info->analysis.staticFields.size() << " fields\n");

      funcInfos[funcOp] = std::move(info);
      auto &storedInfo = *funcInfos[funcOp];

      // Create frame access.
      auto frameAccess =
          createFrameAccess(funcOp, storedInfo.layout, rewriter, slotAlign);

      // Lower CoroParamsOps: replace each USE with a fresh const lvalue
      // at the param's convention offset. Per-use to avoid cross-block
      // lvalue references (each block gets its own derivation chain).
      {
        SmallVector<unsigned> paramFieldIndices;
        for (auto [fieldIdx, field] :
             llvm::enumerate(storedInfo.analysis.staticFields)) {
          if (field.kind == FieldKind::Param)
            paramFieldIndices.push_back(fieldIdx);
        }

        funcOp.walk([&](CoroParamsOp paramsOp) {
          Location loc = paramsOp.getLoc();
          for (unsigned i = 0; i < paramsOp.getNumResults(); ++i) {
            Value paramResult = paramsOp.getResult(i);
            auto constTy = mlir::cast<csl::LValueType>(paramResult.getType());
            unsigned fieldIdx = paramFieldIndices[i];

            SmallVector<OpOperand *> uses;
            for (OpOperand &use : paramResult.getUses())
              uses.push_back(&use);

            for (OpOperand *use : uses) {
              rewriter.setInsertionPoint(use->getOwner());
              Value lval = frameAccess.getSlotLValue(rewriter, loc, fieldIdx);
              Value constLval =
                  ConstCastOp::create(rewriter, loc, constTy, lval);
              use->set(constLval);
            }
          }
          rewriter.eraseOp(paramsOp);
        });
      }

      // Step A: Move VariableOps to frame.
      if (failed(moveVariablesToFrame(funcOp, storedInfo.analysis,
                                      storedInfo.layout, frameAccess,
                                      rewriter))) {
        signalPassFailure();
        return;
      }

      // Step B: SSA spill (store-after-def).
      auto storeOps = spillStoreAfterDef(funcOp, storedInfo.analysis,
                                         frameAccess, rewriter);

      // Step D: SSA reload (per-use-block).
      reloadPerUseBlock(funcOp, storedInfo.analysis, frameAccess, storeOps,
                        rewriter);

      // Convert self-referencing !csl.coroframe → [N]u8 now that we know the
      // size of the current function's frame.
      {
        AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](csl::CoroFrameType type)
                -> std::optional<std::pair<Type, WalkResult>> {
              auto calleeFuncOp =
                  unit.lookupSymbol<FuncOp>(type.getSymbolName().getValue());
              if (!calleeFuncOp)
                return std::nullopt;
              auto it = funcInfos.find(calleeFuncOp);
              if (it == funcInfos.end())
                return std::nullopt;
              uint32_t frameSize = it->second->layout.maxTotalSize;
              auto arrayTy =
                  getCoroFrameArrayType(&getContext(), frameSize, slotAlign);
              return std::make_pair(arrayTy, WalkResult::advance());
            });
        replacer.recursivelyReplaceElementsIn(funcOp, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/false,
                                              /*replaceTypes=*/true);
      }

      // Verify every block is self-contained after spill/reload.
      if (failed(verifyCrossBlockValues(funcOp))) {
        signalPassFailure();
        return;
      }

      // Clean up async_result_type now that frame layout is computed.
      if (funcOp.getAsyncResultTypeAttr())
        funcOp.removeAsyncResultTypeAttr();
    }
  }
};

} // namespace
