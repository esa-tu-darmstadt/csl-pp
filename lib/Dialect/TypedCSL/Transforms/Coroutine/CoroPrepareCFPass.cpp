//===- CoroPrepareCFPass.cpp - Prepare CF for coroutine transform ---------===//
//
// Lowers the async calling convention in three stages:
//
// Phase A — Pre-analysis & signature rewriting:
//    Computes task footprints (worker count per @await_parallel site),
//    annotates AwaitParallelOps with worker ID offsets, and generates the
//    shared worker pool (global struct array + trampoline tasks + bindings).
//    Rewrites async fn signatures: (params...) -> result  →  ([*]uN) -> void
//    where N depends on slotAlignment (u8 for align=1, u16 for align=2).
//    All async functions get the uniform signature fn([*]uN) void so that
//    function pointers can be stored in the worker pool without @ptrcast to
//    a comptime-only function pointer type (which cslc rejects).
//    Inserts CoroParamsOp at entry to extract params from the frame pointer.
//    Rewrites ReturnOps: store result, return void.
//    Tasks are excluded (their frame is a global).
//
// Phase B — FutureType lowering (AttrTypeReplacer, whole unit):
//    Replaces all !csl.future<@fn> with *!csl.coroframe<@fn>.  After this
//    point no FutureType remains in the IR.  CoroFrameRewritePass later
//    replaces !csl.coroframe<@fn> with [N x u8].
//
// Phase C — Call/await/parallel lowering (greedy rewrite, per async fn):
//
//    LowerAsyncCallToFrame:
//      Allocates a callee frame variable, stores state=0, propagates
//      workerid, stores params at convention offsets.
//
//    LowerAwaitToCall:
//      Single call (nop.term convention — if callee returns, it's done):
//        call(frame);
//      The CallOp is tagged with "coro.async_call" so a later pass
//      can split the block to ensure re-entry skips frame init.
//
//    LowerAwaitParallelToWorkerDispatch:
//      Stores frame/callee/caller_wid to worker pool arrays, activates
//      worker tasks via @activate, then joins:
//        while (any child not done) { suspend; }
//
//    FrameAddressOp is NOT lowered here — it needs the task frame global
//    which is created by CoroFrameRewritePass. It is lowered in
//    TrampolineEmitPass instead.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/CoroFrameBuilder.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/WorkerPoolBuilder.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <atomic>

#define DEBUG_TYPE "coro-prepare-cf"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_COROPREPARECFPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

//===----------------------------------------------------------------------===//
// Calling Convention Helpers
//===----------------------------------------------------------------------===//

static void storeAtOffset(OpBuilder &b, Location loc, Value framePtr,
                          uint32_t offset, Type storageType, Value data,
                          MLIRContext *ctx, unsigned slotAlign) {
  uint32_t elemOffset = byteToElementOffset(offset, slotAlign);
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto mutableElemLvalTy = csl::LValueType::get(elemTy, /*isConst=*/false);

  Value offsetVal =
      ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, elemOffset));
  Value elemLval =
      PtrIndexAccessOp::create(b, loc, mutableElemLvalTy, framePtr, offsetVal);

  auto elemPtrTy =
      csl::PointerType::get(elemTy, /*isConst=*/false, /*isArray=*/false);
  Value elemPtr = AddressOfOp::create(b, loc, elemPtrTy, elemLval);

  auto targetPtrTy =
      csl::PointerType::get(storageType, /*isConst=*/false, /*isArray=*/false);
  Value typedPtr = CastOp::create(b, loc, targetPtrTy, elemPtr).getResult();

  auto mutableLvalTy = csl::LValueType::get(storageType, /*isConst=*/false);
  Value lval = DereferenceOp::create(b, loc, mutableLvalTy, typedPtr);
  AssignOp::create(b, loc, lval, data);
}

static Value loadFromOffset(OpBuilder &b, Location loc, Value framePtr,
                            uint32_t offset, Type storageType, MLIRContext *ctx,
                            unsigned slotAlign) {
  uint32_t elemOffset = byteToElementOffset(offset, slotAlign);
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto mutableElemLvalTy = csl::LValueType::get(elemTy, /*isConst=*/false);

  Value offsetVal =
      ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, elemOffset));
  Value elemLval =
      PtrIndexAccessOp::create(b, loc, mutableElemLvalTy, framePtr, offsetVal);

  auto elemPtrTy =
      csl::PointerType::get(elemTy, /*isConst=*/false, /*isArray=*/false);
  Value elemPtr = AddressOfOp::create(b, loc, elemPtrTy, elemLval);

  auto targetPtrTy =
      csl::PointerType::get(storageType, /*isConst=*/false, /*isArray=*/false);
  Value typedPtr = CastOp::create(b, loc, targetPtrTy, elemPtr).getResult();

  auto mutableLvalTy = csl::LValueType::get(storageType, /*isConst=*/false);
  Value lval = DereferenceOp::create(b, loc, mutableLvalTy, typedPtr);
  return LoadOp::create(b, loc, lval);
}

//===----------------------------------------------------------------------===//
// Phase A: Signature Rewriting
//===----------------------------------------------------------------------===//

/// Rewrite an async fn's signature and body:
///   - Signature: (params...) -> result  →  ([*]uN) -> void
///   - Insert CoroParamsOp to extract param lvalues from the frame pointer
///   - Rewrite all return ops: store result, return void
///
/// Using a type-erased many-pointer gives all async functions the uniform
/// signature fn([*]uN) void.  This avoids @ptrcast to function pointer types
/// (which cslc rejects) when storing callee addresses in the worker pool.
static LogicalResult rewriteAsyncFnSignature(FuncOp funcOp,
                                             IRRewriter &rewriter,
                                             unsigned slotAlign) {
  Location loc = funcOp.getLoc();
  MLIRContext *ctx = funcOp.getContext();

  // All async functions take a type-erased many-pointer to the frame.
  auto bytePtrTy = getCoroFramePtrType(ctx, slotAlign);

  auto newFnType = FunctionType::get(ctx, {bytePtrTy}, {});
  funcOp.setFunctionType(newFnType);

  Block &entry = funcOp.getBody().front();
  unsigned numOldArgs = entry.getNumArguments();
  SmallVector<Value> oldArgs;
  for (unsigned i = 0; i < numOldArgs; ++i)
    oldArgs.push_back(entry.getArgument(i));

  // TypedCSL wraps function params in lvalues.
  auto framePtrLvalTy = csl::LValueType::get(bytePtrTy, /*isConst=*/true);
  BlockArgument framePtrLvalArg = entry.addArgument(framePtrLvalTy, loc);

  rewriter.setInsertionPointToStart(&entry);
  Value framePtrArg = LoadOp::create(rewriter, loc, framePtrLvalArg);

  // Insert CoroParamsOp to extract param lvalues from the frame.
  SmallVector<Type> paramLvalTypes;
  for (unsigned i = 0; i < numOldArgs; ++i)
    paramLvalTypes.push_back(oldArgs[i].getType());

  auto coroParamsOp =
      CoroParamsOp::create(rewriter, loc, paramLvalTypes, framePtrArg);

  // Replace old block arg uses with CoroParamsOp results.
  for (unsigned i = 0; i < numOldArgs; ++i)
    oldArgs[i].replaceAllUsesWith(coroParamsOp.getResult(i));

  // Remove old block args.
  for (int i = numOldArgs - 1; i >= 0; --i)
    entry.eraseArgument(i);

  // Rewrite all return ops: store result, return void.
  // The byte pointer is loaded locally at each return site (not hoisted to the
  // entry block) because return ops may be in different blocks separated by
  // suspend points — a hoisted value would cross those blocks and require
  // spilling.
  auto asyncResultType = funcOp.getAsyncResultType();
  SmallVector<ReturnOp> returns;
  funcOp.walk([&](ReturnOp op) { returns.push_back(op); });

  for (ReturnOp returnOp : returns) {
    rewriter.setInsertionPoint(returnOp);

    // Load frame pointer locally at this return site.
    Value bytePtr = LoadOp::create(rewriter, loc, framePtrLvalArg);

    if (returnOp.getOperand() && asyncResultType) {
      uint32_t resultOffset = coroConventionOffset(kCoroConventionPrefixSize,
                                                   *asyncResultType, slotAlign);
      Value resultVal = returnOp.getOperand();
      if (mlir::isa<csl::LValueType>(resultVal.getType()))
        resultVal = LoadOp::create(rewriter, loc, resultVal);
      storeAtOffset(rewriter, loc, bytePtr, resultOffset, *asyncResultType,
                    resultVal, ctx, slotAlign);
    }

    // With nop.term, no done-sentinel is needed: if the function returns
    // normally the caller knows the callee is done (nop.term would have
    // killed the task otherwise).
    ReturnOp::create(rewriter, loc, returnOp.getNumBreakingRegions());
    rewriter.eraseOp(returnOp);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Rewrote signature: " << funcOp.getSymName()
                          << " : " << newFnType << "\n");
  return success();
}

/// Match calls to async functions (check callee's isAsync, not return type).
struct LowerAsyncCallToFrame : public OpRewritePattern<CallOp> {
  unsigned slotAlign;

  LowerAsyncCallToFrame(MLIRContext *ctx, unsigned slotAlign)
      : OpRewritePattern(ctx), slotAlign(slotAlign) {}

  LogicalResult matchAndRewrite(CallOp callOp,
                                PatternRewriter &rewriter) const override {
    // Check if the callee is an async function.
    auto unit = callOp->getParentOfType<csl::CompilationUnitOp>();
    auto calleeFuncOp =
        unit.lookupSymbol<FuncOp>(callOp.getCalleeAttr().getLeafReference());
    if (!calleeFuncOp || !calleeFuncOp.getIsAsync() || calleeFuncOp.getIsTask())
      return failure();

    // Only match calls that still return *CoroFrameType (not yet lowered).
    // FutureType was replaced by *CoroFrameType in Phase B.
    if (!callOp.getResult())
      return failure();
    auto resultPtrTy =
        mlir::dyn_cast<csl::PointerType>(callOp.getResult().getType());
    if (!resultPtrTy ||
        !mlir::isa<csl::CoroFrameType>(resultPtrTy.getPointeeType()))
      return failure();

    Location loc = callOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    StringRef calleeName = calleeFuncOp.getSymName();

    auto coroFrameTy =
        csl::CoroFrameType::get(ctx, /*moduleIdentifier=*/std::nullopt,
                                FlatSymbolRefAttr::get(ctx, calleeName));

    // Allocate callee frame variable at the function entry.
    static std::atomic<unsigned> frameCounter{0};
    auto funcOp = callOp->getParentOfType<FuncOp>();
    auto frameLvalTy = csl::LValueType::get(coroFrameTy, /*isConst=*/false);
    auto frameVarName = (calleeName + "__frame_" + Twine(frameCounter++)).str();
    VariableOp varOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      varOp = VariableOp::create(rewriter, loc, frameLvalTy, frameVarName,
                                 /*initializer=*/Value(), /*is_const=*/false);
    }

    // Get frame element pointer for field init.
    rewriter.setInsertionPoint(callOp);
    auto ptrTy = csl::PointerType::get(coroFrameTy, /*isConst=*/false,
                                       /*isArray=*/false);
    Value promise =
        AddressOfOp::create(rewriter, loc, ptrTy, varOp.getResult());
    auto promiseTyped = mlir::cast<TypedValue<csl::PointerType>>(promise);
    Value framePtr = CoroutineFrameAccess::ensureFramePtr(
        rewriter, loc, promiseTyped, slotAlign);

    // Init state = 0.
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    Value zero = ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 0));
    storeAtOffset(rewriter, loc, framePtr, kCoroStateOffset, ui16Ty, zero, ctx,
                  slotAlign);

    // Workerid is propagated by the caller's await or await_parallel, so no
    // need to init it here.

    // Init params at convention offsets (after state + workerid prefix).
    uint32_t offset = kCoroConventionPrefixSize;
    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      Value arg = callOp.getOperand(i);
      Type argType = arg.getType();
      if (auto lvalTy = mlir::dyn_cast<csl::LValueType>(argType)) {
        argType = lvalTy.getValueType();
        arg = LoadOp::create(rewriter, loc, arg);
      }
      offset = coroConventionOffset(offset, argType, slotAlign);
      storeAtOffset(rewriter, loc, framePtr, offset, argType, arg, ctx,
                    slotAlign);
      auto sizeOpt = csl::getSizeInBytes(argType);
      assert(sizeOpt && "param type must have known size");
      offset += *sizeOpt;
    }

    // FutureType was replaced by *CoroFrameType in Phase B, so the
    // replacement type matches directly.
    rewriter.replaceOp(callOp, promise);
    return success();
  }
};

/// Get the caller's task ID as a u16 Value (works for both tasks and async
/// fns — every async fn runs in the context of some hardware task).
static Value getCallerWorkerId(OpBuilder &b, Location loc) {
  auto *ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);
  Value cur = BuiltinCallOp::create(b, loc, localTaskIdTy,
                                    csl::BuiltinKind::CurrentTask, ValueRange{})
                  .getResult();
  return BuiltinCallOp::create(b, loc, ui16Ty, csl::BuiltinKind::GetInt,
                               ValueRange{cur})
      .getResult();
}

/// Get lvalue to `__coro_pool_<fieldName>[idx]` (SoA pool).
static Value getSlotField(const WorkerPoolInfo &pool, OpBuilder &b,
                          Location loc, Value idx, StringRef fieldName) {
  auto it = pool.fields.find(fieldName);
  assert(it != pool.fields.end() && "unknown pool field");
  const auto &field = it->second;
  auto arrayLvalTy = csl::LValueType::get(field.arrayTy, /*isConst=*/false);
  Value arrayLval = createGlobalRef(b, loc, arrayLvalTy, field.globalName);
  auto elemLvalTy = csl::LValueType::get(field.elementTy, /*isConst=*/false);
  return ArrayIndexAccessOp::create(b, loc, elemLvalTy, arrayLval, idx);
}

/// Emit `%counter = csl.variable u16`, `csl.assign %counter, N`, return
/// `%counter_ptr = &%counter` as a `*u16`. The variable is a function-local
/// CSL var that will be spilled into the caller's coro frame by
/// CoroFrameRewritePass when it detects cross-block uses (the post-suspend
/// load lives in a different block from the pre-suspend store).
static std::pair<Value, Value> emitJoinCounter(OpBuilder &b, Location loc,
                                               unsigned initCount) {
  auto *ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto u16PtrTy =
      csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/false);
  auto lvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);

  // Allocate the variable at the function entry so it's available across
  // blocks (the await is going to introduce a SuspendBr, splitting the
  // block; CoroFrameRewrite handles the cross-block lifetime).
  // getParentOfType starts at the op's parent, so it never matches the op
  // itself. When the await sits directly in the function's own body block the
  // insertion block's parent already *is* the FuncOp, and asking for an
  // enclosing one yields null. Only walk up when we are nested inside some
  // other region (an `if`/`while` body).
  Operation *blockParent = b.getInsertionBlock()->getParentOp();
  auto funcOp = mlir::dyn_cast<FuncOp>(blockParent);
  if (!funcOp)
    funcOp = blockParent->getParentOfType<FuncOp>();
  assert(funcOp && "join counter emitted outside any function");
  static std::atomic<unsigned> counterId{0};
  std::string name = ("__coro_join_counter_" + Twine(counterId++)).str();
  VariableOp varOp;
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&funcOp.getBody().front());
    varOp = VariableOp::create(b, loc, lvalTy, name,
                               /*initializer=*/Value(), /*is_const=*/false);
  }

  Value n = ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, initCount));
  AssignOp::create(b, loc, varOp.getResult(), n);
  Value counterPtr = AddressOfOp::create(b, loc, u16PtrTy, varOp.getResult());
  return {varOp.getResult(), counterPtr};
}

/// Load the await result from the callee frame. Replaces or erases the op.
static LogicalResult extractAwaitResult(AwaitOp awaitOp, StringRef calleeName,
                                        Value frameBytePtr,
                                        PatternRewriter &rewriter,
                                        unsigned slotAlign) {
  if (!awaitOp.getResult()) {
    rewriter.eraseOp(awaitOp);
    return success();
  }

  Location loc = awaitOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  auto unit = awaitOp->getParentOfType<csl::CompilationUnitOp>();
  auto calleeFuncOp = unit.lookupSymbol<FuncOp>(calleeName);
  if (!calleeFuncOp)
    return awaitOp->emitError("could not find async callee '")
           << calleeName << "'";

  auto asyncResultType = calleeFuncOp.getAsyncResultType();
  if (!asyncResultType)
    return awaitOp->emitError("callee has no async_result_type");

  uint32_t resultOffset = coroConventionOffset(kCoroConventionPrefixSize,
                                               *asyncResultType, slotAlign);
  Value result = loadFromOffset(rewriter, loc, frameBytePtr, resultOffset,
                                *asyncResultType, ctx, slotAlign);
  rewriter.replaceOp(awaitOp, result);
  return success();
}

/// Lower AwaitOp to a single call (nop.term convention).
///
/// With nop.term, if a callee suspends, it kills the task instantly.
/// If the callee returns normally, it is done. No polling loop or
/// state check needed.
///
/// The CallOp is tagged with "coro.async_call" so that a later pass
/// (splitAtAsyncCalls) splits the block before the call. This ensures
/// the frame init ops (state=0 store, param stores) are in a separate
/// trampoline state, so they are not re-executed on task reactivation.
struct LowerAwaitToCall : public OpRewritePattern<AwaitOp> {
  unsigned slotAlign;

  LowerAwaitToCall(MLIRContext *ctx, unsigned slotAlign)
      : OpRewritePattern(ctx), slotAlign(slotAlign) {}

  LogicalResult matchAndRewrite(AwaitOp awaitOp,
                                PatternRewriter &rewriter) const override {
    // Offloaded awaits are handled by LowerOffloadedAwaitToWorkerDispatch.
    if (awaitOp.getOffload())
      return failure();

    Location loc = awaitOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // FutureType was replaced by *CoroFrameType in Phase B.
    // Use raw operand access since the typed accessor expects FutureType.
    Value promise = awaitOp->getOperand(0);
    auto ptrTy = mlir::cast<csl::PointerType>(promise.getType());
    auto coroTy = mlir::cast<csl::CoroFrameType>(ptrTy.getPointeeType());
    StringRef calleeName = coroTy.getSymbolName().getValue();

    auto promiseTyped = mlir::cast<TypedValue<csl::PointerType>>(promise);
    Value ptr = CoroutineFrameAccess::ensureFramePtr(rewriter, loc,
                                                     promiseTyped, slotAlign);

    rewriter.setInsertionPoint(awaitOp);

    // Single call: if callee returns, it's done (nop.term kills the task
    // on callee suspend). Tag with "coro.async_call" for block splitting.
    auto callOp = CallOp::create(rewriter, loc, TypeRange{},
                                 FlatSymbolRefAttr::get(ctx, calleeName),
                                 ValueRange{ptr}, {}, {});
    callOp->setAttr("coro.async_call", rewriter.getUnitAttr());

    rewriter.setInsertionPointAfter(callOp);
    return extractAwaitResult(awaitOp, calleeName, ptr, rewriter, slotAlign);
  }
};

//===----------------------------------------------------------------------===//
// Phase C: @await_parallel lowering
//===----------------------------------------------------------------------===//

/// Lower AwaitParallelOp to dynamic worker dispatch with the first callee
/// running inline on the caller's task.
///
/// For @await_parallel(a(), b(), c()) on caller task T_c:
///   1. Allocate a local `%counter: u16` (will be spilled to T_c's frame).
///   2. Assign `%counter = N` (total callees, including the inline one).
///   3. For callees 1..N-1: claim a worker task, runtime-bind __coro_worker,
///      fill pool slot (frame, callee, &counter, caller_wid), @activate.
///   4. Call calleeFn_0(framePtr_0) inline (tagged coro.async_call so the
///      block is split: pre-call setup is in its own state, the call site
///      is in another state — on re-entry the workers are NOT re-dispatched).
///   5. Decrement *(&counter) by 1. If still > 0, SuspendOp (wait for the
///      remaining workers to drive the last decrement and activate us).
struct LowerAwaitParallelToWorkerDispatch
    : public OpRewritePattern<AwaitParallelOp> {
  const WorkerPoolInfo &pool;
  unsigned slotAlign;

  LowerAwaitParallelToWorkerDispatch(MLIRContext *ctx,
                                     const WorkerPoolInfo &pool,
                                     unsigned slotAlign)
      : OpRewritePattern(ctx), pool(pool), slotAlign(slotAlign) {}

  LogicalResult matchAndRewrite(AwaitParallelOp apOp,
                                PatternRewriter &rewriter) const override {
    Location loc = apOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto boolTy = IntegerType::get(ctx, 1);
    auto unit = apOp->getParentOfType<csl::CompilationUnitOp>();
    rewriter.setInsertionPoint(apOp);

    // FutureType was replaced by *CoroFrameType in Phase B; the operand
    // type is *CoroFrameType.
    SmallVector<Value> framePtrs;
    SmallVector<StringRef> calleeNames;
    for (Value future : apOp->getOperands()) {
      auto ptrTy = mlir::cast<csl::PointerType>(future.getType());
      auto coroTy = mlir::cast<csl::CoroFrameType>(ptrTy.getPointeeType());
      calleeNames.push_back(coroTy.getSymbolName().getValue());

      auto typedPtr = mlir::cast<TypedValue<csl::PointerType>>(future);
      framePtrs.push_back(CoroutineFrameAccess::ensureFramePtr(
          rewriter, loc, typedPtr, slotAlign));
    }

    unsigned N = framePtrs.size();
    assert(N >= 1 && "await_parallel with zero callees");

    // Caller's task id (used as the activate target for workers).
    Value callerWorkerId = getCallerWorkerId(rewriter, loc);

    // %counter = csl.variable u16; assign N; %counter_ptr = &counter
    auto [counterVar, counterPtr] = emitJoinCounter(rewriter, loc, N);
    (void)counterVar; // Lifetime-extended by spill; we only need the ptr.

    // Worker entry func ref (looked up once).
    auto voidFnType = FunctionType::get(ctx, {}, {});
    Value workerEntryRef =
        createFuncRef(rewriter, loc, voidFnType, "__coro_worker");

    auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);

    // Dispatch callees 1..N-1 to claimed workers. Collect the claimed
    // tasks separately so we can activate them after all slots are filled.
    SmallVector<Value> claimedTasks;
    claimedTasks.reserve(N - 1);
    for (unsigned i = 1; i < N; ++i) {
      Value framePtr = framePtrs[i];

      // Claim a fresh worker task ID.
      Value workerTask =
          BuiltinCallOp::create(rewriter, loc, localTaskIdTy,
                                csl::BuiltinKind::ClaimLocalTask, ValueRange{})
              .getResult();
      Value workerId =
          BuiltinCallOp::create(rewriter, loc, ui16Ty, csl::BuiltinKind::GetInt,
                                ValueRange{workerTask})
              .getResult();

      // Runtime-bind __coro_worker to the claimed task.
      BuiltinCallOp::create(rewriter, loc, Type(),
                            csl::BuiltinKind::BindLocalTask,
                            ValueRange{workerEntryRef, workerTask});

      // pool_frame[workerId] = framePtr
      AssignOp::create(rewriter, loc,
                       getSlotField(pool, rewriter, loc, workerId, "frame"),
                       framePtr);

      // pool_callee[workerId] = &calleeFn_i
      auto calleeFuncOp = unit.lookupSymbol<FuncOp>(calleeNames[i]);
      auto funcRefAttr =
          csl::FuncRefAttr::get(ctx, /*module_identifier=*/std::nullopt,
                                FlatSymbolRefAttr::get(ctx, calleeNames[i]),
                                calleeFuncOp.getFunctionType());
      Value funcAddr = ConstantOp::create(rewriter, loc,
                                          csl::FuncAddrAttr::get(funcRefAttr));
      AssignOp::create(rewriter, loc,
                       getSlotField(pool, rewriter, loc, workerId, "callee"),
                       funcAddr);

      // pool_counter_ptr[workerId] = %counter_ptr
      AssignOp::create(
          rewriter, loc,
          getSlotField(pool, rewriter, loc, workerId, "counter_ptr"),
          counterPtr);

      // pool_activate[workerId] = caller_wid
      AssignOp::create(rewriter, loc,
                       getSlotField(pool, rewriter, loc, workerId, "activate"),
                       callerWorkerId);

      claimedTasks.push_back(workerTask);
    }

    // Activate the dispatched workers.
    for (Value t : claimedTasks) {
      BuiltinCallOp::create(rewriter, loc, Type(), csl::BuiltinKind::Activate,
                            ValueRange{t});
    }

    // Inline-call callee 0 on the caller's task. Tag the call with
    // coro.async_call so a later pass splits the block at the call site —
    // ensuring re-entry on internal suspend doesn't re-dispatch workers.
    auto inlineCall = CallOp::create(
        rewriter, loc, TypeRange{}, FlatSymbolRefAttr::get(ctx, calleeNames[0]),
        ValueRange{framePtrs[0]}, /*arg_attrs=*/ArrayAttr(),
        /*res_attrs=*/ArrayAttr());
    inlineCall->setAttr("coro.async_call", rewriter.getUnitAttr());

    rewriter.setInsertionPointAfter(inlineCall);

    // Decrement *counter_ptr by 1 and conditionally suspend.
    auto u16LvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
    Value ctrLval = DereferenceOp::create(rewriter, loc, u16LvalTy, counterPtr);
    Value cur = LoadOp::create(rewriter, loc, ctrLval);
    Value one = ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 1));
    Value newCount = BinaryOp::create(rewriter, loc, ui16Ty,
                                      csl::BinaryOpKind::Sub, cur, one);
    AssignOp::create(rewriter, loc, ctrLval, newCount);

    // if (new_count != 0) suspend
    Value zero = ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 0));
    Value notDone = BinaryOp::create(rewriter, loc, boolTy,
                                     csl::BinaryOpKind::Ne, newCount, zero);
    auto ifNotDone = IfOp::create(rewriter, loc, TypeRange{}, notDone);
    {
      Block *thenBlock = rewriter.createBlock(&ifNotDone.getThenRegion());
      rewriter.setInsertionPointToEnd(thenBlock);
      SuspendOp::create(rewriter, loc);
      YieldOp::create(rewriter, loc, /*value=*/Value());
    }
    rewriter.setInsertionPointAfter(ifNotDone);

    rewriter.eraseOp(apOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase C: offloaded @await lowering
//===----------------------------------------------------------------------===//

/// Lower offloaded AwaitOp to dynamic worker dispatch + suspend + result.
///
/// Single callee, no inlining: the user explicitly asked for off-load
/// (callee runs on its own task so HW events can drive it independently).
struct LowerOffloadedAwaitToWorkerDispatch : public OpRewritePattern<AwaitOp> {
  const WorkerPoolInfo &pool;
  unsigned slotAlign;

  LowerOffloadedAwaitToWorkerDispatch(MLIRContext *ctx,
                                      const WorkerPoolInfo &pool,
                                      unsigned slotAlign)
      : OpRewritePattern(ctx), pool(pool), slotAlign(slotAlign) {}

  LogicalResult matchAndRewrite(AwaitOp awaitOp,
                                PatternRewriter &rewriter) const override {
    if (!awaitOp.getOffload())
      return failure();

    Location loc = awaitOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto unit = awaitOp->getParentOfType<csl::CompilationUnitOp>();

    // Extract frame pointer and callee name from operand.
    Value promise = awaitOp->getOperand(0);
    auto ptrTy = mlir::cast<csl::PointerType>(promise.getType());
    auto coroTy = mlir::cast<csl::CoroFrameType>(ptrTy.getPointeeType());
    StringRef calleeName = coroTy.getSymbolName().getValue();

    auto promiseTyped = mlir::cast<TypedValue<csl::PointerType>>(promise);
    Value framePtr = CoroutineFrameAccess::ensureFramePtr(
        rewriter, loc, promiseTyped, slotAlign);

    rewriter.setInsertionPoint(awaitOp);

    // Caller's task id (used as the activate target for the worker).
    Value callerWorkerId = getCallerWorkerId(rewriter, loc);

    // %counter = 1; %counter_ptr = &counter.
    auto [counterVar, counterPtr] = emitJoinCounter(rewriter, loc, 1);
    (void)counterVar;

    auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);

    // Claim a worker task ID.
    Value workerTask =
        BuiltinCallOp::create(rewriter, loc, localTaskIdTy,
                              csl::BuiltinKind::ClaimLocalTask, ValueRange{})
            .getResult();
    Value workerId =
        BuiltinCallOp::create(rewriter, loc, ui16Ty, csl::BuiltinKind::GetInt,
                              ValueRange{workerTask})
            .getResult();

    // Runtime-bind __coro_worker to the claimed task.
    auto voidFnType = FunctionType::get(ctx, {}, {});
    Value workerEntryRef =
        createFuncRef(rewriter, loc, voidFnType, "__coro_worker");
    BuiltinCallOp::create(rewriter, loc, Type(),
                          csl::BuiltinKind::BindLocalTask,
                          ValueRange{workerEntryRef, workerTask});

    // pool_frame[workerId] = framePtr
    AssignOp::create(rewriter, loc,
                     getSlotField(pool, rewriter, loc, workerId, "frame"),
                     framePtr);

    // pool_callee[workerId] = &callee_fn
    auto calleeFuncOp = unit.lookupSymbol<FuncOp>(calleeName);
    auto funcRefAttr =
        csl::FuncRefAttr::get(ctx, /*module_identifier=*/std::nullopt,
                              FlatSymbolRefAttr::get(ctx, calleeName),
                              calleeFuncOp.getFunctionType());
    Value funcAddr =
        ConstantOp::create(rewriter, loc, csl::FuncAddrAttr::get(funcRefAttr));
    AssignOp::create(rewriter, loc,
                     getSlotField(pool, rewriter, loc, workerId, "callee"),
                     funcAddr);

    // pool_counter_ptr[workerId] = %counter_ptr
    AssignOp::create(rewriter, loc,
                     getSlotField(pool, rewriter, loc, workerId, "counter_ptr"),
                     counterPtr);

    // pool_activate[workerId] = caller_wid
    AssignOp::create(rewriter, loc,
                     getSlotField(pool, rewriter, loc, workerId, "activate"),
                     callerWorkerId);

    // Activate the worker.
    BuiltinCallOp::create(rewriter, loc, Type(), csl::BuiltinKind::Activate,
                          ValueRange{workerTask});

    // Suspend until the worker drives the last decrement and activates us.
    SuspendOp::create(rewriter, loc);

    return extractAwaitResult(awaitOp, calleeName, framePtr, rewriter,
                              slotAlign);
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

struct CoroPrepareCFPass
    : public typed_csl::impl::CoroPrepareCFPassBase<CoroPrepareCFPass> {
  using CoroPrepareCFPassBase::CoroPrepareCFPassBase;

  WorkerPoolBuilder poolBuilder;

  void runOnOperation() override {
    unsigned slotAlign = slotAlignment;
    auto unit = getOperation();
    unit->setAttr(kCoroSlotAlignAttr,
                  IntegerAttr::get(IntegerType::get(&getContext(), 32,
                                                    IntegerType::Unsigned),
                                   slotAlign));
    IRRewriter rewriter(&getContext());

    // The stackless lowering has no place for the callee results of
    // @await_parallel.
    bool usesResults = false;
    unit.walk([&](AwaitParallelOp apOp) {
      if (!apOp.getResult())
        return;
      apOp.emitError("the results of @await_parallel are not supported with "
                     "--coro-mode=trampoline; use the default --coro-mode, "
                     "or let the callees write their results through "
                     "pointers");
      usesResults = true;
    });
    if (usesResults) {
      signalPassFailure();
      return;
    }

    // Phase A — collect async-task bindings (for diagnostics) and emit the
    // SoA worker pool + shared __coro_worker trampoline if the unit has any
    // @await_parallel or offloaded @await site.
    if (failed(poolBuilder.analyze(unit))) {
      signalPassFailure();
      return;
    }
    if (failed(poolBuilder.generate(unit, rewriter, slotAlign))) {
      signalPassFailure();
      return;
    }
    const WorkerPoolInfo &workerPool = poolBuilder.getPoolInfo();

    // Phase A: Rewrite all async fn signatures (not tasks).
    // After this, all async fns have the uniform frame pointer signature
    // with CoroParamsOp at entry and returns rewritten to store the
    // result. Tasks are excluded: they keep their signature (coro frame
    // is a global), and the trampoline resets their state to 0 (entry)
    // so they restart on the next HW activation.
    unit.walk([&](FuncOp funcOp) {
      if (!funcOp.getIsAsync() || funcOp.getIsTask())
        return;
      if (failed(rewriteAsyncFnSignature(funcOp, rewriter, slotAlign))) {
        signalPassFailure();
        return;
      }
    });

    // Lower FutureType → *CoroFrameType throughout the compilation unit.
    // After this, no FutureType remains in the IR. Phase C patterns work
    // exclusively with *CoroFrameType.
    {
      AttrTypeReplacer replacer;
      replacer.addReplacement(
          [](csl::FutureType type)
              -> std::optional<std::pair<Type, WalkResult>> {
            auto coroFrameTy = csl::CoroFrameType::get(
                type.getContext(), type.getModuleIdentifier(),
                type.getSymbolName());
            auto ptrTy = csl::PointerType::get(coroFrameTy, /*isConst=*/false,
                                               /*isArray=*/false);
            return std::make_pair(ptrTy, WalkResult::advance());
          });
      replacer.recursivelyReplaceElementsIn(unit, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
    }

    // Phase C: Lower calls to async fns, awaits, suspending for loops.
    // All callee signatures are already rewritten so call type-checking
    // succeeds.  FutureType has been lowered to *CoroFrameType.
    SmallVector<FuncOp> asyncFuncs;
    unit.walk([&](FuncOp funcOp) {
      if (funcOp.getIsAsync())
        asyncFuncs.push_back(funcOp);
    });

    for (FuncOp funcOp : asyncFuncs) {
      LLVM_DEBUG(llvm::dbgs() << "=== CoroPrepareCF Phase C: "
                              << funcOp.getSymName() << " ===\n");

      RewritePatternSet patterns(&getContext());
      patterns.add<LowerAsyncCallToFrame>(&getContext(), slotAlign);
      patterns.add<LowerAwaitToCall>(&getContext(), slotAlign);
      if (poolBuilder.hasAsyncDispatch()) {
        patterns.add<LowerAwaitParallelToWorkerDispatch>(&getContext(),
                                                         workerPool, slotAlign);
        patterns.add<LowerOffloadedAwaitToWorkerDispatch>(
            &getContext(), workerPool, slotAlign);
      }

      GreedyRewriteConfig config;
      config.setStrictness(GreedyRewriteStrictness::ExistingOps);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
      LLVM_DEBUG(llvm::dbgs() << "=== IR after Phase C for "
                              << funcOp.getSymName() << " ===\n";
                 funcOp.dump());
    }

    // Phase D: Replace is_async with coro.wip marker.
    // Done in a separate walk after all Phase C iterations so that
    // Phase C's callee detection (getIsAsync) works throughout.
    unit.walk([&](FuncOp funcOp) {
      if (!funcOp.getIsAsync())
        return;
      funcOp->setDiscardableAttr("coro.wip", rewriter.getUnitAttr());
      funcOp.removeIsAsyncAttr();
      // Keep async_result_type — CoroFrameRewritePass needs it to allocate
      // the result slot in the frame. It is cleaned up there instead.
    });
  }
};

} // namespace
