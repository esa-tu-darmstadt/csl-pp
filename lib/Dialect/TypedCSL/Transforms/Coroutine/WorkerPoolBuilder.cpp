//===- WorkerPoolBuilder.cpp - Worker pool generation ---------*- C++ -*-===//
//
// Emits the SoA worker pool globals and the shared `__coro_worker` task
// used by the trampoline backend's dynamic dispatch lowering.
//
// See WorkerPoolBuilder.h for the high-level design and IR layout.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/WorkerPoolBuilder.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/CoroFrameBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "coro-worker-pool"

using namespace mlir;
using namespace mlir::typed_csl;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

/// Pool size: covers local_task_id 0..28 (the claimable range on both
/// archs). Entries 0..7 are unused on WSE-3 since `@claim_local_task` only
/// returns ids in [8,28] there — harmless ~7×6-byte waste.
static constexpr uint32_t kPoolSize = 29;

static constexpr llvm::StringLiteral kPoolFramePrefix = "__coro_pool_frame";
static constexpr llvm::StringLiteral kPoolCalleePrefix = "__coro_pool_callee";
static constexpr llvm::StringLiteral kPoolCounterPtrPrefix =
    "__coro_pool_counter_ptr";
static constexpr llvm::StringLiteral kPoolActivatePrefix =
    "__coro_pool_activate";
static constexpr llvm::StringLiteral kWorkerEntryName = "__coro_worker";

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract a uint16_t task ID from a ConstantOp operand.
static std::optional<uint16_t> extractTaskIdValue(ConstantOp constOp) {
  TypedAttr val = constOp.getValue();
  if (auto localAttr = mlir::dyn_cast<csl::LocalTaskIdAttr>(val))
    return localAttr.getValue();
  if (auto controlAttr = mlir::dyn_cast<csl::ControlTaskIdAttr>(val))
    return controlAttr.getValue();
  // DataTaskIdAttr stores a source (color/queue), not a plain integer.
  return std::nullopt;
}

/// Get an lvalue to `__coro_pool_<fieldName>[idx]`.
static Value getPoolFieldLValue(OpBuilder &b, Location loc,
                                const WorkerPoolInfo &pool, Value idx,
                                StringRef fieldName) {
  auto it = pool.fields.find(fieldName);
  assert(it != pool.fields.end() && "unknown pool field");
  const auto &field = it->second;
  auto arrayLvalTy = csl::LValueType::get(field.arrayTy, /*isConst=*/false);
  Value arrayLval = createGlobalRef(b, loc, arrayLvalTy, field.globalName);
  auto elemLvalTy = csl::LValueType::get(field.elementTy, /*isConst=*/false);
  return ArrayIndexAccessOp::create(b, loc, elemLvalTy, arrayLval, idx);
}

//===----------------------------------------------------------------------===//
// Analyze
//===----------------------------------------------------------------------===//

LogicalResult WorkerPoolBuilder::analyze(csl::CompilationUnitOp unit) {
  bindings.clear();
  hasAsyncDispatchFlag = false;

  auto isBindKind = [](csl::BuiltinKind k) {
    return k == csl::BuiltinKind::BindLocalTask ||
           k == csl::BuiltinKind::BindControlTask ||
           k == csl::BuiltinKind::BindDataTask;
  };

  LogicalResult result = success();
  unit.walk([&](BuiltinCallOp bindOp) {
    csl::BuiltinKind kind = bindOp.getBuiltin();
    if (!isBindKind(kind))
      return;
    if (bindOp.getArgs().size() < 2)
      return;

    csl::FuncRefAttr funcRef = getFuncRef(bindOp.getArgs()[0]);
    FuncOp funcOp;
    if (funcRef)
      funcOp = unit.lookupSymbol<FuncOp>(funcRef.getSymbolName());
    if (!funcOp || !funcOp.getIsAsync())
      return;

    auto constOp = bindOp.getArgs()[1].getDefiningOp<ConstantOp>();
    std::optional<uint16_t> taskId;
    if (constOp)
      taskId = extractTaskIdValue(constOp);
    if (!taskId) {
      bindOp.emitError("async task '")
          << funcOp.getSymName()
          << "' bound with @bind_*_task but task ID is not comptime-known";
      result = failure();
      return;
    }

    bindings.push_back(AsyncTaskBinding{funcOp, bindOp, *taskId, kind});

    LLVM_DEBUG(llvm::dbgs() << "  async binding: " << funcOp.getSymName()
                            << " task_id=" << *taskId << "\n");
  });
  if (failed(result))
    return failure();

  // Determine whether the unit has any dispatch site. Only those sites
  // require the worker pool / trampoline.
  unit.walk([&](Operation *op) {
    if (mlir::isa<AwaitParallelOp>(op)) {
      hasAsyncDispatchFlag = true;
      return WalkResult::interrupt();
    }
    if (auto awaitOp = mlir::dyn_cast<AwaitOp>(op)) {
      if (awaitOp.getOffload()) {
        hasAsyncDispatchFlag = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "  hasAsyncDispatch="
                          << (hasAsyncDispatchFlag ? "yes" : "no") << "\n");

  return success();
}

//===----------------------------------------------------------------------===//
// Generate
//===----------------------------------------------------------------------===//

LogicalResult WorkerPoolBuilder::generate(csl::CompilationUnitOp unit,
                                          IRRewriter &rewriter,
                                          unsigned slotAlign) {
  LLVM_DEBUG(llvm::dbgs() << "=== WorkerPoolBuilder: generate ===\n");

  if (!hasAsyncDispatchFlag)
    return success();

  MLIRContext *ctx = unit.getContext();
  Location loc = unit.getLoc();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto boolTy = IntegerType::get(ctx, 1);

  // Element types.
  auto bytePtrTy = getCoroFramePtrType(ctx, slotAlign); // [*]u8 (or [*]u16)
  auto calleeFnTy = FunctionType::get(ctx, {bytePtrTy}, {});
  auto calleeFnPtrTy =
      csl::PointerType::get(calleeFnTy, /*isConst=*/true, /*isArray=*/false);
  auto u16PtrTy =
      csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/false);

  pool.poolSize = kPoolSize;
  pool.fields.clear();

  // Emit one global per field. The map is keyed by the short field name
  // ("frame", "callee", "counter_ptr", "activate"); the global symbol
  // gets the full `__coro_pool_<name>` prefix.
  auto emitField = [&](StringRef shortName, StringRef globalName, Type elemTy) {
    auto arrayTy = csl::ArrayType::get({pool.poolSize}, elemTy);
    auto sym = rewriter.getStringAttr(globalName);
    rewriter.setInsertionPointToEnd(unit.getBody());
    GlobalOp::create(rewriter, loc,
                     /*type=*/TypeAttr::get(arrayTy),
                     /*sym_name=*/sym,
                     /*initializer=*/TypedAttr(),
                     /*is_const=*/UnitAttr(),
                     /*is_export=*/UnitAttr(),
                     /*is_extern=*/UnitAttr(),
                     /*linkname=*/StringAttr(),
                     /*alignment=*/IntegerAttr());
    WorkerPoolInfo::Field f;
    f.globalName = FlatSymbolRefAttr::get(ctx, sym);
    f.arrayTy = arrayTy;
    f.elementTy = elemTy;
    pool.fields[shortName] = f;
  };
  emitField("frame", kPoolFramePrefix, bytePtrTy);
  emitField("callee", kPoolCalleePrefix, calleeFnPtrTy);
  emitField("counter_ptr", kPoolCounterPtrPrefix, u16PtrTy);
  emitField("activate", kPoolActivatePrefix, ui16Ty);

  //===---- Emit __coro_worker task --------------------------------------===//
  //
  // task __coro_worker() void {
  //     const my_id    = @get_int(@current_task());
  //     const frame    = __coro_pool_frame[my_id];
  //     const callee   = __coro_pool_callee[my_id];
  //     callee(frame);
  //     const ctr_ptr  = __coro_pool_counter_ptr[my_id];
  //     const act_task = __coro_pool_activate[my_id];
  //     @release_local_task(@current_task());
  //     const new_count = ctr_ptr.* - 1;
  //     ctr_ptr.* = new_count;
  //     if (new_count == 0) {
  //         @activate(@get_local_task_id(act_task));
  //     }
  // }

  rewriter.setInsertionPointToEnd(unit.getBody());
  auto voidFnType = FunctionType::get(ctx, {}, {});
  auto workerFunc =
      FuncOp::create(rewriter, loc, rewriter.getStringAttr(kWorkerEntryName),
                     TypeAttr::get(voidFnType),
                     /*is_task=*/rewriter.getUnitAttr(),
                     /*is_inline=*/UnitAttr(),
                     /*is_noinline=*/UnitAttr(),
                     /*is_export=*/UnitAttr(),
                     /*is_extern=*/UnitAttr(),
                     /*is_async=*/UnitAttr(),
                     /*stack_size=*/IntegerAttr(),
                     /*linkname=*/StringAttr(),
                     /*arg_attrs=*/ArrayAttr(),
                     /*res_attrs=*/ArrayAttr(),
                     /*async_result_type=*/TypeAttr(),
                     /*max_recursion_depth=*/IntegerAttr());

  Block *body = rewriter.createBlock(&workerFunc.getBody());
  rewriter.setInsertionPointToStart(body);

  auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);

  // my_task = @current_task()
  Value myTask =
      BuiltinCallOp::create(rewriter, loc, localTaskIdTy,
                            csl::BuiltinKind::CurrentTask, ValueRange{})
          .getResult();
  // my_id = @get_int(my_task)
  Value myId =
      BuiltinCallOp::create(rewriter, loc, ui16Ty, csl::BuiltinKind::GetInt,
                            ValueRange{myTask})
          .getResult();

  // frame = pool_frame[my_id]
  Value frame = LoadOp::create(
      rewriter, loc, getPoolFieldLValue(rewriter, loc, pool, myId, "frame"));
  // callee = pool_callee[my_id]
  Value callee = LoadOp::create(
      rewriter, loc, getPoolFieldLValue(rewriter, loc, pool, myId, "callee"));

  // callee(frame); on internal suspend nop.term ends the task.
  IndirectCallOp::create(rewriter, loc, /*result=*/Type(), callee,
                         ValueRange{frame});

  // Callee returned ⇒ it's done.
  Value ctrPtr = LoadOp::create(
      rewriter, loc,
      getPoolFieldLValue(rewriter, loc, pool, myId, "counter_ptr"));
  Value actTask = LoadOp::create(
      rewriter, loc, getPoolFieldLValue(rewriter, loc, pool, myId, "activate"));

  // Release MUST happen before activating the caller (so the caller can
  // re-claim this slot immediately on resume). Mirrors the fiber
  // completion handler ordering.
  BuiltinCallOp::create(rewriter, loc, Type(),
                        csl::BuiltinKind::ReleaseLocalTask, ValueRange{myTask});

  // *counter_ptr -= 1
  auto ctrLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
  Value ctrLval = DereferenceOp::create(rewriter, loc, ctrLvalTy, ctrPtr);
  Value count = LoadOp::create(rewriter, loc, ctrLval);
  Value one = ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 1));
  Value newCount = BinaryOp::create(rewriter, loc, ui16Ty,
                                    csl::BinaryOpKind::Sub, count, one);
  AssignOp::create(rewriter, loc, ctrLval, newCount);

  // if (new_count == 0) @activate(@get_local_task_id(act_task));
  Value zero = ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 0));
  Value isLast = BinaryOp::create(rewriter, loc, boolTy, csl::BinaryOpKind::Eq,
                                  newCount, zero);
  auto ifLast = IfOp::create(rewriter, loc, TypeRange{}, isLast);
  {
    Block *lastBlock = rewriter.createBlock(&ifLast.getThenRegion());
    rewriter.setInsertionPointToEnd(lastBlock);
    Value callerTask = BuiltinCallOp::create(rewriter, loc, localTaskIdTy,
                                             csl::BuiltinKind::GetLocalTaskId,
                                             ValueRange{actTask})
                           .getResult();
    BuiltinCallOp::create(rewriter, loc, Type(), csl::BuiltinKind::Activate,
                          ValueRange{callerTask});
    YieldOp::create(rewriter, loc, /*value=*/Value());
  }
  rewriter.setInsertionPointAfter(ifLast);

  ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);

  LLVM_DEBUG(llvm::dbgs() << "  Emitted __coro_worker task and SoA pool ("
                          << pool.poolSize << " entries)\n");
  return success();
}
