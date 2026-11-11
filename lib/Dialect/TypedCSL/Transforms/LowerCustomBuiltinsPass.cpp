//===- LowerCustomBuiltinsPass.cpp - Lower custom builtins to HW primitives ==//
//
// Lowers custom CSL builtins to hardware-specific sequences:
//
//   @current_task() → read TAMAP_CE_TASK_ID register (0x7E19), extract
//                     task_id bits [10:5], wrap with @get_local_task_id
//
//   @bind_local_task(fn, task_id) [runtime] → call to runtime bind function
//                     that patches instruction memory
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/ResourceBitmaps.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"

#include <set>

#define DEBUG_TYPE "lower-custom-builtins"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_LOWERCUSTOMBUILTINSPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

/// Shorthand for dialect-defined prefixes.
static StringRef rtPrefix() { return csl::CSLDialect::getFiberRuntimePrefix(); }
static StringRef allocPrefix() {
  return csl::CSLDialect::getAllocRuntimePrefix();
}

struct LowerCustomBuiltinsPass
    : public typed_csl::impl::LowerCustomBuiltinsPassBase<
          LowerCustomBuiltinsPass> {
  using LowerCustomBuiltinsPassBase::LowerCustomBuiltinsPassBase;

  /// Look up a runtime function by prefixed name.
  /// Export stripping is handled by StripRuntimeExportsPass.
  FuncOp lookupRtFunc(csl::CompilationUnitOp unit, StringRef prefix,
                      StringRef suffix) {
    auto name = (Twine(prefix) + suffix).str();
    return unit.lookupSymbol<FuncOp>(name);
  }

  void runOnOperation() override {
    auto unit = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<BuiltinCallOp> currentTaskOps;
    SmallVector<BuiltinCallOp> bindTaskOps;
    SmallVector<BuiltinCallOp> coroInitOps;
    SmallVector<BuiltinCallOp> claimOps;
    SmallVector<BuiltinCallOp> releaseOps;
    SmallVector<BuiltinCallOp> countFreeOps;
    unit.walk([&](BuiltinCallOp op) {
      auto kind = op.getBuiltin();
      if (kind == csl::BuiltinKind::CurrentTask)
        currentTaskOps.push_back(op);
      // Only lower runtime @bind_*_task (not comptime ones).
      if ((kind == csl::BuiltinKind::BindLocalTask ||
           kind == csl::BuiltinKind::BindDataTask ||
           kind == csl::BuiltinKind::BindControlTask) &&
          !op->getParentOfType<ComptimeOp>())
        bindTaskOps.push_back(op);
      // Claim builtins (comptime ones already folded to constants)
      switch (kind) {
      case csl::BuiltinKind::ClaimColor:
      case csl::BuiltinKind::ClaimLocalTask:
      case csl::BuiltinKind::ClaimDataTask:
      case csl::BuiltinKind::ClaimInputQueue:
      case csl::BuiltinKind::ClaimOutputQueue:
      case csl::BuiltinKind::ClaimControlTask:
        claimOps.push_back(op);
        break;
      case csl::BuiltinKind::ReleaseColor:
      case csl::BuiltinKind::ReleaseLocalTask:
      case csl::BuiltinKind::ReleaseDataTask:
      case csl::BuiltinKind::ReleaseInputQueue:
      case csl::BuiltinKind::ReleaseOutputQueue:
      case csl::BuiltinKind::ReleaseControlTask:
        releaseOps.push_back(op);
        break;
      case csl::BuiltinKind::CountFree:
        countFreeOps.push_back(op);
        break;
      case csl::BuiltinKind::CoroRuntimeInit:
        // Only the fiber backend has runtime state to initialize, and its
        // lowering has already replaced these with a call to the runtime's
        // init(). Anything left here is a trampoline build, where the same
        // source is expected to compile unchanged and there is nothing to do.
        coroInitOps.push_back(op);
        break;
      default:
        break;
      }
    });

    for (BuiltinCallOp op : coroInitOps)
      rewriter.eraseOp(op);

    if (!currentTaskOps.empty()) {
      auto rtGetTaskId = lookupRtFunc(unit, rtPrefix(), "get_current_task_id");
      if (!rtGetTaskId) {
        unit.emitError("runtime function 'get_current_task_id' not found — "
                       "is the runtime module imported?");
        return signalPassFailure();
      }
      for (BuiltinCallOp op : currentTaskOps)
        lowerCurrentTask(op, rewriter, rtGetTaskId);
    }

    if (!bindTaskOps.empty()) {
      auto rtBindFn = lookupRtFunc(unit, rtPrefix(), "bind_task");
      if (!rtBindFn) {
        unit.emitError("runtime function 'bind_task' not found — "
                       "is the runtime module imported?");
        return signalPassFailure();
      }
      for (BuiltinCallOp op : bindTaskOps)
        lowerBindTask(op, rewriter, rtBindFn);
    }

    // Whether the runtime allocator hands out task table entries.  On WSE-2,
    // colors and data tasks share the entries with local tasks.
    bool claimsTaskEntries = llvm::any_of(claimOps, [&](BuiltinCallOp op) {
      csl::BuiltinKind kind = op.getBuiltin();
      return kind == csl::BuiltinKind::ClaimLocalTask ||
             (targetArch == csl::TargetArch::WSE2 &&
              (kind == csl::BuiltinKind::ClaimColor ||
               kind == csl::BuiltinKind::ClaimDataTask));
    });

    for (BuiltinCallOp op : countFreeOps)
      lowerCountFreeOp(op, rewriter, unit);

    // Lower @claim_* / @release_* to runtime calls.
    if (!claimOps.empty() || !releaseOps.empty()) {
      for (BuiltinCallOp op : claimOps)
        lowerClaimOp(op, rewriter, unit);
      for (BuiltinCallOp op : releaseOps)
        lowerReleaseOp(op, rewriter, unit);
    }

    // Patch resource module globals with comptime bitmap values.
    patchResourceGlobals(unit);

    if (claimsTaskEntries)
      fenceRuntimeTaskEntries(unit, rewriter);
  }

  /// cslc places the body of a task bound at comptime into the task table
  /// when it fits into the task's entry and the following entries cslc
  /// considers unused.  Entries the runtime allocator hands out are unused
  /// for cslc, so the body of a task bound to entry k may spill into entry
  /// k + 1, where a runtime bind would overwrite it.  Binding an empty task
  /// to entry k + 1 at comptime keeps the body out of it.  The runtime
  /// allocator still hands out entry k + 1; a runtime bind replaces the
  /// empty task.
  void fenceRuntimeTaskEntries(csl::CompilationUnitOp unit,
                               IRRewriter &rewriter) {
    auto bitmaps = csl::ResourceBitmaps::loadFromUnit(unit, targetArch);

    // Task table entries bound at comptime.
    std::set<uint16_t> bound;
    unit.walk([&](BuiltinCallOp op) {
      csl::BuiltinKind kind = op.getBuiltin();
      if ((kind != csl::BuiltinKind::BindLocalTask &&
           kind != csl::BuiltinKind::BindDataTask) ||
          !op->getParentOfType<ComptimeOp>())
        return;
      auto constant = op.getArgs()[1].getDefiningOp<ConstantOp>();
      if (!constant)
        return;
      Attribute taskId = constant.getValue();
      if (auto local = mlir::dyn_cast<csl::LocalTaskIdAttr>(taskId)) {
        bound.insert(local.getValue());
      } else if (auto data = mlir::dyn_cast<csl::DataTaskIdAttr>(taskId)) {
        if (auto color = mlir::dyn_cast<csl::ColorAttr>(data.getSource()))
          bound.insert(color.getValue());
        else if (auto queue =
                     mlir::dyn_cast<csl::InputQueueAttr>(data.getSource()))
          bound.insert(queue.getValue());
      }
    });

    SmallVector<uint16_t> fences;
    for (uint16_t entry : bound) {
      uint16_t next = entry + 1;
      if (!bound.count(next) && bitmaps.isLocalTaskFree(targetArch, next))
        fences.push_back(next);
    }
    if (fences.empty())
      return;

    auto *ctx = &getContext();
    Location loc = unit.getLoc();
    auto voidFnType = FunctionType::get(ctx, {}, {});
    rewriter.setInsertionPointToEnd(unit.getBody());
    auto comptime = ComptimeOp::create(rewriter, loc);
    Block *comptimeBody = rewriter.createBlock(&comptime.getBody());
    for (uint16_t entry : fences) {
      std::string name = "__task_fence_" + std::to_string(entry);
      rewriter.setInsertionPoint(comptime);
      auto fence = FuncOp::create(
          rewriter, loc, rewriter.getStringAttr(name),
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
      rewriter.createBlock(&fence.getBody());
      ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);

      rewriter.setInsertionPointToEnd(comptimeBody);
      Value fenceRef = createFuncRef(rewriter, loc, voidFnType, name);
      Value taskId = ConstantOp::create(rewriter, loc,
                                        csl::LocalTaskIdAttr::get(ctx, entry));
      BuiltinCallOp::create(rewriter, loc, Type(),
                            csl::BuiltinKind::BindLocalTask,
                            ValueRange{fenceRef, taskId});
    }
    rewriter.setInsertionPointToEnd(comptimeBody);
    YieldOp::create(rewriter, loc, Value());
  }

  /// Lower @current_task() to:
  ///   %task_idx = call get_current_task_id()
  ///   %result = @get_local_task_id(%task_idx)
  void lowerCurrentTask(BuiltinCallOp op, IRRewriter &rewriter,
                        FuncOp rtGetTaskId) {
    Location loc = op.getLoc();
    auto ctx = op.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    rewriter.setInsertionPoint(op);

    Value taskIdx =
        CallOp::create(rewriter, loc, TypeRange{ui16Ty},
                       FlatSymbolRefAttr::get(ctx, rtGetTaskId.getSymName()),
                       ValueRange{},
                       /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr())
            .getResult();

    auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);
    Value result = BuiltinCallOp::create(rewriter, loc, localTaskIdTy,
                                         csl::BuiltinKind::GetLocalTaskId,
                                         ValueRange{taskIdx})
                       .getResult();

    rewriter.replaceOp(op, result);
  }

  /// Lower runtime @bind_*_task(fn, task_id) to a call to the
  /// architecture-specific runtime bind_task(fn_ptr: *u16, task_id: u16).
  void lowerBindTask(BuiltinCallOp op, IRRewriter &rewriter, FuncOp rtBindFn) {
    Location loc = op.getLoc();
    auto ctx = op.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    rewriter.setInsertionPoint(op);

    csl::FuncRefAttr funcRef = getFuncRef(op.getArgs()[0]);
    if (!funcRef) {
      op.emitError("expected comptime known function reference as first "
                   "argument to @bind_*_task");
      signalPassFailure();
      return;
    }

    // Build FuncAddrAttr → ConstantOp → cast to *u16.
    auto funcAddr = csl::FuncAddrAttr::get(funcRef);
    Value funcAddrVal = ConstantOp::create(rewriter, loc, funcAddr);
    auto u16PtrTy =
        csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/false);
    Value fnPtr =
        CastOp::create(rewriter, loc, u16PtrTy, funcAddrVal).getResult();

    // Cast task_id → u16 via @get_int.
    Value taskIdU16 =
        BuiltinCallOp::create(rewriter, loc, ui16Ty, csl::BuiltinKind::GetInt,
                              ValueRange{op.getArgs()[1]})
            .getResult();

    CallOp::create(rewriter, loc, TypeRange{},
                   FlatSymbolRefAttr::get(ctx, rtBindFn.getSymName()),
                   ValueRange{fnPtr, taskIdU16},
                   /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());
    rewriter.eraseOp(op);
  }

  /// Get the runtime function name suffix and the wrapping builtin for a
  /// claim builtin kind.
  struct ClaimInfo {
    StringRef rtSuffix;       // e.g. "claim_color"
    csl::BuiltinKind wrapper; // e.g. GetColor
  };
  ClaimInfo getClaimInfo(csl::BuiltinKind kind) {
    switch (kind) {
    case csl::BuiltinKind::ClaimColor:
      return {"claim_color", csl::BuiltinKind::GetColor};
    case csl::BuiltinKind::ClaimLocalTask:
      return {"claim_local_task", csl::BuiltinKind::GetLocalTaskId};
    case csl::BuiltinKind::ClaimInputQueue:
      return {"claim_input_queue", csl::BuiltinKind::GetInputQueue};
    case csl::BuiltinKind::ClaimOutputQueue:
      return {"claim_output_queue", csl::BuiltinKind::GetOutputQueue};
    case csl::BuiltinKind::ClaimControlTask:
      return {"claim_control_task", csl::BuiltinKind::GetControlTaskId};
    case csl::BuiltinKind::ClaimDataTask:
      // Data task needs arch-specific wrapping, handled in lowerClaimOp.
      return {"claim_data_task", csl::BuiltinKind::GetDataTaskId};
    default:
      llvm_unreachable("not a claim builtin");
    }
  }

  /// Lower @claim_*() to: call runtime function → wrap in typed ID.
  void lowerClaimOp(BuiltinCallOp op, IRRewriter &rewriter,
                    csl::CompilationUnitOp unit) {
    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto kind = op.getBuiltin();
    auto info = getClaimInfo(kind);

    auto rtFunc = lookupRtFunc(unit, allocPrefix(), info.rtSuffix);
    if (!rtFunc) {
      op.emitError("runtime function '")
          << info.rtSuffix << "' not found — is the resource module imported?";
      signalPassFailure();
      return;
    }

    rewriter.setInsertionPoint(op);
    Value rawId =
        CallOp::create(rewriter, loc, TypeRange{ui16Ty},
                       FlatSymbolRefAttr::get(ctx, rtFunc.getSymName()),
                       ValueRange{}, ArrayAttr(), ArrayAttr())
            .getResult();

    Value result;
    if (kind == csl::BuiltinKind::ClaimDataTask) {
      // ClaimDataTask: WSE-2 wraps via @get_color then @get_data_task_id,
      // WSE-3 wraps via @get_input_queue then @get_data_task_id.
      csl::BuiltinKind innerWrapper = (targetArch == csl::TargetArch::WSE2)
                                          ? csl::BuiltinKind::GetColor
                                          : csl::BuiltinKind::GetInputQueue;
      Type innerTy = (targetArch == csl::TargetArch::WSE2)
                         ? (Type)csl::ColorType::get(ctx)
                         : (Type)csl::InputQueueType::get(ctx);
      Value inner = BuiltinCallOp::create(rewriter, loc, innerTy, innerWrapper,
                                          ValueRange{rawId})
                        .getResult();
      result = BuiltinCallOp::create(
                   rewriter, loc, csl::DataTaskIdType::get(ctx),
                   csl::BuiltinKind::GetDataTaskId, ValueRange{inner})
                   .getResult();
    } else {
      result = BuiltinCallOp::create(rewriter, loc, op.getResult().getType(),
                                     info.wrapper, ValueRange{rawId})
                   .getResult();
    }

    rewriter.replaceOp(op, result);
  }

  /// Lower @release_*(arg) to: @get_int(arg) → call runtime release function.
  void lowerReleaseOp(BuiltinCallOp op, IRRewriter &rewriter,
                      csl::CompilationUnitOp unit) {
    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    // Map release kind to runtime function suffix.
    StringRef rtSuffix;
    switch (op.getBuiltin()) {
    case csl::BuiltinKind::ReleaseColor:
      rtSuffix = "release_color";
      break;
    case csl::BuiltinKind::ReleaseLocalTask:
      rtSuffix = "release_local_task";
      break;
    case csl::BuiltinKind::ReleaseDataTask:
      rtSuffix = "release_data_task";
      break;
    case csl::BuiltinKind::ReleaseInputQueue:
      rtSuffix = "release_input_queue";
      break;
    case csl::BuiltinKind::ReleaseOutputQueue:
      rtSuffix = "release_output_queue";
      break;
    case csl::BuiltinKind::ReleaseControlTask:
      rtSuffix = "release_control_task";
      break;
    default:
      llvm_unreachable("not a release builtin");
    }

    auto rtFunc = lookupRtFunc(unit, allocPrefix(), rtSuffix);
    if (!rtFunc) {
      op.emitError("runtime function '")
          << rtSuffix << "' not found — is the resource module imported?";
      signalPassFailure();
      return;
    }

    rewriter.setInsertionPoint(op);
    Value rawId =
        BuiltinCallOp::create(rewriter, loc, ui16Ty, csl::BuiltinKind::GetInt,
                              ValueRange{op.getArgs()[0]})
            .getResult();
    CallOp::create(rewriter, loc, TypeRange{},
                   FlatSymbolRefAttr::get(ctx, rtFunc.getSymName()),
                   ValueRange{rawId}, ArrayAttr(), ArrayAttr());
    rewriter.eraseOp(op);
  }

  /// Lower a runtime @count_free(T) to a call of the allocator's count of
  /// the resource kind T.
  void lowerCountFreeOp(BuiltinCallOp op, IRRewriter &rewriter,
                        csl::CompilationUnitOp unit) {
    auto *ctx = op.getContext();
    auto typeOp = op.getArgs()[0].getDefiningOp<ConstantOp>();
    auto typeAttr =
        typeOp ? mlir::dyn_cast<csl::TypeAttr>(typeOp.getValue()) : nullptr;
    StringRef rtSuffix;
    if (typeAttr) {
      Type idType = typeAttr.getValue();
      if (mlir::isa<csl::ColorType>(idType))
        rtSuffix = "count_free_colors";
      else if (mlir::isa<csl::LocalTaskIdType>(idType))
        rtSuffix = "count_free_local_tasks";
      else if (mlir::isa<csl::DataTaskIdType>(idType))
        rtSuffix = "count_free_data_tasks";
      else if (mlir::isa<csl::InputQueueType>(idType))
        rtSuffix = "count_free_input_queues";
      else if (mlir::isa<csl::OutputQueueType>(idType))
        rtSuffix = "count_free_output_queues";
      else if (mlir::isa<csl::ControlTaskIdType>(idType))
        rtSuffix = "count_free_control_tasks";
    }
    if (rtSuffix.empty()) {
      op.emitError("@count_free expects a comptime-known resource ID type "
                   "(internal error)");
      signalPassFailure();
      return;
    }

    auto rtFunc = lookupRtFunc(unit, allocPrefix(), rtSuffix);
    if (!rtFunc) {
      op.emitError("runtime function '")
          << rtSuffix << "' not found — is the resource module imported?";
      signalPassFailure();
      return;
    }

    rewriter.setInsertionPoint(op);
    auto call = CallOp::create(
        rewriter, op.getLoc(), TypeRange{op.getResult().getType()},
        FlatSymbolRefAttr::get(ctx, rtFunc.getSymName()), ValueRange{},
        ArrayAttr(), ArrayAttr());
    rewriter.replaceOp(op, call.getResults());
  }

  /// Patch resource module globals with comptime-claimed bitmap values.
  void patchResourceGlobals(csl::CompilationUnitOp unit) {
    auto bitmaps = csl::ResourceBitmaps::loadFromUnit(unit, targetArch);

    auto *ctx = &getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    // Map bitmap fields to resource module global names.
    static constexpr StringLiteral globalNames[] = {
        "color_free_lo",      "color_free_hi",     "task_entry_free_lo",
        "task_entry_free_hi", "input_queue_free",  "output_queue_free",
        "ctrl_task_free_lo",  "ctrl_task_free_hi",
    };
    const uint16_t values[] = {
        bitmaps.colorFreeLo,     bitmaps.colorFreeHi,
        bitmaps.taskEntryFreeLo, bitmaps.taskEntryFreeHi,
        bitmaps.inputQueueFree,  bitmaps.outputQueueFree,
        bitmaps.ctrlTaskFreeLo,  bitmaps.ctrlTaskFreeHi,
    };

    for (size_t i = 0; i < std::size(globalNames); ++i) {
      auto globalName = (Twine(allocPrefix()) + globalNames[i]).str();
      if (auto globalOp = unit.lookupSymbol<GlobalOp>(globalName))
        globalOp.setInitializerAttr(IntegerAttr::get(ui16Ty, values[i]));
    }

    // Note: we intentionally do NOT remove the bitmap attribute here.
    // ProcessTileCodePass reads it later to propagate layout-level claims
    // to PE programs.
  }
};

} // namespace
