//===- CountTasksPass.cpp - Static WSE task slot accounting --------------===//
//
// Walks a PE program and reports how many task slots the program needs,
// split by local / control / data. See the pass description in Passes.td
// for the counting model.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_COUNTTASKSPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

/// Sentinel returned by computeFootprint when analysis fails (recursion,
/// missing callee body, indirect call, etc.). Callers should propagate it
/// up and emit a single top-level warning.
constexpr int kFootprintUnknown = -1;

struct CountTasksPass
    : public typed_csl::impl::CountTasksPassBase<CountTasksPass> {
  using CountTasksPassBase::CountTasksPassBase;

  /// True if `op` is lexically nested inside a `typed_csl.comptime` region.
  static bool isInsideComptime(Operation *op) {
    for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
      if (isa<ComptimeOp>(p))
        return true;
    }
    return false;
  }

  static bool isBindBuiltin(csl::BuiltinKind k) {
    return k == csl::BuiltinKind::BindLocalTask ||
           k == csl::BuiltinKind::BindControlTask ||
           k == csl::BuiltinKind::BindDataTask;
  }

  /// Resolve a `bind_*_task` call's first argument to its FuncOp callee.
  /// Returns nullptr if not a direct function reference.
  static FuncOp getBindCallee(csl::CompilationUnitOp unit, BuiltinCallOp op) {
    if (op.getArgs().empty())
      return nullptr;
    csl::FuncRefAttr funcRef = getFuncRef(op.getArgs()[0]);
    if (!funcRef)
      return nullptr;
    return unit.lookupSymbol<FuncOp>(funcRef.getSymbolName());
  }

  /// Resolve a future operand (the `await`/`await_parallel` operand) to the
  /// FuncOp that produced it. Returns nullptr if the operand is not a direct
  /// `typed_csl.call`.
  static FuncOp resolveFutureCallee(csl::CompilationUnitOp unit, Value future) {
    auto callOp = future.getDefiningOp<CallOp>();
    if (!callOp)
      return nullptr;
    return unit.lookupSymbol<FuncOp>(callOp.getCallee());
  }

  /// Recursive memoised computation of footprint(f). See pass description
  /// for the model. Mutates `memo` and `inProgress`; emits diagnostics for
  /// recursion / non-direct calls / missing bodies.
  int computeFootprint(csl::CompilationUnitOp unit, FuncOp f,
                       llvm::DenseMap<Operation *, int> &memo,
                       llvm::DenseSet<Operation *> &inProgress,
                       bool &sawError) {
    auto it = memo.find(f.getOperation());
    if (it != memo.end())
      return it->second;

    if (!inProgress.insert(f.getOperation()).second) {
      f.emitWarning() << "recursive async call graph detected; task count "
                         "analysis cannot proceed for this function";
      sawError = true;
      memo[f.getOperation()] = kFootprintUnknown;
      return kFootprintUnknown;
    }

    int maxFootprint = 0;

    if (f.isExternal()) {
      // External (no body): can't analyse. If it's async, that's a problem;
      // otherwise the footprint is genuinely zero.
      if (f.getIsAsync()) {
        f.emitWarning() << "async function '" << f.getSymName()
                        << "' has no body; assuming footprint = 0";
        sawError = true;
      }
      inProgress.erase(f.getOperation());
      memo[f.getOperation()] = 0;
      return 0;
    }

    WalkResult wr = f.walk([&](Operation *op) {
      int slots = 0;
      if (auto awaitOp = dyn_cast<AwaitOp>(op)) {
        FuncOp callee = resolveFutureCallee(unit, awaitOp.getFuture());
        if (!callee) {
          awaitOp.emitWarning()
              << "@await operand is not a direct typed_csl.call; cannot "
                 "determine callee for task-count analysis";
          sawError = true;
          return WalkResult::advance();
        }
        int cf = computeFootprint(unit, callee, memo, inProgress, sawError);
        if (cf == kFootprintUnknown) {
          slots = kFootprintUnknown;
        } else {
          slots = awaitOp.getOffload() ? 1 + cf : cf;
        }
      } else if (auto apOp = dyn_cast<AwaitParallelOp>(op)) {
        int total = 0;
        bool unknown = false;
        for (auto [i, fut] : llvm::enumerate(apOp.getFutures())) {
          FuncOp callee = resolveFutureCallee(unit, fut);
          if (!callee) {
            apOp.emitWarning()
                << "@await_parallel operand #" << i
                << " is not a direct typed_csl.call; cannot determine "
                   "callee for task-count analysis";
            sawError = true;
            unknown = true;
            break;
          }
          int cf = computeFootprint(unit, callee, memo, inProgress, sawError);
          if (cf == kFootprintUnknown) {
            unknown = true;
            break;
          }
          // First callee runs on the caller's slot, so it doesn't claim
          // an extra slot — only its own descendants do.
          total += (i == 0) ? cf : (1 + cf);
        }
        slots = unknown ? kFootprintUnknown : total;
      } else {
        return WalkResult::advance();
      }

      if (slots == kFootprintUnknown) {
        maxFootprint = kFootprintUnknown;
        return WalkResult::interrupt();
      }
      maxFootprint = std::max(maxFootprint, slots);
      return WalkResult::advance();
    });
    (void)wr;

    inProgress.erase(f.getOperation());
    memo[f.getOperation()] = maxFootprint;
    return maxFootprint;
  }

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();

    // -------- Step 1: count comptime / runtime @bind_*_task calls --------

    int comptimeLocal = 0;
    int comptimeControl = 0;
    int comptimeData = 0;
    SmallVector<BuiltinCallOp> runtimeBinds;
    // Comptime binds whose callee is an async FuncOp — used in step 3.
    SmallVector<std::pair<BuiltinCallOp, FuncOp>> comptimeAsyncBinds;

    unit.walk(
        [&](BuiltinCallOp op) {
          csl::BuiltinKind kind = op.getBuiltin();
          if (!isBindBuiltin(kind))
            return;

          if (!isInsideComptime(op)) {
            runtimeBinds.push_back(op);
            return;
          }

          switch (kind) {
          case csl::BuiltinKind::BindLocalTask:
            ++comptimeLocal;
            break;
          case csl::BuiltinKind::BindControlTask:
            ++comptimeControl;
            break;
          case csl::BuiltinKind::BindDataTask:
            ++comptimeData;
            break;
          default:
            break;
          }

          if (FuncOp callee = getBindCallee(unit, op)) {
            if (callee.getIsAsync())
              comptimeAsyncBinds.emplace_back(op, callee);
          } else {
            op.emitWarning()
                << "@bind_*_task target is not a direct function reference; "
                   "cannot include in task-count analysis";
          }
        });

    // -------- Step 2/3: compute footprint and add to local count --------

    llvm::DenseMap<Operation *, int> memo;
    llvm::DenseSet<Operation *> inProgress;
    bool footprintError = false;
    int asyncFootprint = 0;
    bool asyncFootprintUnknown = false;

    for (auto &[bindOp, callee] : comptimeAsyncBinds) {
      int fp = computeFootprint(unit, callee, memo, inProgress, footprintError);
      if (fp == kFootprintUnknown) {
        asyncFootprintUnknown = true;
        continue;
      }
      asyncFootprint += fp;
    }

    int totalLocal = comptimeLocal + asyncFootprint;
    int totalControl = comptimeControl;
    int totalData = comptimeData;

    // -------- Step 4: emit the summary report ----------------------------

    auto &os = llvm::errs();
    os << "=== count-tasks: task slot accounting ===\n";
    os << "  local tasks   : " << totalLocal << "   (" << comptimeLocal
       << " comptime-bound, " << asyncFootprint << " from async footprint)\n";
    os << "  control tasks : " << totalControl << "\n";
    os << "  data tasks    : " << totalData << "\n";
    os << "  total         : " << (totalLocal + totalControl + totalData)
       << "\n";

    if (!runtimeBinds.empty()) {
      os << "WARNING: " << runtimeBinds.size()
         << " runtime @bind_*_task call(s) detected outside a comptime "
            "block; the reported task count may be too low.\n";
      for (BuiltinCallOp op : runtimeBinds)
        op.emitWarning()
            << "runtime @bind_*_task is not counted by count-tasks";
    }
    if (asyncFootprintUnknown)
      os << "WARNING: one or more async footprints could not be computed "
            "(see diagnostics above); the local task count is a lower "
            "bound only.\n";
    if (footprintError)
      os << "WARNING: analysis encountered issues (recursion, missing "
            "bodies, or indirect calls); see diagnostics above.\n";
    os << "==========================================\n";
  }
};

} // namespace
