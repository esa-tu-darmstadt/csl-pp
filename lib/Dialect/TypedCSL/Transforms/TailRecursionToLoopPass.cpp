//===- TailRecursionToLoopPass.cpp - Turn self tail calls into a loop -----===//
//
// Rewrites a function that calls itself in tail position into a loop:
//
//   fn f(a: T, b: U) R {          fn f(a0: T, b0: U) R {
//     ...                           var a = a0; var b = b0;
//     return f(x, y);               while (true) {
//   }                                 ...
//                                     var t0 = x; var t1 = y;
//                                     a = t0; b = t1;
//                                     continue;
//                                   }
//                                 }
//
// CSL has no labeled `continue`, so a tail call inside another loop of the
// function stays a call.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/Analysis/IndirectCallTargets.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_TAILRECURSIONTOLOOPPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

/// A call of a function to itself whose result is returned right away.
struct TailCall {
  CallOp call;
  ReturnOp ret;
};

/// The tail call of `func` to itself that `ret` returns from, if any.  A
/// call inside another loop of `func` does not count: an unlabeled
/// `continue` there would continue that loop.
std::optional<TailCall> matchTailCall(FuncOp func, ReturnOp ret) {
  auto call = mlir::dyn_cast_or_null<CallOp>(ret->getPrevNode());
  if (!call || call.getCalleeAttr().getRootReference() != func.getSymNameAttr())
    return std::nullopt;
  for (Operation *parent = ret->getParentOp(); parent != func;
       parent = parent->getParentOp())
    if (mlir::isa<WhileOp, ForOp>(parent))
      return std::nullopt;
  if (Value returned = ret.getOperand()) {
    if (call->getNumResults() != 1 || returned != call->getResult(0) ||
        !returned.hasOneUse())
      return std::nullopt;
  } else if (call->getNumResults() != 0) {
    return std::nullopt;
  }
  return TailCall{call, ret};
}

/// Whether the tail calls of `func` may be turned into a loop.
bool canConvertToLoop(FuncOp func) {
  if (func.isExternal() || func.getIsAsync() || func.getIsTask())
    return false;
  // Every iteration of the loop reuses the storage of the parameters and
  // locals, while every recursive call has its own.  An address could tell
  // them apart.
  return !func.walk([](AddressOfOp) { return WalkResult::interrupt(); })
              .wasInterrupted();
}

/// Rewrite the body of `func` into a loop that each of `tailCalls`
/// continues.
void convertToLoop(FuncOp func, ArrayRef<TailCall> tailCalls,
                   IRRewriter &rewriter) {
  Location loc = func.getLoc();
  Block &entry = func.getBody().front();

  // Terminators that leave the function leave one more region once the body
  // is the body of the loop.
  SmallVector<Operation *> exits;
  collectNestedRegionExits(func, exits);

  auto *loopBody = new Block();
  loopBody->getOperations().splice(loopBody->end(), entry.getOperations());

  // A mutable variable per parameter.  The body reads it through a const
  // view, so the uses keep their types.  Only the tail calls assign it,
  // right before they continue the loop.
  rewriter.setInsertionPointToStart(&entry);
  SmallVector<Value> paramVars;
  SmallVector<Value> paramViews;
  for (BlockArgument param : entry.getArguments()) {
    auto paramType = mlir::cast<csl::LValueType>(param.getType());
    auto initial = LoadOp::create(rewriter, loc, param);
    Value var = VariableOp::create(
        rewriter, loc,
        csl::LValueType::get(paramType.getValueType(), /*isConst=*/false),
        "__tailrec_param" + std::to_string(param.getArgNumber()), initial,
        /*is_const=*/false);
    Value view = ConstCastOp::create(rewriter, loc, paramType, var);
    param.replaceAllUsesExcept(view, initial);
    paramVars.push_back(var);
    paramViews.push_back(view);
  }

  auto loop = WhileOp::create(rewriter, loc, /*label=*/StringAttr());

  // The loop only ends by returning, but cslc requires a function that
  // returns a value to end with a return statement.  CSL has no
  // `unreachable`, so return an uninitialized value.
  rewriter.setInsertionPointAfter(loop);
  ArrayRef<Type> results = func.getFunctionType().getResults();
  if (results.empty() || mlir::isa<csl::VoidType>(results.front())) {
    ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);
  } else {
    Value never = VariableOp::create(
        rewriter, loc, csl::LValueType::get(results.front(), /*isConst=*/false),
        "__tailrec_unreachable", /*initializer=*/Value(), /*is_const=*/false);
    ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1,
                     LoadOp::create(rewriter, loc, never));
  }

  Block *cond = rewriter.createBlock(&loop.getCondRegion());
  rewriter.setInsertionPointToEnd(cond);
  Value always = ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
  CondYieldOp::create(rewriter, loc, always);
  loop.getBody().push_back(loopBody);

  for (Operation *exit : exits) {
    auto breaking = mlir::cast<BreakingTerminatorOpInterface>(exit);
    breaking.setNumBreakingRegions(breaking.getNumBreakingRegions() + 1);
  }

  for (TailCall tailCall : tailCalls) {
    Location callLoc = tailCall.call.getLoc();
    rewriter.setInsertionPoint(tailCall.ret);

    // Parameters passed on unchanged need no assignment.
    SmallVector<unsigned> changed;
    for (auto [index, arg] : llvm::enumerate(tailCall.call.getArgOperands())) {
      auto load = arg.getDefiningOp<LoadOp>();
      if (!load || load.getLvalue() != paramViews[index])
        changed.push_back(index);
    }

    // An argument may read a parameter that an earlier assignment already
    // changed, so with several assignments all arguments are evaluated into
    // temporaries first.
    SmallVector<Value> newValues;
    for (unsigned index : changed) {
      Value arg = tailCall.call.getArgOperands()[index];
      if (changed.size() > 1) {
        Value temp = VariableOp::create(
            rewriter, callLoc,
            csl::LValueType::get(arg.getType(), /*isConst=*/false),
            "__tailrec_arg" + std::to_string(index), arg, /*is_const=*/false);
        arg = LoadOp::create(rewriter, callLoc, temp);
      }
      newValues.push_back(arg);
    }
    for (auto [index, value] : llvm::zip_equal(changed, newValues))
      AssignOp::create(rewriter, callLoc, paramVars[index], value);

    // The return now leaves the loop body, too; the continue stays in it.
    ContinueOp::create(rewriter, callLoc,
                       tailCall.ret.getNumBreakingRegions() - 1,
                       /*targetLabel=*/StringAttr());
    rewriter.eraseOp(tailCall.ret);
    rewriter.eraseOp(tailCall.call);
  }
}

struct TailRecursionToLoopPass
    : public typed_csl::impl::TailRecursionToLoopPassBase<
          TailRecursionToLoopPass> {
  using TailRecursionToLoopPassBase::TailRecursionToLoopPassBase;

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<FuncOp> funcs;
    unit.walk([&](FuncOp func) { funcs.push_back(func); });

    for (FuncOp func : funcs) {
      if (!canConvertToLoop(func))
        continue;
      SmallVector<TailCall> tailCalls;
      func.walk([&](ReturnOp ret) {
        if (std::optional<TailCall> tailCall = matchTailCall(func, ret))
          tailCalls.push_back(*tailCall);
      });
      if (tailCalls.empty())
        continue;
      convertToLoop(func, tailCalls, rewriter);

      if (func.getMaxRecursionDepth().has_value() && !isRecursive(func)) {
        func.emitWarning("'max_recursion_depth' has no effect because the "
                         "recursive calls of '")
            << func.getSymName()
            << "' are tail calls, which are compiled into a loop; remove it";
        func.removeMaxRecursionDepthAttr();
      }
    }
  }

  /// Whether `func` can still reach itself through direct or indirect calls.
  bool isRecursive(FuncOp func) {
    const IndirectCallTargets &indirectTargets =
        getAnalysis<IndirectCallTargets>();
    csl::CompilationUnitOp unit = getOperation();
    DenseSet<Operation *> visited;
    SmallVector<FuncOp> worklist{func};
    while (!worklist.empty()) {
      FuncOp caller = worklist.pop_back_val();
      bool found = false;
      auto visit = [&](FuncOp callee) {
        if (!callee)
          return;
        if (callee == func)
          found = true;
        else if (visited.insert(callee).second)
          worklist.push_back(callee);
      };
      caller.walk([&](Operation *op) {
        if (auto call = mlir::dyn_cast<CallOp>(op))
          visit(unit.lookupSymbol<FuncOp>(call.getCalleeAttr()));
        else if (auto icall = mlir::dyn_cast<IndirectCallOp>(op))
          for (FuncOp callee : indirectTargets.getPossibleCallees(icall))
            visit(callee);
      });
      if (found)
        return true;
    }
    return false;
  }
};

} // namespace
