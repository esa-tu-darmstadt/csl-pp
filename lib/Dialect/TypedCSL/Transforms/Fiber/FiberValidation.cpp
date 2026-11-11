//===- FiberValidation.cpp - Reachability + validation impl --------------===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberValidation.h"

#include "csl-mlir/Dialect/CSL/Utils/ResourceBitmaps.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;

namespace mlir::typed_csl {

/// Whether `call` produces a future that an `@await_parallel` with several
/// callees consumes.  Each such callee runs in a stack region of fixed size.
static bool isMultiCalleeParallelCall(CallOp call) {
  for (Value result : call->getResults()) {
    for (Operation *user : result.getUsers()) {
      auto awaitParallel = mlir::dyn_cast<AwaitParallelOp>(user);
      if (awaitParallel && awaitParallel.getFutures().size() > 1)
        return true;
    }
  }
  return false;
}

namespace {

/// A call edge of the call graph.
struct CallEdge {
  /// The direct or indirect call op.
  Operation *call;
  StringAttr callee;
  /// Whether the callee runs as a callee of a multi-callee `@await_parallel`.
  bool isParallel;
};

} // namespace

/// Invoke `callback` for every call edge in `fn`: the callee of each direct
/// `typed_csl.call` and each possible callee of each indirect call.
static WalkResult
walkCallEdges(FuncOp fn, const IndirectCallTargets &indirectTargets,
              function_ref<WalkResult(const CallEdge &edge)> callback) {
  return fn.walk([&](Operation *op) -> WalkResult {
    if (auto call = mlir::dyn_cast<CallOp>(op)) {
      auto sym = call.getCalleeAttr();
      if (!sym)
        return WalkResult::advance();
      return callback(CallEdge{call, sym.getRootReference(),
                               isMultiCalleeParallelCall(call)});
    }
    if (auto icall = mlir::dyn_cast<IndirectCallOp>(op)) {
      for (FuncOp callee : indirectTargets.getPossibleCallees(icall)) {
        if (callback(CallEdge{icall, callee.getSymNameAttr(),
                              /*isParallel=*/false})
                .wasInterrupted())
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
}

FiberReachableFuncs
computeFiberReachable(csl::CompilationUnitOp unit,
                      const IndirectCallTargets &indirectTargets) {
  FiberReachableFuncs result;
  SmallVector<StringAttr> worklist;

  // Seed: every `async` FuncOp.  SymbolDCE runs before stage 2, so any
  // surviving async FuncOp is genuinely live (bound, exported, or
  // referenced from another live function).
  unit.walk([&](FuncOp fn) {
    if (!fn.getIsAsync())
      return;
    if (result.funcs.insert(fn.getSymNameAttr()).second)
      worklist.push_back(fn.getSymNameAttr());
  });

  // Frontier: every direct `typed_csl.call` callee and every possible
  // callee of an indirect call, transitively.  A CallOp producing a
  // `future` type is the same op from the reachability perspective — the
  // consumed-by-await edge re-enters via the produced future's callee
  // symbol, which we already grabbed from the CallOp.
  while (!worklist.empty()) {
    StringAttr name = worklist.pop_back_val();
    auto fn = unit.lookupSymbol<FuncOp>(name);
    if (!fn)
      continue; // unknown symbol — validation pass will diagnose
    walkCallEdges(fn, indirectTargets, [&](const CallEdge &edge) {
      if (result.funcs.insert(edge.callee).second)
        worklist.push_back(edge.callee);
      return WalkResult::advance();
    });
  }
  return result;
}

namespace {

/// The call graph among the reachable functions with their strongly
/// connected components.  Contains every call edge, including the edges out
/// of functions with a `stack_size(N)` override.
class ReachableCallGraph {
public:
  ReachableCallGraph(csl::CompilationUnitOp unit,
                     const FiberReachableFuncs &reachable,
                     const IndirectCallTargets &indirectTargets) {
    // Source order, so diagnostics are stable across runs.
    unit.walk([&](FuncOp fn) {
      StringAttr name = fn.getSymNameAttr();
      if (!reachable.contains(name))
        return;
      functions.push_back(name);
      SmallVector<CallEdge> &fnEdges = edges[name];
      walkCallEdges(fn, indirectTargets, [&](const CallEdge &edge) {
        fnEdges.push_back(edge);
        return WalkResult::advance();
      });
    });
    for (StringAttr name : functions)
      if (!index.contains(name))
        strongConnect(name);
  }

  /// The reachable functions in source order.
  ArrayRef<StringAttr> getFunctions() const { return functions; }

  /// The call edges out of `fn`, in source order.
  ArrayRef<CallEdge> getEdges(StringAttr fn) const {
    auto it = edges.find(fn);
    if (it == edges.end())
      return {};
    return it->second;
  }

  bool inSameComponent(StringAttr a, StringAttr b) const {
    auto itA = component.find(a);
    auto itB = component.find(b);
    return itA != component.end() && itB != component.end() &&
           itA->second == itB->second;
  }

  /// The strongly connected component of `fn`, if it is a reachable
  /// function.
  std::optional<unsigned> getComponent(StringAttr fn) const {
    auto it = component.find(fn);
    if (it == component.end())
      return std::nullopt;
    return it->second;
  }

  /// Whether `fn` lies on a cycle of the call graph.
  bool isRecursive(StringAttr fn) const {
    auto it = component.find(fn);
    if (it == component.end())
      return false;
    if (componentSizes[it->second] > 1)
      return true;
    return llvm::any_of(getEdges(fn), [&](const CallEdge &edge) {
      return edge.callee == fn;
    });
  }

  /// The functions on a shortest path of call edges from `from` to `to`
  /// within one component, including both ends.
  SmallVector<StringAttr> findPath(StringAttr from, StringAttr to) const {
    DenseMap<StringAttr, StringAttr> predecessor;
    SmallVector<StringAttr> queue{from};
    predecessor[from] = StringAttr();
    for (size_t i = 0; i < queue.size() && !predecessor.contains(to); ++i) {
      for (const CallEdge &edge : getEdges(queue[i])) {
        if (!inSameComponent(edge.callee, from) ||
            !predecessor.try_emplace(edge.callee, queue[i]).second)
          continue;
        queue.push_back(edge.callee);
      }
    }
    SmallVector<StringAttr> path;
    for (StringAttr node = to; node; node = predecessor.lookup(node))
      path.push_back(node);
    std::reverse(path.begin(), path.end());
    return path;
  }

private:
  /// Tarjan's algorithm.
  void strongConnect(StringAttr name) {
    index[name] = lowLink[name] = nextIndex++;
    tarjanStack.push_back(name);
    onStack.insert(name);
    for (const CallEdge &edge : getEdges(name)) {
      if (!edges.contains(edge.callee))
        continue; // not a reachable function with a body
      if (!index.contains(edge.callee)) {
        strongConnect(edge.callee);
        lowLink[name] = std::min(lowLink[name], lowLink[edge.callee]);
      } else if (onStack.contains(edge.callee)) {
        lowLink[name] = std::min(lowLink[name], index[edge.callee]);
      }
    }
    if (lowLink[name] != index[name])
      return;
    unsigned id = componentSizes.size();
    componentSizes.push_back(0);
    StringAttr member;
    do {
      member = tarjanStack.pop_back_val();
      onStack.erase(member);
      component[member] = id;
      ++componentSizes[id];
    } while (member != name);
  }

  SmallVector<StringAttr> functions;
  DenseMap<StringAttr, SmallVector<CallEdge>> edges;
  DenseMap<StringAttr, unsigned> component;
  SmallVector<unsigned> componentSizes;

  DenseMap<StringAttr, unsigned> index;
  DenseMap<StringAttr, unsigned> lowLink;
  SmallVector<StringAttr> tarjanStack;
  DenseSet<StringAttr> onStack;
  unsigned nextIndex = 0;
};

/// Join function names with " -> ".
std::string formatChain(ArrayRef<StringAttr> names) {
  std::string chain;
  for (StringAttr name : names) {
    if (!chain.empty())
      chain += " -> ";
    chain += name.str();
  }
  return chain;
}

/// Reject recursion through a callee of a multi-callee `@await_parallel`.
/// Such a callee runs in a stack region of fixed size, which would have to
/// hold its own recursive calls.  Returns true if an error was emitted.
bool rejectParallelRecursion(const ReachableCallGraph &graph) {
  bool hadError = false;
  for (StringAttr caller : graph.getFunctions()) {
    for (const CallEdge &edge : graph.getEdges(caller)) {
      if (!edge.isParallel || !graph.inSameComponent(caller, edge.callee))
        continue;
      SmallVector<StringAttr> chain{caller};
      llvm::append_range(chain, graph.findPath(edge.callee, caller));
      edge.call->emitError("recursive calls ")
          << formatChain(chain)
          << " run through @await_parallel: each callee of an "
             "@await_parallel with several callees gets a stack region of "
             "fixed size, which cannot hold a recursion; call '"
          << edge.callee.getValue()
          << "' outside of @await_parallel, directly or with @await";
      hadError = true;
    }
  }
  return hadError;
}

/// Color-marked DFS through the call graph to find cycles without any
/// `max_recursion_depth(N)` or `stack_size(N)` annotation along the cycle.
/// Mirrors the resolver's behavior: at a node with `stack_size(N)`, we stop
/// walking the body (the override is opaque), and a node with
/// `max_recursion_depth(N)` bounds every cycle through it.  Edges into
/// callees of a multi-callee `@await_parallel` are skipped; recursion through
/// them is rejected by `rejectParallelRecursion`.
struct CycleDetector {
  enum Color : uint8_t { White = 0, Gray = 1, Black = 2 };

  csl::CompilationUnitOp unit;
  const FiberReachableFuncs &reachable;
  const IndirectCallTargets &indirectTargets;
  llvm::DenseMap<StringAttr, Color> color;
  SmallVector<StringAttr> stack;
  bool hadError = false;

  CycleDetector(csl::CompilationUnitOp unit,
                const FiberReachableFuncs &reachable,
                const IndirectCallTargets &indirectTargets)
      : unit(unit), reachable(reachable), indirectTargets(indirectTargets) {}

  void run() {
    // Walk in source order so the cycle representation in diagnostics is
    // stable across runs (DenseSet iteration is hash-order).
    unit.walk([&](FuncOp fn) {
      StringAttr name = fn.getSymNameAttr();
      if (reachable.contains(name) && color[name] == White)
        dfs(name);
    });
  }

  void dfs(StringAttr name) {
    auto fn = unit.lookupSymbol<FuncOp>(name);
    if (!fn) {
      color[name] = Black;
      return;
    }
    if (fn.getStackSize().has_value() ||
        fn.getMaxRecursionDepth().has_value() || fn.isExternal()) {
      // `stack_size(N)` makes the body opaque (resolver does the same), and
      // `max_recursion_depth(N)` bounds every cycle through this function.
      // Extern fns have no body to walk; they're flagged separately by
      // validateFiberSources, no point counting them here.
      color[name] = Black;
      return;
    }

    color[name] = Gray;
    stack.push_back(name);

    walkCallEdges(fn, indirectTargets,
                  [&](const CallEdge &edge) -> WalkResult {
                    if (edge.isParallel)
                      return WalkResult::advance();
                    Color &c = color[edge.callee];
                    if (c == Gray) {
                      reportCycle(edge);
                      return WalkResult::interrupt();
                    }
                    if (c == White)
                      dfs(edge.callee);
                    return WalkResult::advance();
                  });

    stack.pop_back();
    color[name] = Black;
  }

  /// Report the cycle closed by the back-edge `edge`.
  void reportCycle(const CallEdge &edge) {
    SmallVector<StringAttr> chain(llvm::find(stack, edge.callee), stack.end());
    chain.push_back(edge.callee);
    auto calleeFn = unit.lookupSymbol<FuncOp>(edge.callee);
    std::string keyword = calleeFn && calleeFn.getIsTask() ? "task" : "fn";
    if (calleeFn && calleeFn.getIsAsync())
      keyword = "async " + keyword;
    edge.call->emitError("recursive calls ")
        << formatChain(chain)
        << " are reachable from an async task, so the task's stack size "
           "cannot be computed; add `max_recursion_depth(N)` to '"
        << edge.callee.getValue()
        << "' (or to another function of the cycle), where N is the maximum "
           "number of its calls that are active at the same time, including "
           "the outermost one, e.g. `"
        << keyword << " " << edge.callee.getValue()
        << "(...) max_recursion_depth(16) <return type>`.  Alternatively, "
           "`stack_size(N)` declares the function's maximum stack use in "
           "bytes, including all recursive calls";
    hadError = true;
  }
};

} // namespace

LogicalResult validateFiberSources(csl::CompilationUnitOp unit,
                                   const FiberReachableFuncs &reachable,
                                   const IndirectCallTargets &indirectTargets) {
  bool hadError = false;

  // Iterate in source order so diagnostics fire in a stable order across
  // runs (DenseSet iteration is hash-order).
  unit.walk([&](FuncOp fn) {
    if (!reachable.contains(fn.getSymNameAttr()))
      return;

    // `stack_size(N)` is the universal escape hatch — vouches for the
    // function's max stack reach, body opaque from the resolver's view.
    if (fn.getStackSize().has_value())
      return;

    // Extern: no body to walk.  `.stack_sizes` only reveals the extern's
    // own frame at link time and tells us nothing about what it calls.
    if (fn.isExternal()) {
      fn.emitError("extern function '")
          << fn.getSymName()
          << "' is reachable from an async task but has no `stack_size(N)`.  "
             "The stack reach of an extern function cannot be computed; "
             "add `stack_size(N)` to its declaration, where N is its maximum "
             "stack reach in bytes including everything it calls.";
      hadError = true;
      return;
    }

    // Indirect calls are charged with their possible callees.  A call
    // without any cannot call anything that needs stack, but it is most
    // likely a mistake.
    fn.walk([&](IndirectCallOp icall) {
      if (indirectTargets.getPossibleCallees(icall).empty())
        icall.emitWarning("indirect call has no possible callee: no function "
                          "of type '")
            << csl::prettyPrintType(icall.getCalleeType())
            << "' has its address taken";
    });
  });

  // Recursion detection — separate pass so cycle diagnostics aren't
  // interleaved with the per-function rejections above.
  ReachableCallGraph graph(unit, reachable, indirectTargets);
  if (rejectParallelRecursion(graph))
    hadError = true;

  CycleDetector cycles(unit, reachable, indirectTargets);
  cycles.run();
  if (cycles.hadError)
    hadError = true;

  // `max_recursion_depth(N)` only bounds recursion.  Together with
  // `stack_size(N)` it is ignored, which the parser already reports.
  unit.walk([&](FuncOp fn) {
    StringAttr name = fn.getSymNameAttr();
    if (!reachable.contains(name) || !fn.getMaxRecursionDepth().has_value() ||
        fn.getStackSize().has_value() || graph.isRecursive(name))
      return;
    fn.emitWarning("'max_recursion_depth' has no effect because '")
        << name.getValue() << "' is not recursive; remove it";
  });

  return failure(hadError);
}

namespace {

/// Computes the most workers of `@await_parallel` that can be active at the
/// same time while a function runs.
class WorkerCount {
public:
  WorkerCount(csl::CompilationUnitOp unit, const ReachableCallGraph &graph,
              const IndirectCallTargets &indirectTargets)
      : unit(unit), graph(graph), indirectTargets(indirectTargets) {}

  /// The most workers active at once while `fn` runs.  All functions of a
  /// recursion cycle share the count: recursion never runs through a
  /// callee of a multi-callee `@await_parallel`, so the recursive calls run
  /// one after the other.
  uint64_t of(StringAttr fn) {
    std::optional<unsigned> id = graph.getComponent(fn);
    if (!id)
      return 0;
    auto it = memo.find(*id);
    if (it != memo.end())
      return it->second;
    uint64_t result = 0;
    for (StringAttr member : graph.getFunctions())
      if (graph.getComponent(member) == id)
        if (auto memberFn = unit.lookupSymbol<FuncOp>(member))
          result = std::max(result, deepestSite(memberFn, *id));
    memo[*id] = result;
    return result;
  }

private:
  /// The most workers active at once at any call site of `fn`, a function
  /// of the component `id`.
  uint64_t deepestSite(FuncOp fn, unsigned id) {
    auto callee = [&](StringAttr name) -> uint64_t {
      return graph.getComponent(name) == id ? 0 : of(name);
    };
    uint64_t deepest = 0;
    fn.walk([&](Operation *op) {
      if (auto awaitParallel = mlir::dyn_cast<AwaitParallelOp>(op)) {
        if (awaitParallel.getFutures().size() < 2)
          return;
        // One worker per callee after the first, and every callee may be
        // using its own workers at the same time.
        uint64_t site = awaitParallel.getFutures().size() - 1;
        for (Value future : awaitParallel.getFutures())
          if (auto call = future.getDefiningOp<CallOp>())
            site += callee(call.getCalleeAttr().getRootReference());
        deepest = std::max(deepest, site);
      } else if (auto call = mlir::dyn_cast<CallOp>(op)) {
        if (!isMultiCalleeParallelCall(call))
          deepest = std::max(
              deepest, callee(call.getCalleeAttr().getRootReference()));
      } else if (auto icall = mlir::dyn_cast<IndirectCallOp>(op)) {
        for (FuncOp target : indirectTargets.getPossibleCallees(icall))
          deepest = std::max(deepest, callee(target.getSymNameAttr()));
      }
    });
    return deepest;
  }

  csl::CompilationUnitOp unit;
  const ReachableCallGraph &graph;
  const IndirectCallTargets &indirectTargets;
  DenseMap<unsigned, uint64_t> memo;
};

} // namespace

void checkWorkerTasks(csl::CompilationUnitOp unit,
                      const FiberReachableFuncs &reachable,
                      const IndirectCallTargets &indirectTargets,
                      ArrayRef<FuncOp> rootTasks, csl::TargetArch arch) {
  // The first @await_parallel that needs workers, in source order.
  AwaitParallelOp firstSite;
  unit.walk([&](AwaitParallelOp awaitParallel) {
    if (!firstSite && awaitParallel.getFutures().size() > 1)
      firstSite = awaitParallel;
  });
  if (!firstSite)
    return;

  ReachableCallGraph graph(unit, reachable, indirectTargets);
  WorkerCount workers(unit, graph, indirectTargets);
  uint64_t needed = 0;
  SmallVector<std::pair<FuncOp, uint64_t>> rootNeeds;
  for (FuncOp root : rootTasks) {
    uint64_t rootNeed = workers.of(root.getSymNameAttr());
    if (rootNeed == 0)
      continue;
    needed += rootNeed;
    rootNeeds.emplace_back(root, rootNeed);
  }
  if (needed == 0)
    return;
  auto formatWorkers = [](uint64_t count) {
    return std::to_string(count) + (count == 1 ? " worker" : " workers");
  };

  // User code that claims task table entries at runtime competes with the
  // workers, and how many entries it holds is unknown.  On WSE-2, colors and
  // data tasks share the entries.
  BuiltinCallOp runtimeClaim;
  unit.walk([&](BuiltinCallOp op) {
    csl::BuiltinKind kind = op.getBuiltin();
    if (!runtimeClaim &&
        (kind == csl::BuiltinKind::ClaimLocalTask ||
         (arch == csl::TargetArch::WSE2 &&
          (kind == csl::BuiltinKind::ClaimColor ||
           kind == csl::BuiltinKind::ClaimDataTask))))
      runtimeClaim = op;
  });
  if (runtimeClaim) {
    InFlightDiagnostic diag = firstSite.emitRemark()
        << "cannot check whether enough local tasks are free for the "
           "workers of @await_parallel, because the program also claims "
           "tasks at runtime (up to "
        << formatWorkers(needed)
        << " may be active at the same time); if they run out at runtime, "
           "the PE halts";
    diag.attachNote(runtimeClaim.getLoc()) << "task claimed at runtime here";
    return;
  }

  csl::ResourceBitmaps bitmaps = csl::ResourceBitmaps::loadFromUnit(unit, arch);
  uint64_t available = 0;
  for (uint16_t id = 0; id <= 28; ++id)
    if (bitmaps.isLocalTaskFree(arch, id))
      ++available;
  if (needed <= available)
    return;
  InFlightDiagnostic diag = firstSite.emitWarning()
      << "cannot prove that enough local tasks are free for the workers of "
         "@await_parallel: up to "
      << formatWorkers(needed) << " may be active at the same time, but only "
      << available
      << " local task entries are left after the tasks bound or claimed at "
         "comptime; if they run out at runtime, the PE halts";
  diag.attachNote()
      << "the count is an upper bound: it assumes that all callees of every "
         "@await_parallel are still running at the same time, although a "
         "worker returns its task as soon as its callee finishes, and that "
         "every indirect call calls its most demanding possible callee";
  for (auto [root, rootNeed] : rootNeeds)
    diag.attachNote(root.getLoc())
        << "async task '" << root.getSymName() << "' may need up to "
        << formatWorkers(rootNeed);
  diag.attachNote() << "to stay within the free entries, run fewer callees "
                       "of @await_parallel at the same time, or bind or "
                       "claim fewer tasks";
}

} // namespace mlir::typed_csl