//===- FiberStackResolver.cpp - Compute per-fn max_stack from ELF data ----===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberStackResolver.h"

#include "csl-mlir/Dialect/TypedCSL/Transforms/DwarfInlineReader.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/StackSizesReader.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "fiber-sizing"

using namespace mlir;

namespace mlir::typed_csl {

namespace {

/// Round byte count up to an even u16-word count (= multiple of 4 bytes).
/// Worker SPs derived by chained subtraction stay 32-bit aligned.
static uint16_t bytesToEvenWords(uint64_t bytes) {
  uint64_t words = (bytes + 1) / 2;
  if (words & 1)
    ++words;
  if (words > std::numeric_limits<uint16_t>::max())
    words = std::numeric_limits<uint16_t>::max();
  return static_cast<uint16_t>(words);
}

/// Round byte count up to a u32-word count (= multiple of 4 bytes,
/// expressed in u32 units).  Used for `__fiber_stack_<root>` array dim.
static uint32_t bytesToU32Words(uint64_t bytes) {
  return static_cast<uint32_t>((bytes + 3) / 4);
}

/// Strongly connected components of the captured call graph.  Edges are the
/// sync callees and the callees of every parallel site; a function with a
/// `stack_size(N)` override has no outgoing edges, since the resolver does
/// not walk its body.
class CallGraphComponents {
public:
  explicit CallGraphComponents(const CapturedCallGraph &cg) : cg(cg) {
    for (const auto &entry : cg.funcs)
      if (!index.contains(entry.getKey()))
        strongConnect(entry.getKey());
  }

  /// Whether `name` lies on a cycle of the call graph.
  bool isRecursive(llvm::StringRef name) const {
    auto it = component.find(name);
    if (it == component.end())
      return false;
    if (members[it->second].size() > 1)
      return true;
    bool selfCall = false;
    forEachCallee(name, [&](llvm::StringRef callee, bool) {
      selfCall |= callee == name;
    });
    return selfCall;
  }

  bool inSameComponent(llvm::StringRef a, llvm::StringRef b) const {
    auto itA = component.find(a);
    auto itB = component.find(b);
    return itA != component.end() && itB != component.end() &&
           itA->second == itB->second;
  }

  /// The functions in the component of `name`.
  llvm::ArrayRef<std::string> getMembers(llvm::StringRef name) const {
    return members[component.at(name)];
  }

  /// Invoke `callback` with every callee of `name` and whether the call is
  /// a parallel site with several callees.
  void forEachCallee(
      llvm::StringRef name,
      llvm::function_ref<void(llvm::StringRef callee, bool isParallel)>
          callback) const {
    auto it = cg.funcs.find(name);
    if (it == cg.funcs.end() || it->second.stackSizeOverride.has_value())
      return;
    for (const std::string &callee : it->second.syncCallees)
      callback(callee, /*isParallel=*/false);
    for (const auto &site : it->second.parallelSites)
      for (const std::string &callee : site)
        callback(callee, /*isParallel=*/site.size() > 1);
  }

private:
  /// Tarjan's algorithm.
  void strongConnect(llvm::StringRef name) {
    index[name] = lowLink[name] = nextIndex++;
    tarjanStack.push_back(name.str());
    onStack.insert(name);
    forEachCallee(name, [&](llvm::StringRef callee, bool) {
      if (!cg.funcs.contains(callee))
        return;
      if (!index.contains(callee)) {
        strongConnect(callee);
        lowLink[name] = std::min(lowLink[name], lowLink[callee]);
      } else if (onStack.contains(callee)) {
        lowLink[name] = std::min(lowLink[name], index[callee]);
      }
    });
    if (lowLink[name] != index[name])
      return;
    unsigned id = members.size();
    members.emplace_back();
    std::string member;
    do {
      member = tarjanStack.pop_back_val();
      onStack.erase(member);
      component[member] = id;
      members[id].push_back(member);
    } while (member != name);
  }

  const CapturedCallGraph &cg;
  llvm::StringMap<unsigned> component;
  std::vector<std::vector<std::string>> members;

  llvm::StringMap<unsigned> index;
  llvm::StringMap<unsigned> lowLink;
  llvm::SmallVector<std::string> tarjanStack;
  llvm::StringSet<> onStack;
  unsigned nextIndex = 0;
};

/// Resolver state — shared by the recursive contribution computation.
struct Resolver {
  llvm::StringRef context;
  const llvm::StringMap<uint32_t> &stackSizes;
  const DwarfInlineInfo &inlineInfo;
  const CapturedCallGraph &cg;
  CallGraphComponents components{cg};

  /// Memoized `contrib(F)` (in bytes).  contrib(F) is the stack reach
  /// contributed by F when invoked from a caller — F's own frame plus
  /// the worst-case reach of F's body.  Equals `max_stack(F)` for an
  /// out-of-line F; equals `extra_stack_from(F)` for an inlined-only F
  /// (own frame = 0 because it's already in the inlining caller's
  /// `.stack_sizes` entry).
  llvm::StringMap<uint64_t> memo;
  llvm::StringSet<> inProgress;

  /// Memoized reach of a recursive function's body when at most `budget`
  /// further activations of annotated functions of its cycle may follow.
  llvm::DenseMap<std::pair<llvm::StringRef, uint64_t>, uint64_t>
      recursiveBodyMemo;

  /// Functions whose missing frame size was already reported.
  llvm::StringSet<> reportedMissingFrames;

  /// A function whose calls contribute nothing.  Used to bound the body of
  /// a function with a `stack_size(N)` override from below: a recursive
  /// call back into it may not happen at all.
  llvm::StringRef excluded;

  /// Set when a hard error was emitted.  Callers stop on first error.
  bool hadError = false;

  Resolver(llvm::StringRef context, const llvm::StringMap<uint32_t> &stackSizes,
           const DwarfInlineInfo &inlineInfo, const CapturedCallGraph &cg)
      : context(context), stackSizes(stackSizes), inlineInfo(inlineInfo),
        cg(cg) {}

  void emitError(llvm::StringRef msg) {
    llvm::errs() << "error: stack size resolution: " << msg << " (in "
                 << context << ")\n";
    hadError = true;
  }

  /// Look up F's own frame size in bytes, returning false if it has no
  /// `.stack_sizes` entry.
  bool getOwnFrameBytes(llvm::StringRef name, uint64_t &out) const {
    auto it = stackSizes.find(name);
    if (it == stackSizes.end())
      return false;
    out = it->second;
    return true;
  }

  /// Determine the own frame size of a captured function F whose body the
  /// resolver walks.  Returns false if F's body contributes nothing: F was
  /// dead-code eliminated, or its frame size is missing (reported once).
  ///
  /// For inlined-only F the own frame is 0 — already baked into whichever
  /// caller inlined F.  F's CALLEES are still made from F's body and count
  /// toward the enclosing function's reach.
  bool getBodyFrameBytes(llvm::StringRef name, uint64_t &own) {
    own = 0;
    if (getOwnFrameBytes(name, own) || inlineInfo.isInlinedOnly(name))
      return true;
    // Three sub-cases:
    //   (a) DWARF has no trace at all → LLVM dead-code-eliminated the
    //       function (no actual call site reached it from the
    //       surviving entry points).  Charge contrib = 0; recursion
    //       would have no .stack_sizes / DWARF data to act on anyway.
    //   (b) DWARF lists the function as `DW_TAG_subprogram` (it
    //       survives out-of-line) but `.stack_sizes` doesn't — that
    //       means -fstack-size-section was NOT applied to the
    //       translation unit holding F.  Hard error: our shim missed
    //       a TU.
    //   (c) any other inconsistency: hard error.
    bool dwarfKnowsSurviving = inlineInfo.survivingSubprograms.contains(name);
    bool dwarfKnowsInlined = inlineInfo.seenInlined.contains(name);
    if (!dwarfKnowsSurviving && !dwarfKnowsInlined) {
      LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: contrib('" << name
                              << "') = 0 B (dead — no .stack_sizes / "
                                 "DWARF presence)\n");
      return false;
    }
    if (reportedMissingFrames.insert(name).second)
      emitError(("function '" + name +
                 "' has no `.stack_sizes` entry but DWARF does list it as "
                 "a surviving subprogram — the LD_PRELOAD shim likely "
                 "missed the translation unit hosting it")
                    .str());
    return false;
  }

  /// Per-site contribution to the enclosing function's max stack — i.e.
  /// the byte distance below `sp_at_site` the deepest reach extends, given
  /// the reach of each callee.
  ///
  /// - `sync` site to g: contribution = reach(g)
  /// - `parallel(g0..gn-1)` site:
  ///     * If only one callee (n==1), the pass emits a direct sync call
  ///       and skips workers; contribution = reach(g0).
  ///     * Otherwise the chain math reserves (in this order, below
  ///       sp_at_site):
  ///         (1) `max(reach(g0), contrib(caller_join))` bytes — the
  ///             sync callee g0's frame chain, and (after g0 returns, so
  ///             never live at the same time) the caller's post-site
  ///             frames (`caller_join` + the `suspend` it tail-calls),
  ///             whose reach is measured from the ELF like any callee.
  ///         (2) For each worker g_i (i = 1..n-1):
  ///             `kArgAreaU16Slots*2` bytes (arg-area at the worker's
  ///             stack top) + `reach(g_i)` bytes (the worker's frame
  ///             chain).
  ///       Workers' regions never overlap with each other or with (1).
  uint64_t
  parallelSite(const std::vector<std::string> &callees,
               llvm::function_ref<uint64_t(llvm::StringRef)> reach) {
    if (callees.empty())
      return 0;
    if (callees.size() == 1)
      return reach(callees[0]);

    // Post-site reserve = measured reach of caller_join (+ the suspend it
    // tail-calls), which shares g0's region since it runs after g0 returns.
    uint64_t reserveBytes = contrib(kCallerJoinRuntimeSym);
    constexpr uint64_t argAreaBytes = kArgAreaU16Slots * 2;
    uint64_t total = std::max(reserveBytes, reach(callees[0]));
    for (size_t i = 1; i < callees.size(); ++i)
      total += argAreaBytes + reach(callees[i]);
    return total;
  }

  /// The deepest reach over all call sites in a function body.
  uint64_t deepestSite(const FuncCallInfo &info,
                       llvm::function_ref<uint64_t(llvm::StringRef)> reach) {
    uint64_t deepest = 0;
    for (const std::string &g : info.syncCallees)
      deepest = std::max(deepest, reach(g));
    for (const auto &site : info.parallelSites)
      deepest = std::max(deepest, parallelSite(site, reach));
    return deepest;
  }

  /// contrib(F) — stack reach F contributes when called.
  uint64_t contrib(llvm::StringRef nameSR) {
    std::string name = nameSR.str();
    if (hadError || nameSR == excluded)
      return 0;

    auto memoIt = memo.find(name);
    if (memoIt != memo.end())
      return memoIt->second;

    if (!inProgress.insert(name).second) {
      emitError(("recursive call cycle through '" + nameSR +
                 "'; add `max_recursion_depth(N)` or `stack_size(N)` to one "
                 "function in the cycle")
                    .str());
      return 0;
    }

    uint64_t result = 0;

    auto fIt = cg.funcs.find(name);
    if (fIt == cg.funcs.end()) {
      // Function isn't in the captured graph at all (e.g. a sync callee
      // from a non-async region we didn't record).  Charge its own frame
      // size if we can find one; otherwise 0 — e.g. an inline-asm-only
      // helper or a function inlined everywhere.
      uint64_t own = 0;
      if (getOwnFrameBytes(name, own))
        result = own;
      inProgress.erase(name);
      memo[name] = result;
      return result;
    }

    const FuncCallInfo &info = fIt->second;

    // Honor user-supplied stack_size(N): contribution = N, period.  We
    // still validate the lower bound separately.
    if (info.stackSizeOverride.has_value()) {
      result = *info.stackSizeOverride;
      LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: contrib('" << nameSR << "') = "
                              << result << " B (stack_size override)\n");
    } else if (components.isRecursive(name)) {
      // Entering the cycle from outside: every annotated function of the
      // cycle may still be activated as often as it declares.
      result = enterRecursive(name, recursionBudget(name));
      LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: contrib('" << nameSR
                              << "') = " << result << " B [recursive]\n");
    } else {
      uint64_t own = 0;
      if (getBodyFrameBytes(name, own)) {
        uint64_t deepest = deepestSite(
            info, [&](llvm::StringRef callee) { return contrib(callee); });
        result = own + deepest;
        LLVM_DEBUG(llvm::dbgs()
                   << "fiber-sizing: contrib('" << nameSR << "') = " << result
                   << " B = own " << own << " B + deepest " << deepest << " B "
                   << (inlineInfo.isInlinedOnly(name) ? "[inlined-only]"
                                                      : "[out-of-line]")
                   << "\n");
      }
    }

    inProgress.erase(name);
    memo[name] = result;
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Bounded recursion
  //
  // Every cycle of the call graph contains a function with
  // `max_recursion_depth(N)` (FiberValidation rejects the others).  A cycle
  // is sized by unrolling it: the reach of a function in a cycle is computed
  // for a budget of remaining activations of the cycle's annotated
  // functions, and a call within the cycle passes the budget on.  Entering
  // an annotated function uses up one activation; once the budget is
  // exhausted, the call cannot happen.  The budget of a cycle is the sum of
  // its annotations, which is exact for a single annotated function and an
  // upper bound otherwise.
  //
  // A callee of a multi-callee @await_parallel runs in a region of fixed
  // size, which cannot hold a recursion, so no cycle runs through one
  // (FiberValidation rejects that, too).
  //===--------------------------------------------------------------------===//

  /// The activation budget of the cycle containing `name`.
  uint64_t recursionBudget(llvm::StringRef name) {
    uint64_t budget = 0;
    for (const std::string &member : components.getMembers(name)) {
      const FuncCallInfo &info = cg.funcs.find(member)->second;
      budget += info.maxRecursionDepth.value_or(0);
    }
    if (budget == 0)
      emitError(("recursive call cycle through '" + name +
                 "'; add `max_recursion_depth(N)` or `stack_size(N)` to one "
                 "function in the cycle")
                    .str());
    return budget;
  }

  /// The reach of a call to `name` within its cycle, with `budget`
  /// activations of annotated functions left.
  uint64_t enterRecursive(llvm::StringRef name, uint64_t budget) {
    if (cg.funcs.find(name)->second.maxRecursionDepth.has_value()) {
      if (budget == 0)
        return 0;
      --budget;
    }
    return recursiveBody(name, budget);
  }

  /// The reach of the body of `name`, a function in a cycle, with `budget`
  /// activations of annotated functions left for calls within the cycle.
  uint64_t recursiveBody(llvm::StringRef name, uint64_t budget) {
    if (hadError)
      return 0;
    // Key the memo by the captured graph's own copy of the name, which
    // outlives every caller's string.
    auto fIt = cg.funcs.find(name);
    name = fIt->getKey();
    auto memoIt = recursiveBodyMemo.find({name, budget});
    if (memoIt != recursiveBodyMemo.end())
      return memoIt->second;

    const FuncCallInfo &info = fIt->second;
    uint64_t result = 0;
    uint64_t own = 0;
    if (getBodyFrameBytes(name, own)) {
      uint64_t deepest = deepestSite(info, [&](llvm::StringRef callee) {
        if (components.inSameComponent(name, callee))
          return enterRecursive(callee, budget);
        return contrib(callee);
      });
      result = own + deepest;
    }
    for (const auto &site : info.parallelSites) {
      if (site.size() < 2)
        continue;
      for (const std::string &callee : site)
        if (components.inSameComponent(name, callee))
          emitError(("recursive call from '" + name + "' to '" + callee +
                     "' through @await_parallel cannot be sized")
                        .str());
    }
    recursiveBodyMemo[{name, budget}] = result;
    return result;
  }
};

} // namespace

FailureOr<ResolvedSizes>
resolveFiberSizes(const llvm::StringMap<uint32_t> &stackSizes,
                  const DwarfInlineInfo &inlineInfo,
                  const CapturedCallGraph &cg, llvm::StringRef context) {
  Resolver r(context, stackSizes, inlineInfo, cg);

  ResolvedSizes out;

  // Compute each parallel callee's required size.  Round 2 of the
  // compile feeds these into a ResolvedSizeSource, which the re-run
  // FiberLoweringPass queries when emitting the partition arithmetic.
  // We don't know the exact query set up-front, so we cover every
  // function that appears as a parallel callee in the captured graph.
  // Functions with a `stack_size(N)` override are still resolved (so the
  // resolver is the single source of truth) — their `contrib()` just
  // returns the override.
  llvm::StringSet<> parallelCallees;
  for (const auto &kv : cg.funcs)
    for (const auto &site : kv.second.parallelSites)
      for (const std::string &c : site)
        parallelCallees.insert(c);

  for (const auto &entry : parallelCallees) {
    llvm::StringRef name = entry.getKey();
    uint64_t bytes = r.contrib(name);
    uint16_t words = bytesToEvenWords(bytes);
    out.calleeSizeWords[name] = words;
    LLVM_DEBUG(llvm::dbgs()
               << "fiber-sizing: callee '" << name << "' → " << bytes << " B → "
               << words << " u16-words (rounded even)\n");
  }

  // Publish the post-site reserve (caller_join's measured reach) so the
  // round-2 lowering can query it through the ResolvedSizeSource just like
  // a regular callee.  The lowering uses it for the sync slot of every
  // multi-callee @await_parallel: max(contrib(g0), contrib(caller_join)).
  {
    uint64_t bytes = r.contrib(kCallerJoinRuntimeSym);
    uint16_t words = bytesToEvenWords(bytes);
    out.calleeSizeWords[kCallerJoinRuntimeSym] = words;
    LLVM_DEBUG(llvm::dbgs()
               << "fiber-sizing: post-site reserve (caller_join) → " << bytes
               << " B → " << words << " u16-words (rounded even)\n");
  }

  // Per-root-task fiber stack array dimension.  The array's top
  // `kArgAreaU16Slots` u16 words hold the task's initial r2..r5/r15
  // arg-area (where pool[task].sp points on entry); the function frame
  // chain runs BELOW that.  So we need room for `contrib(root)` bytes of
  // frames PLUS `kArgAreaU16Slots*2` bytes of arg-area.
  constexpr uint64_t argAreaBytes = kArgAreaU16Slots * 2;
  for (const auto &entry : cg.rootTasks) {
    llvm::StringRef name = entry.getKey();
    uint64_t bytes = r.contrib(name) + argAreaBytes;
    uint32_t words = bytesToU32Words(bytes);
    out.rootStackU32Words[name] = words;
    LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: root task '" << name << "' → "
                            << r.contrib(name) << " B contrib + "
                            << argAreaBytes << " B arg-area = " << bytes
                            << " B → " << words << " u32-words\n");
  }

  if (r.hadError)
    return failure();

  // Lower-bound validation for every stack_size override: the body's
  // demand, measured without the override at this node.  Deeper overrides
  // stay opaque, as in contrib().  A recursive call back into the
  // overridden function counts as 0 — it may not happen at all — so each
  // override is checked with its own resolver.
  bool overrideRejected = false;
  for (const auto &kv : cg.funcs) {
    if (!kv.second.stackSizeOverride.has_value())
      continue;
    uint64_t override_ = *kv.second.stackSizeOverride;
    Resolver lowerBound(context, stackSizes, inlineInfo, cg);
    lowerBound.excluded = kv.first();
    uint64_t own = 0;
    lowerBound.getOwnFrameBytes(kv.first(), own);
    uint64_t deepest =
        lowerBound.deepestSite(kv.second, [&](llvm::StringRef callee) {
          return lowerBound.contrib(callee);
        });
    uint64_t lb = own + deepest;
    if (override_ < lb) {
      llvm::errs() << "error: stack size resolution: stack_size(" << override_
                   << ") on function '" << kv.first()
                   << "' is provably too small — lower bound on its actual "
                      "stack reach is "
                   << lb << " bytes (own frame " << own
                   << " B + deepest call site " << deepest
                   << " B).  Increase "
                      "the override, or remove it to let the resolver compute "
                      "the value.\n";
      overrideRejected = true;
    }
  }
  if (overrideRejected)
    return failure();

  return out;
}

FailureOr<ResolvedSizes> resolveFiberSizes(llvm::StringRef elfPath,
                                           const CapturedCallGraph &cg) {
  FailureOr<llvm::StringMap<uint32_t>> stackSizesOrErr =
      readStackSizesFromElf(elfPath);
  if (failed(stackSizesOrErr))
    return failure();

  FailureOr<DwarfInlineInfo> inlineOrErr = readDwarfInlineInfo(elfPath);
  if (failed(inlineOrErr))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: resolveFiberSizes for ELF '"
                          << elfPath << "': " << stackSizesOrErr->size()
                          << " .stack_sizes entries, "
                          << inlineOrErr->survivingSubprograms.size()
                          << " surviving DWARF subprograms, "
                          << inlineOrErr->seenInlined.size()
                          << " inlined origins\n");

  return resolveFiberSizes(*stackSizesOrErr, *inlineOrErr, cg, elfPath);
}

} // namespace mlir::typed_csl
