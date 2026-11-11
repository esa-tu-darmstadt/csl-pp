//===- FiberSizingContext.cpp - Extract captured call graph -------------===//
//
// Mirror of FiberLoweringPass's pre-lowering capture step: walks a
// post-stage-A PE module, reads the `csl.fiber.captured_*` attributes
// off each typed_csl.func, and assembles a CapturedCallGraph for the
// resolver.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizingContext.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fiber-sizing"

using namespace mlir;
using namespace mlir::typed_csl;

CapturedCallGraph mlir::typed_csl::extractCapturedCallGraph(ModuleOp peModule) {
  CapturedCallGraph cg;

  peModule.walk([&](FuncOp funcOp) {
    std::string name = funcOp.getSymName().str();
    FuncCallInfo info;

    if (std::optional<uint32_t> n = funcOp.getStackSize())
      info.stackSizeOverride = *n;
    if (std::optional<uint32_t> n = funcOp.getMaxRecursionDepth())
      info.maxRecursionDepth = *n;

    if (auto arr =
            funcOp->getAttrOfType<ArrayAttr>(kCapturedSyncCalleesAttrName)) {
      info.syncCallees.reserve(arr.size());
      for (Attribute a : arr) {
        if (auto s = mlir::dyn_cast<FlatSymbolRefAttr>(a))
          info.syncCallees.push_back(s.getValue().str());
        else if (auto s = mlir::dyn_cast<StringAttr>(a))
          info.syncCallees.push_back(s.getValue().str());
      }
    }

    if (auto arr =
            funcOp->getAttrOfType<ArrayAttr>(kCapturedParallelSitesAttrName)) {
      info.parallelSites.reserve(arr.size());
      for (Attribute a : arr) {
        auto siteArr = mlir::dyn_cast<ArrayAttr>(a);
        if (!siteArr)
          continue;
        std::vector<std::string> site;
        site.reserve(siteArr.size());
        for (Attribute c : siteArr) {
          if (auto s = mlir::dyn_cast<FlatSymbolRefAttr>(c))
            site.push_back(s.getValue().str());
          else if (auto s = mlir::dyn_cast<StringAttr>(c))
            site.push_back(s.getValue().str());
        }
        info.parallelSites.push_back(std::move(site));
      }
    }

    bool isRoot = funcOp->hasAttr(kCapturedRootTaskAttrName);
    if (isRoot)
      cg.rootTasks.insert(name);

    // Only retain functions that contributed at least one piece of
    // captured info or are root tasks — otherwise the entry is noise.
    bool keep = info.stackSizeOverride.has_value() ||
                info.maxRecursionDepth.has_value() ||
                !info.syncCallees.empty() || !info.parallelSites.empty() ||
                isRoot;
    if (keep)
      cg.funcs[name] = std::move(info);
  });

  return cg;
}

bool mlir::typed_csl::peNeedsFiberSizeResolution(ModuleOp peModule) {
  // Build a name → FuncOp map once.  FuncOps live inside the inner
  // csl.compilation_unit, so ModuleOp::lookupSymbol on the outer module
  // returns nullptr; we keep direct FuncOp references instead.
  llvm::DenseMap<StringRef, FuncOp> funcByName;
  peModule.walk([&](FuncOp f) { funcByName[f.getSymName()] = f; });

  auto lacksStackSize = [&](StringRef sym) -> bool {
    auto it = funcByName.find(sym);
    if (it == funcByName.end())
      return false; // symbol unknown — not a query target
    if (!it->second.getIsAsync())
      return false; // sync fns aren't queried
    return !it->second.getStackSize().has_value();
  };

  bool needs = false;

  // Root tasks: async FuncOps bound via @bind_*_task.
  peModule.walk([&](BuiltinCallOp call) -> WalkResult {
    csl::BuiltinKind kind = call.getBuiltin();
    if (kind != csl::BuiltinKind::BindLocalTask &&
        kind != csl::BuiltinKind::BindControlTask &&
        kind != csl::BuiltinKind::BindDataTask)
      return WalkResult::advance();
    if (call.getArgs().empty())
      return WalkResult::advance();
    csl::FuncRefAttr funcRef = getFuncRef(call.getArgs()[0]);
    if (!funcRef)
      return WalkResult::advance();
    if (lacksStackSize(funcRef.getSymbolName().getValue())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "fiber-sizing: '" << funcRef.getSymbolName().getValue()
                 << "' is a bound async root task without stack_size(N) "
                    "→ round 2 needed\n");
      needs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (needs)
    return true;

  // Parallel-await callees.
  peModule.walk([&](AwaitParallelOp ap) -> WalkResult {
    for (Value future : ap.getFutures()) {
      auto ft = mlir::dyn_cast<csl::FutureType>(future.getType());
      if (!ft)
        continue;
      StringRef sym = ft.getSymbolName().getValue();
      if (lacksStackSize(sym)) {
        LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: '" << sym
                                << "' is an @await_parallel callee without "
                                   "stack_size(N) → round 2 needed\n");
        needs = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  LLVM_DEBUG(if (!needs) llvm::dbgs()
             << "fiber-sizing: PE has no round-2-relevant size queries\n");
  return needs;
}
