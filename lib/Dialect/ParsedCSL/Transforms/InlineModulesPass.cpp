//===- InlineModulesPass.cpp - Inline Module Contents -----------*- C++ -*-===//
//
// This file implements a pass that inlines module contents to the top level.
//
// Algorithm:
// 1. Pre-pass: Rename nested module names (inner -> outer__inner)
// 2. Batch-update all FuncRefAttr/GlobalRefAttr references in a single walk
//    using a moduleId -> moduleName map (avoids per-module full-IR walks)
// 3. For each top-level module:
//    a. Rename symbols to final names + check collisions
//    b. Move contents to parent scope, erase module
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "inline-modules"

using namespace mlir;
using namespace mlir::parsed_csl;

namespace mlir::parsed_csl {
#define GEN_PASS_DEF_INLINEMODULESPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"
} // namespace mlir::parsed_csl

namespace {

struct InlineModulesPass
    : public parsed_csl::impl::InlineModulesPassBase<InlineModulesPass> {
  using InlineModulesPassBase::InlineModulesPassBase;

  llvm::StringSet<> usedNames;
  bool hasCollision = false;

  //===--------------------------------------------------------------------===//
  // Name computation
  //===--------------------------------------------------------------------===//

  /// Computes the final symbol name for a function.
  /// All module functions get the module prefix. The linkname attribute
  /// (required for export/extern) determines the external name separately.
  std::string computeFinalName(FuncOp func, parsed_csl::ModuleOp module) {
    StringRef original = func.getSymName();

    if (!module || !module.getModuleName())
      return original.str();

    return (*module.getModuleName() + "__" + original).str();
  }

  /// Computes the final symbol name for a global.
  /// All module globals get the module prefix. The linkname attribute
  /// (required for export/extern) determines the external name separately.
  std::string computeFinalName(GlobalOp global, parsed_csl::ModuleOp module) {
    StringRef original = global.getSymName();

    if (!module || !module.getModuleName())
      return original.str();

    return (*module.getModuleName() + "__" + original).str();
  }

  //===--------------------------------------------------------------------===//
  // Collision detection
  //===--------------------------------------------------------------------===//

  void checkCollision(StringRef name, Operation *op) {
    if (!usedNames.insert(name).second) {
      op->emitError()
          << "internal symbol name collision: '" << name
          << "'; the compiler uses '__' to create unique names for module "
             "members (e.g., module 'foo' member 'bar' becomes 'foo__bar'). "
             "Avoid using '__' in global symbol names to prevent collisions";
      hasCollision = true;
    }
  }

  /// Register top-level symbols for collision detection.
  void registerTopLevelSymbols(csl::CompilationUnitOp unit) {
    for (auto &op : unit.getBody()->getOperations()) {
      if (auto func = dyn_cast<FuncOp>(op))
        checkCollision(func.getSymName(), func);
      else if (auto global = dyn_cast<GlobalOp>(op))
        checkCollision(global.getSymName(), global);
    }
  }

  //===--------------------------------------------------------------------===//
  // Auto-linkname
  //===--------------------------------------------------------------------===//

  /// Converts a "__"-separated symbol name to dot-separated form for
  /// auto-linkname. E.g., "outer__inner__func" -> "outer.inner.func".
  static std::string toDotSeparated(StringRef name) {
    std::string result;
    size_t pos = 0;
    while (pos < name.size()) {
      size_t next = name.find("__", pos);
      if (next == StringRef::npos) {
        result += name.substr(pos).str();
        break;
      }
      result += name.substr(pos, next - pos).str();
      result += ".";
      pos = next + 2;
    }
    return result;
  }

  /// Auto-attach dot-separated linknames to private symbols in a module
  /// that don't already have a user-specified linkname.
  /// Must be called AFTER symbols have been renamed to their final names.
  void attachAutoLinknames(parsed_csl::ModuleOp module, MLIRContext *ctx) {
    for (auto func : module.getOps<FuncOp>()) {
      if (func.getLinkname() || func.getIsExport() || func.getIsExtern() ||
          func.getIsInline())
        continue;
      std::string linkname = toDotSeparated(func.getSymName());
      LDBG() << "  Auto-linkname: @" << func.getSymName() << " -> \""
             << linkname << "\"";
      func.setLinknameAttr(StringAttr::get(ctx, linkname));
    }

    for (auto global : module.getOps<GlobalOp>()) {
      if (global.getLinkname() || global.getIsExport() || global.getIsExtern())
        continue;
      std::string linkname = toDotSeparated(global.getSymName());
      LDBG() << "  Auto-linkname: @" << global.getSymName() << " -> \""
             << linkname << "\"";
      global.setLinknameAttr(StringAttr::get(ctx, linkname));
    }
  }

  //===--------------------------------------------------------------------===//
  // Module inlining
  //===--------------------------------------------------------------------===//

  /// Returns the next top-level module, or nullptr if none remain.
  parsed_csl::ModuleOp getNextTopLevelModule(csl::CompilationUnitOp unit) {
    for (auto &op : unit.getBody()->getOperations()) {
      if (auto module = dyn_cast<parsed_csl::ModuleOp>(op))
        return module;
    }
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Batched reference update
  //===--------------------------------------------------------------------===//

  /// Update all FuncRefAttr/GlobalRefAttr references for ALL modules in a
  /// single walk over the compilation unit. This replaces the previous
  /// per-module AttrTypeReplacer walk which was O(modules * ops).
  void updateAllModuleReferences(csl::CompilationUnitOp unit) {
    // Build moduleId -> moduleName map for all modules.
    // Module names are already fully qualified from the pre-pass.
    DenseMap<uint64_t, StringRef> moduleNames;
    walkTopLevelDecls<parsed_csl::ModuleOp>(unit, [&](parsed_csl::ModuleOp m) {
      if (m.getModuleName())
        moduleNames[m.getModuleIdentifier()] = *m.getModuleName();
    });

    if (moduleNames.empty())
      return;

    MLIRContext *ctx = unit->getContext();
    AttrTypeReplacer replacer;

    replacer.addReplacement(
        [&](csl::FuncRefAttr attr)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          if (!attr.getModuleIdentifier())
            return std::nullopt;
          auto it = moduleNames.find(*attr.getModuleIdentifier());
          if (it == moduleNames.end())
            return std::nullopt;

          std::string finalName =
              (it->second + "__" + attr.getSymbolName().getValue()).str();
          LDBG() << "  Updating FuncRef @" << attr.getSymbolName().getValue()
                 << " -> @" << finalName;

          return std::make_pair(
              csl::FuncRefAttr::get(ctx, std::nullopt,
                                    FlatSymbolRefAttr::get(ctx, finalName),
                                    attr.getType()),
              WalkResult::advance());
        });

    replacer.addReplacement(
        [&](csl::GlobalRefAttr attr)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          if (!attr.getModuleIdentifier())
            return std::nullopt;
          auto it = moduleNames.find(*attr.getModuleIdentifier());
          if (it == moduleNames.end())
            return std::nullopt;

          std::string finalName =
              (it->second + "__" + attr.getSymbolName().getValue()).str();
          LDBG() << "  Updating GlobalRef @" << attr.getSymbolName().getValue()
                 << " -> @" << finalName;

          return std::make_pair(
              csl::GlobalRefAttr::get(ctx, std::nullopt,
                                      FlatSymbolRefAttr::get(ctx, finalName),
                                      attr.getType()),
              WalkResult::advance());
        });

    replacer.addReplacement([&](csl::CoroFrameType type)
                                -> std::optional<std::pair<Type, WalkResult>> {
      if (!type.getModuleIdentifier())
        return std::nullopt;
      auto it = moduleNames.find(*type.getModuleIdentifier());
      if (it == moduleNames.end())
        return std::nullopt;

      std::string finalName =
          (it->second + "__" + type.getSymbolName().getValue()).str();
      LDBG() << "  Updating CoroFrame @" << type.getSymbolName().getValue()
             << " -> @" << finalName;

      return std::make_pair(
          csl::CoroFrameType::get(ctx, std::nullopt,
                                  FlatSymbolRefAttr::get(ctx, finalName)),
          WalkResult::advance());
    });

    replacer.addReplacement([&](csl::FutureType type)
                                -> std::optional<std::pair<Type, WalkResult>> {
      if (!type.getModuleIdentifier())
        return std::nullopt;
      auto it = moduleNames.find(*type.getModuleIdentifier());
      if (it == moduleNames.end())
        return std::nullopt;

      std::string finalName =
          (it->second + "__" + type.getSymbolName().getValue()).str();
      LDBG() << "  Updating Future @" << type.getSymbolName().getValue()
             << " -> @" << finalName;

      return std::make_pair(
          csl::FutureType::get(ctx, std::nullopt,
                               FlatSymbolRefAttr::get(ctx, finalName),
                               type.getResultType()),
          WalkResult::advance());
    });

    replacer.recursivelyReplaceElementsIn(unit);
  }

  //===--------------------------------------------------------------------===//
  // Module inlining
  //===--------------------------------------------------------------------===//

  /// Inlines a single module into the compilation unit.
  /// References have already been updated by updateAllModuleReferences().
  LogicalResult inlineModule(parsed_csl::ModuleOp module,
                             csl::CompilationUnitOp unit) {
    if (!module.getModuleName()) {
      // Unnamed module was never used
      module->erase();
      return success();
    }

    MLIRContext *ctx = module->getContext();

    LDBG() << "Inlining module '" << *module.getModuleName()
           << "' (id=" << module.getModuleIdentifier() << ")";

    // Step 1: Set linkname on export/extern symbols that don't have one.
    // The linkname defaults to the original (unprefixed) symbol name.
    // Must happen before renaming so we capture the original name.
    for (auto func : module.getOps<FuncOp>()) {
      if ((func.getIsExport() || func.getIsExtern()) && !func.getLinkname()) {
        LDBG() << "  Auto-linkname (export/extern): @" << func.getSymName();
        func.setLinknameAttr(StringAttr::get(ctx, func.getSymName()));
      }
    }
    for (auto global : module.getOps<GlobalOp>()) {
      if ((global.getIsExport() || global.getIsExtern()) &&
          !global.getLinkname()) {
        LDBG() << "  Auto-linkname (export/extern): @" << global.getSymName();
        global.setLinknameAttr(StringAttr::get(ctx, global.getSymName()));
      }
    }

    // Step 2: Rename symbols and check for collisions.
    for (auto func : module.getOps<FuncOp>()) {
      std::string finalName = computeFinalName(func, module);
      checkCollision(finalName, func);
      LDBG() << "  Renaming function @" << func.getSymName() << " -> @"
             << finalName;
      func.setSymNameAttr(StringAttr::get(ctx, finalName));
    }

    for (auto global : module.getOps<GlobalOp>()) {
      std::string finalName = computeFinalName(global, module);
      checkCollision(finalName, global);
      LDBG() << "  Renaming global @" << global.getSymName() << " -> @"
             << finalName;
      global.setSymNameAttr(StringAttr::get(ctx, finalName));
    }

    // Step 2.5: Auto-attach dot-separated linknames (if enabled).
    // Must happen after renaming so computeFinalName is unaffected.
    if (cslFlags.autoLinkname)
      attachAutoLinknames(module, ctx);

    // Step 3: Move contents out and erase module.
    IRRewriter rewriter(module);
    rewriter.inlineBlockBefore(module.getBody(), module);
    rewriter.eraseOp(module);

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Main entry point
  //===--------------------------------------------------------------------===//

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();

    // Verify preconditions: no GetSymbolOp should remain
    if (unit.walk([&](GetSymbolOp op) {
              op.emitError() << "GetSymbolOp should have been replaced during "
                                "comptime evaluation";
              return WalkResult::interrupt();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Pre-pass: Rename nested module names.
    // Walk in pre-order so parent names are updated before children.
    unit.walk<WalkOrder::PreOrder>([](parsed_csl::ModuleOp module) {
      auto parent = module->getParentOfType<parsed_csl::ModuleOp>();
      if (parent && parent.getModuleName() && module.getModuleName()) {
        std::string newName =
            (*parent.getModuleName() + "__" + *module.getModuleName()).str();
        LDBG() << "Renaming nested module " << *module.getModuleName() << " -> "
               << newName;
        module.setModuleName(newName);
      }
      return WalkResult::advance();
    });

    // Register top-level symbols for collision detection
    registerTopLevelSymbols(unit);

    // Batch-update all module references in a single walk over the unit.
    // This replaces the per-module AttrTypeReplacer walks.
    updateAllModuleReferences(unit);

    // Inline modules one by one (nested modules get moved to top level
    // when their parent is inlined, then processed in subsequent iterations)
    while (parsed_csl::ModuleOp module = getNextTopLevelModule(unit)) {
      if (failed(inlineModule(module, unit))) {
        signalPassFailure();
        return;
      }
    }

    if (hasCollision) {
      signalPassFailure();
      return;
    }

    // Clean up
    IRRewriter rewriter(unit);
    (void)simplifyRegions(rewriter, unit->getRegions());
  }
};

} // namespace
