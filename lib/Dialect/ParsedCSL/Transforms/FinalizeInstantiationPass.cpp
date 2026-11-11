//===- FinalizeInstantiationPass.cpp - Erase generics, rename funcs -------===//
//
// After comptime evaluation, erase GenericFuncOps and rename instantiated
// FuncOps back to their original names where unambiguous.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::parsed_csl;

namespace mlir {
namespace parsed_csl {

#define GEN_PASS_DEF_FINALIZEINSTANTIATIONPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

} // namespace parsed_csl
} // namespace mlir

namespace {

struct FinalizeInstantiationPass
    : public parsed_csl::impl::FinalizeInstantiationPassBase<
          FinalizeInstantiationPass> {

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();

    // Step 1: Erase all GenericFuncOps (collect first since we modify IR)
    SmallVector<parsed_csl::GenericFuncOp> generics;
    walkTopLevelDecls<parsed_csl::GenericFuncOp>(
        unit, [&](parsed_csl::GenericFuncOp op) { generics.push_back(op); });
    for (auto op : generics)
      op->erase();

    // Step 2: Rename instantiated FuncOps back to original names where
    // unambiguous. Grouping is per parent scope (CompilationUnit or ModuleOp)
    // so that identically-named functions in different modules don't block
    // each other.

    // Group FuncOps by {parent scope, original_name}
    using ScopeKey = std::pair<Operation *, StringRef>;
    llvm::DenseMap<ScopeKey, SmallVector<parsed_csl::FuncOp>>
        scopedOriginalNameToFuncs;
    walkTopLevelDecls<parsed_csl::FuncOp>(unit, [&](parsed_csl::FuncOp funcOp) {
      Operation *scope = funcOp->getParentOp();
      scopedOriginalNameToFuncs[{scope, funcOp.getOriginalName()}].push_back(
          funcOp);
    });

    // Build rename map: only rename if exactly one instantiation in scope
    llvm::StringMap<std::string> renameMap; // current name -> original name
    for (auto &[key, funcs] : scopedOriginalNameToFuncs) {
      if (funcs.size() == 1)
        renameMap[funcs[0].getName()] = key.second.str();
    }

    if (renameMap.empty())
      return;

    // Update all FuncRefAttr and FuncAddrAttr references
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](csl::FuncRefAttr attr)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          auto it = renameMap.find(attr.getSymbolName().getValue());
          if (it == renameMap.end())
            return std::nullopt;
          return std::make_pair(
              csl::FuncRefAttr::get(
                  &getContext(), attr.getModuleIdentifier(),
                  FlatSymbolRefAttr::get(&getContext(), it->second),
                  attr.getType()),
              WalkResult::advance());
        });
    replacer.addReplacement(
        [&](csl::FuncAddrAttr attr)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          auto funcRef = attr.getFuncRef();
          auto it = renameMap.find(funcRef.getSymbolName().getValue());
          if (it == renameMap.end())
            return std::nullopt;
          auto newRef = csl::FuncRefAttr::get(
              &getContext(), funcRef.getModuleIdentifier(),
              FlatSymbolRefAttr::get(&getContext(), it->second),
              funcRef.getType());
          return std::make_pair(csl::FuncAddrAttr::get(newRef),
                                WalkResult::advance());
        });
    replacer.addReplacement([&](csl::CoroFrameType type)
                                -> std::optional<std::pair<Type, WalkResult>> {
      auto it = renameMap.find(type.getSymbolName().getValue());
      if (it == renameMap.end())
        return std::nullopt;
      return std::make_pair(
          csl::CoroFrameType::get(
              &getContext(), type.getModuleIdentifier(),
              FlatSymbolRefAttr::get(&getContext(), it->second)),
          WalkResult::advance());
    });
    replacer.addReplacement([&](csl::FutureType type)
                                -> std::optional<std::pair<Type, WalkResult>> {
      auto it = renameMap.find(type.getSymbolName().getValue());
      if (it == renameMap.end())
        return std::nullopt;
      return std::make_pair(
          csl::FutureType::get(
              &getContext(), type.getModuleIdentifier(),
              FlatSymbolRefAttr::get(&getContext(), it->second),
              type.getResultType()),
          WalkResult::advance());
    });
    replacer.recursivelyReplaceElementsIn(unit);

    // Rename the FuncOps themselves
    walkTopLevelDecls<parsed_csl::FuncOp>(unit, [&](parsed_csl::FuncOp funcOp) {
      auto it = renameMap.find(funcOp.getName());
      if (it != renameMap.end())
        funcOp.setSymNameAttr(StringAttr::get(&getContext(), it->second));
    });

    // Verify no references to generic functions remain
    verifyNoGenericFunctionRefs(unit);
  }

  /// Check that no FuncRefAttr with GenericFunctionType remains in attrs
  /// or result types. Generic functions must be fully resolved at all runtime
  /// use points during comptime evaluation.
  void verifyNoGenericFunctionRefs(csl::CompilationUnitOp unit) {
    unit.walk([&](Operation *op) {
      AttrTypeWalker walker;
      walker.addWalk([&](Attribute attr) {
        if (auto funcRef = mlir::dyn_cast<csl::FuncRefAttr>(attr)) {
          if (mlir::isa<csl::GenericFunctionType>(funcRef.getType())) {
            op->emitError() << "generic function '" << funcRef.getSymbolName()
                            << "' cannot be used as a runtime value; "
                            << "generic functions must be called directly";
            signalPassFailure();
          }
        }
      });
      walker.addWalk([&](Type type) {
        if (mlir::isa<csl::GenericFunctionType>(type)) {
          op->emitError() << "GenericFunctionType remains after "
                             "comptime evaluation (internal error)";
          signalPassFailure();
        }
      });
      for (NamedAttribute attr : op->getAttrs())
        walker.walk(attr.getValue());
      for (Type resultType : op->getResultTypes())
        walker.walk(resultType);
    });
  }
};

} // namespace

namespace mlir {
namespace parsed_csl {

std::unique_ptr<Pass> createFinalizeInstantiationPass() {
  return std::make_unique<FinalizeInstantiationPass>();
}

} // namespace parsed_csl
} // namespace mlir
