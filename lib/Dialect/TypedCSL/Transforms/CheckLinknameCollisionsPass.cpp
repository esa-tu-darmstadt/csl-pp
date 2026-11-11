//===- CheckLinknameCollisionsPass.cpp - Check linkname collisions -*- C++
//-*-===//
//
// Verifies that no two symbols share the same effective name.
// For symbols with a linkname, the linkname is the effective name.
// For symbols without a linkname, the symbol name is the effective name.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"

#include "llvm/ADT/StringMap.h"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_CHECKLINKNAMECOLLISIONSPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

struct CheckLinknameCollisionsPass
    : public typed_csl::impl::CheckLinknameCollisionsPassBase<
          CheckLinknameCollisionsPass> {
  using CheckLinknameCollisionsPassBase::CheckLinknameCollisionsPassBase;

  /// Returns the effective name for a symbol: linkname if present, otherwise
  /// the symbol name.
  static StringRef getEffectiveName(FuncOp func) {
    if (auto linkname = func.getLinkname())
      return *linkname;
    return func.getSymName();
  }

  static StringRef getEffectiveName(GlobalOp global) {
    if (auto linkname = global.getLinkname())
      return *linkname;
    return global.getSymName();
  }

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    bool hasCollision = false;

    // Map from effective name to the first op that claimed it.
    llvm::StringMap<Operation *> seen;

    for (auto &op : unit.getBody()->getOperations()) {
      StringRef effectiveName;
      if (auto func = dyn_cast<FuncOp>(op))
        effectiveName = getEffectiveName(func);
      else if (auto global = dyn_cast<GlobalOp>(op))
        effectiveName = getEffectiveName(global);
      else
        continue;

      auto [it, inserted] = seen.try_emplace(effectiveName, &op);
      if (!inserted) {
        op.emitError() << "symbol name collision: '" << effectiveName << "'";
        it->second->emitRemark() << "previous definition here";
        hasCollision = true;
      }
    }

    if (hasCollision)
      signalPassFailure();
  }
};

} // namespace