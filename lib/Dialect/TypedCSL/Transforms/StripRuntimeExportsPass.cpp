//===- StripRuntimeExportsPass.cpp - Strip exports from runtime symbols ---===//
//
// Removes export attributes from runtime module functions and globals.
// Runtime .csl files mark symbols as `export` to survive DCE until compiler
// passes have emitted calls to them. This pass cleans up those exports.
//
// Stripping `export` leaves the functions ordinary, so cslc inlines all of
// them except `worker_entry` (which is a task entry and must stay addressable).
// Their cycles are then charged to whichever user function they were inlined
// into. `runtime-noinline` keeps them out of line instead, which is what a
// profiler needs to attribute cost to the runtime rather than to user code.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_STRIPRUNTIMEEXPORTSPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

struct StripRuntimeExportsPass
    : public typed_csl::impl::StripRuntimeExportsPassBase<
          StripRuntimeExportsPass> {
  using StripRuntimeExportsPassBase::StripRuntimeExportsPassBase;

  void runOnOperation() override {
    auto unit = getOperation();

    StringRef prefixes[] = {
        csl::CSLDialect::getFiberRuntimePrefix(),
        csl::CSLDialect::getAllocRuntimePrefix(),
    };

    auto hasRuntimePrefix = [&](StringRef name) {
      for (StringRef prefix : prefixes)
        if (name.starts_with(prefix))
          return true;
      return false;
    };

    unit.walk([&](FuncOp func) {
      if (!hasRuntimePrefix(func.getSymName()))
        return;
      func.setIsExport(false);
      func.setLinkname(std::nullopt);
      // Tasks are entry points: already out of line, and the verifier
      // rejects `noinline` on them.
      if (runtimeNoinline && !func.getIsTask()) {
        // `inline` and `noinline` are contradictory, and the runtime marks
        // some helpers inline; noinline is what was asked for, so it wins.
        func.setIsInline(false);
        func.setIsNoinline(true);
      }
    });

    unit.walk([&](GlobalOp global) {
      if (hasRuntimePrefix(global.getSymName())) {
        global.setIsExport(false);
        global.setLinkname(std::nullopt);
      }
    });
  }
};

} // namespace
