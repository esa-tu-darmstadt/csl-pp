//===- ValidateParsedCSL.cpp - ParsedCSL IR Validation ----------*- C++ -*-===//
//
// This file implements a validation pass for ParsedCSL IR before lowering.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "validate-parsed-csl"

using namespace mlir;
using namespace mlir::parsed_csl;

namespace mlir {
namespace parsed_csl {

#define GEN_PASS_DEF_VALIDATEPARSEDCSLPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

} // namespace parsed_csl
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ValidateParsedCSLPass
    : public parsed_csl::impl::ValidateParsedCSLPassBase<
          ValidateParsedCSLPass> {

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    bool hasErrors = false;

    // Check for remaining comptime variables — they should have been
    // evaluated and DCE'd by the canonicalizer after ComptimeEvalPass.
    unit.walk([&](parsed_csl::VariableOp varOp) {
      if (varOp.getIsComptime()) {
        varOp.emitError() << "comptime variable remains after "
                             "comptime evaluation (internal error)";
        hasErrors = true;
      }
    });

    // `@await` and `suspend` only mean anything inside an async function:
    // they name a point where the enclosing coroutine yields. Outside one
    // there is no coroutine to yield, and the coroutine lowering has nothing
    // to rewrite -- the op would survive untouched and be emitted verbatim as
    // CSL that cslc cannot parse. Rejecting it here keeps the failure at the
    // user's source location.
    auto requireAsyncContext = [&](Operation *op, StringRef what) {
      auto funcOp = op->getParentOfType<parsed_csl::FuncOp>();
      if (funcOp && funcOp.getIsAsync())
        return;
      InFlightDiagnostic diag = op->emitError()
                                << what << " is only allowed inside an async "
                                           "function or async task";
      if (funcOp)
        diag.attachNote(funcOp.getLoc())
            << "enclosing function '" << funcOp.getSymName()
            << "' is not declared async";
      hasErrors = true;
    };

    unit.walk([&](parsed_csl::SuspendOp op) {
      requireAsyncContext(op, "'suspend'");
    });
    unit.walk([&](parsed_csl::BuiltinCallOp op) {
      StringRef name = op.getBuiltinName();
      if (name == csl::stringifyBuiltinKind(csl::BuiltinKind::Await) ||
          name == csl::stringifyBuiltinKind(csl::BuiltinKind::AwaitParallel))
        requireAsyncContext(op, ("'@" + name + "'").str());
    });

    if (hasErrors)
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

std::unique_ptr<Pass> createValidateParsedCSLPass() {
  return std::make_unique<ValidateParsedCSLPass>();
}

} // namespace parsed_csl
} // namespace mlir
