#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "csl-mlir/Conversion/Passes.h.inc"

} // namespace mlir