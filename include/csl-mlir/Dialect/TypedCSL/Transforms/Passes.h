//===- Passes.h - TypedCSL Transformation Passes ----------------*- C++ -*-===//
//
// Pass registration and declarations for TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_PASSES_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_PASSES_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::typed_csl {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"

struct FiberSizeSource;

/// Overload of `createFiberLoweringPass` that wires in a per-callee size
/// oracle.  In the two-round compile flow, round 1 passes a
/// `PlaceholderSizeSource` and round 2 passes a `ResolvedSizeSource`
/// backed by the ELF measurements from round 1.  Passing `nullptr` (or
/// using the auto-generated zero-arg overload) is equivalent to
/// `PlaceholderSizeSource{}`.
std::unique_ptr<Pass>
createFiberLoweringPass(const FiberLoweringPassOptions &options,
                        FiberSizeSource *sizeSource);

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_PASSES_H
