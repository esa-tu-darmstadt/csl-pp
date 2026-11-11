//===- EmitCSL.h - Emit CSL from TypedCSL IR -------------------*- C++ -*-===//
//
// Public API for emitting CSL source code from TypedCSL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_TARGET_CSL_EMITCSL_H
#define CSL_MLIR_TARGET_CSL_EMITCSL_H

#include "csl-mlir/Conversion/Passes.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace csl {

/// Emit CSL source code from a CompilationUnitOp containing TypedCSL
/// operations. Returns success if emission was successful.
LogicalResult emitCSL(CompilationUnitOp op, llvm::raw_ostream &os,
                      const std::string &varPrefix = "__csl_tmp",
                      unsigned indentWidth = 4);

/// Create a pass that emits CSL source code.
/// Note: The pass writes to llvm::outs() by default. For custom output streams,
/// use the emitCSL function directly.
std::unique_ptr<Pass> createEmitCSLPass();
std::unique_ptr<Pass> createEmitCSLPass(const EmitCSLPassOptions &options);

} // namespace csl
} // namespace mlir

#endif // CSL_MLIR_TARGET_CSL_EMITCSL_H
