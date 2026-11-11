//===- CslLspServerMain.h - CSL Language Server main ------------*- C++ -*-===//
//
// Public entry point for the CSL Language Server.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_TOOLS_CSL_LSP_SERVER_CSLLSPSERVERMAIN_H
#define CSL_MLIR_TOOLS_CSL_LSP_SERVER_CSLLSPSERVERMAIN_H

#include "llvm/Support/LogicalResult.h"

namespace mlir::csl {

/// Entry point for tools like `csl-lsp-server`.
llvm::LogicalResult CslLspServerMain(int argc, char **argv);

} // namespace mlir::csl

#endif // CSL_MLIR_TOOLS_CSL_LSP_SERVER_CSLLSPSERVERMAIN_H
