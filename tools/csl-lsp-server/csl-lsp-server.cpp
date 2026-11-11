//===- csl-lsp-server.cpp - CSL Language Server main
//-----------------------===//
//
// Entry point for the CSL Language Server.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Tools/csl-lsp-server/CslLspServerMain.h"

int main(int argc, char **argv) {
  return llvm::failed(mlir::csl::CslLspServerMain(argc, argv));
}
