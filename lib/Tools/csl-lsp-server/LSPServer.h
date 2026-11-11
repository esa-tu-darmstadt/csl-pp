//===- LSPServer.h - CSL LSP Server -----------------------------*- C++ -*-===//
//
// Protocol bridge between JSON-RPC and the CSLServer.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TOOLS_CSL_LSP_SERVER_LSPSERVER_H
#define LIB_TOOLS_CSL_LSP_SERVER_LSPSERVER_H

#include <memory>

namespace llvm {
struct LogicalResult;
namespace lsp {
class JSONTransport;
} // namespace lsp
} // namespace llvm

namespace mlir::csl {

class CSLServer;

/// Run the main loop of the LSP server using the given CSL server and
/// transport.
llvm::LogicalResult runCslLSPServer(CSLServer &server,
                                    llvm::lsp::JSONTransport &transport);

} // namespace mlir::csl

#endif // LIB_TOOLS_CSL_LSP_SERVER_LSPSERVER_H
