//===- LSPServer.cpp - CSL Language Server
//---------------------------------===//
//
// Protocol bridge: maps LSP JSON-RPC methods to CSLServer calls.
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "LspUtils.h"

#include "CSLServer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/LSP/Transport.h"
#include <optional>

#define DEBUG_TYPE "csl-lsp-server"

using namespace mlir::csl;

using llvm::lsp::Callback;
using llvm::lsp::CompletionContext;
using llvm::lsp::CompletionList;
using llvm::lsp::CompletionParams;
using llvm::lsp::DidChangeTextDocumentParams;
using llvm::lsp::DidCloseTextDocumentParams;
using llvm::lsp::DidOpenTextDocumentParams;
using llvm::lsp::DocumentSymbol;
using llvm::lsp::DocumentSymbolParams;
using llvm::lsp::Hover;
using llvm::lsp::InitializedParams;
using llvm::lsp::InitializeParams;
using llvm::lsp::JSONTransport;
using llvm::lsp::Location;
using llvm::lsp::Logger;
using llvm::lsp::MessageHandler;
using llvm::lsp::NoParams;
using llvm::lsp::OutgoingNotification;
using llvm::lsp::PublishDiagnosticsParams;
using llvm::lsp::SignatureHelp;
using llvm::lsp::TextDocumentPositionParams;
using llvm::lsp::TextDocumentSyncKind;
using llvm::lsp::URIForFile;

//===----------------------------------------------------------------------===//
// csl/getFileContent types
//===----------------------------------------------------------------------===//

struct GetFileContentParams {
  URIForFile uri;
};

struct GetFileContentResult {
  std::string content;
};

static bool fromJSON(const llvm::json::Value &value,
                     GetFileContentParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri);
}

static llvm::json::Value toJSON(const GetFileContentResult &result) {
  return llvm::json::Object{{"content", result.content}};
}

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

namespace {
struct LSPServer {
  LSPServer(CSLServer &server, JSONTransport &transport)
      : server(server), transport(transport) {}

  //===--------------------------------------------------------------------===//
  // Initialization

  void onInitialize(const InitializeParams &params,
                    Callback<llvm::json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Document Change

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Definitions

  void onGoToDefinition(const TextDocumentPositionParams &params,
                        Callback<std::vector<Location>> reply);

  //===--------------------------------------------------------------------===//
  // Hover

  void onHover(const TextDocumentPositionParams &params,
               Callback<std::optional<Hover>> reply);

  //===--------------------------------------------------------------------===//
  // Document Symbols

  void onDocumentSymbol(const DocumentSymbolParams &params,
                        Callback<std::vector<DocumentSymbol>> reply);

  //===--------------------------------------------------------------------===//
  // Completion

  void onCompletion(const CompletionParams &params,
                    Callback<CompletionList> reply);

  //===--------------------------------------------------------------------===//
  // Signature Help

  void onSignatureHelp(const TextDocumentPositionParams &params,
                       Callback<SignatureHelp> reply);

  //===--------------------------------------------------------------------===//
  // Custom Methods

  void onGetFileContent(const GetFileContentParams &params,
                        Callback<GetFileContentResult> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  CSLServer &server;
  JSONTransport &transport;

  /// Send diagnostics to the client. Editor-friendly hints are added
  /// centrally in mlirDiagToLspDiag (LspUtils.cpp).
  OutgoingNotification<PublishDiagnosticsParams> publishDiagnostics;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool shutdownRequestReceived = false;
};
} // namespace

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

void LSPServer::onInitialize(const InitializeParams &params,
                             Callback<llvm::json::Value> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onInitialize\n");
  llvm::json::Object serverCaps{
      {"textDocumentSync",
       llvm::json::Object{
           {"openClose", true},
           {"change", (int)TextDocumentSyncKind::Full},
       }},
      {"hoverProvider", true},
      {"definitionProvider", true},
      {"documentSymbolProvider", true},
      {"completionProvider",
       llvm::json::Object{
           {"triggerCharacters", llvm::json::Array{".", "@"}},
           {"resolveProvider", false},
       }},
      {"signatureHelpProvider",
       llvm::json::Object{
           {"triggerCharacters", llvm::json::Array{"(", ","}},
       }},
  };

  llvm::json::Object result{
      {{"serverInfo",
        llvm::json::Object{{"name", "csl-lsp-server"}, {"version", "0.0.1"}}},
       {"capabilities", std::move(serverCaps)}}};
  reply(std::move(result));
}
void LSPServer::onInitialized(const InitializedParams &) {
  LLVM_DEBUG(llvm::dbgs() << "onInitialized\n");
}
void LSPServer::onShutdown(const NoParams &, Callback<std::nullptr_t> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onShutdown\n");
  shutdownRequestReceived = true;
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  LLVM_DEBUG(llvm::dbgs() << "onDocumentDidOpen: uri='"
                          << params.textDocument.uri.uri() << "' file='"
                          << params.textDocument.uri.file() << "' scheme='"
                          << params.textDocument.uri.scheme()
                          << "' version=" << params.textDocument.version
                          << " contentLen=" << params.textDocument.text.size()
                          << "\n");
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.addDocument(params.textDocument.uri, params.textDocument.text,
                     params.textDocument.version, diagParams.diagnostics);
  LLVM_DEBUG(llvm::dbgs() << "onDocumentDidOpen: publishing "
                          << diagParams.diagnostics.size() << " diagnostics\n");
  publishDiagnostics(diagParams);
}

void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  LLVM_DEBUG(llvm::dbgs() << "onDocumentDidClose: uri='"
                          << params.textDocument.uri.uri() << "' file='"
                          << params.textDocument.uri.file() << "'\n");
  std::optional<int64_t> version =
      server.removeDocument(params.textDocument.uri);
  if (!version) {
    LLVM_DEBUG(llvm::dbgs() << "onDocumentDidClose: removeDocument returned "
                               "nullopt, not clearing diagnostics\n");
    return;
  }

  // Clear diagnostics for this document.
  LLVM_DEBUG(llvm::dbgs() << "onDocumentDidClose: clearing diagnostics for "
                             "version "
                          << *version << "\n");
  publishDiagnostics(
      PublishDiagnosticsParams(params.textDocument.uri, *version));
}

void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  LLVM_DEBUG(llvm::dbgs() << "onDocumentDidChange: uri='"
                          << params.textDocument.uri.uri()
                          << "' version=" << params.textDocument.version
                          << " changes=" << params.contentChanges.size()
                          << "\n");
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.updateDocument(params.textDocument.uri, params.contentChanges,
                        params.textDocument.version, diagParams.diagnostics);
  LLVM_DEBUG(llvm::dbgs() << "onDocumentDidChange: publishing "
                          << diagParams.diagnostics.size() << " diagnostics\n");
  publishDiagnostics(diagParams);
}

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

void LSPServer::onGoToDefinition(const TextDocumentPositionParams &params,
                                 Callback<std::vector<Location>> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onGoToDefinition: uri='"
                          << params.textDocument.uri.uri()
                          << "' pos=" << params.position.line << ":"
                          << params.position.character << "\n");
  std::vector<Location> locations;
  server.getLocationsOf(params.textDocument.uri, params.position, locations);
  LLVM_DEBUG(llvm::dbgs() << "onGoToDefinition: returning " << locations.size()
                          << " location(s)\n");
  reply(std::move(locations));
}

//===----------------------------------------------------------------------===//
// Hover
//===----------------------------------------------------------------------===//

void LSPServer::onHover(const TextDocumentPositionParams &params,
                        Callback<std::optional<Hover>> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onHover: uri='" << params.textDocument.uri.uri()
                          << "' pos=" << params.position.line << ":"
                          << params.position.character << "\n");
  auto result = server.findHover(params.textDocument.uri, params.position);
  LLVM_DEBUG(llvm::dbgs() << "onHover: " << (result ? "found" : "none")
                          << "\n");
  reply(std::move(result));
}

//===----------------------------------------------------------------------===//
// Document Symbols
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentSymbol(const DocumentSymbolParams &params,
                                 Callback<std::vector<DocumentSymbol>> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onDocumentSymbol: uri='"
                          << params.textDocument.uri.uri() << "'\n");
  std::vector<DocumentSymbol> symbols;
  server.findDocumentSymbols(params.textDocument.uri, symbols);
  LLVM_DEBUG(llvm::dbgs() << "onDocumentSymbol: returning " << symbols.size()
                          << " symbol(s)\n");
  reply(std::move(symbols));
}

//===----------------------------------------------------------------------===//
// Completion
//===----------------------------------------------------------------------===//

void LSPServer::onCompletion(const CompletionParams &params,
                             Callback<CompletionList> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onCompletion: uri='"
                          << params.textDocument.uri.uri()
                          << "' pos=" << params.position.line << ":"
                          << params.position.character << "\n");
  reply(server.getCodeCompletion(params.textDocument.uri, params.position,
                                 params.context));
}

//===----------------------------------------------------------------------===//
// Signature Help
//===----------------------------------------------------------------------===//

void LSPServer::onSignatureHelp(const TextDocumentPositionParams &params,
                                Callback<SignatureHelp> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onSignatureHelp: uri='"
                          << params.textDocument.uri.uri()
                          << "' pos=" << params.position.line << ":"
                          << params.position.character << "\n");
  reply(server.getSignatureHelp(params.textDocument.uri, params.position));
}

//===----------------------------------------------------------------------===//
// Custom Methods
//===----------------------------------------------------------------------===//

void LSPServer::onGetFileContent(const GetFileContentParams &params,
                                 Callback<GetFileContentResult> reply) {
  LLVM_DEBUG(llvm::dbgs() << "onGetFileContent: uri='" << params.uri.uri()
                          << "' file='" << params.uri.file() << "' scheme='"
                          << params.uri.scheme() << "'\n");
  auto content = server.getFileContent(params.uri);
  if (content) {
    LLVM_DEBUG(llvm::dbgs() << "onGetFileContent: returning " << content->size()
                            << " bytes\n");
    reply(GetFileContentResult{std::move(*content)});
  } else {
    LLVM_DEBUG(llvm::dbgs() << "onGetFileContent: file not found\n");
    reply(llvm::make_error<llvm::StringError>("file not found: " +
                                                  params.uri.uri().str(),
                                              llvm::inconvertibleErrorCode()));
  }
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

llvm::LogicalResult mlir::csl::runCslLSPServer(CSLServer &server,
                                               JSONTransport &transport) {
  LLVM_DEBUG(llvm::dbgs() << "runCslLSPServer: starting\n");
  // Register the csl-sdk URI scheme so URIForFile can parse csl-sdk:// URIs.
  URIForFile::registerSupportedScheme(lsp_utils::kCslSdkScheme);

  LSPServer lspServer(server, transport);
  MessageHandler messageHandler(transport);

  // Initialization
  messageHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  messageHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
  messageHandler.notification("textDocument/didOpen", &lspServer,
                              &LSPServer::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", &lspServer,
                              &LSPServer::onDocumentDidClose);
  messageHandler.notification("textDocument/didChange", &lspServer,
                              &LSPServer::onDocumentDidChange);

  // Definitions
  messageHandler.method("textDocument/definition", &lspServer,
                        &LSPServer::onGoToDefinition);

  // Hover
  messageHandler.method("textDocument/hover", &lspServer, &LSPServer::onHover);

  // Document Symbols
  messageHandler.method("textDocument/documentSymbol", &lspServer,
                        &LSPServer::onDocumentSymbol);

  // Completion
  messageHandler.method("textDocument/completion", &lspServer,
                        &LSPServer::onCompletion);

  // Signature Help
  messageHandler.method("textDocument/signatureHelp", &lspServer,
                        &LSPServer::onSignatureHelp);

  // Custom Methods
  messageHandler.method("csl/getFileContent", &lspServer,
                        &LSPServer::onGetFileContent);

  // Diagnostics
  lspServer.publishDiagnostics =
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics");

  // Wire the diagnostics callback so CSLServer can proactively publish
  // diagnostics when a layout change affects already-open PE program files.
  server.setDiagnosticsCallback(
      [&lspServer](const llvm::lsp::URIForFile &uri,
                   llvm::ArrayRef<llvm::lsp::Diagnostic> diags) {
        PublishDiagnosticsParams params(uri, /*version=*/0);
        params.diagnostics.assign(diags.begin(), diags.end());
        lspServer.publishDiagnostics(params);
      });

  // Run the main loop of the transport.
  if (llvm::Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    return failure();
  }
  return success(lspServer.shutdownRequestReceived);
}
