//===- CSLServer.h - CSL Language Server -------------------------*- C++
//-*-===//
//
// Business logic for the CSL language server: document management, parsing,
// analysis, and query handling. No LSP protocol knowledge.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TOOLS_CSL_LSP_SERVER_CSLSERVER_H
#define LIB_TOOLS_CSL_LSP_SERVER_CSLSERVER_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "llvm/Support/LSP/Protocol.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::csl {

using llvm::lsp::Diagnostic;
using llvm::lsp::DocumentSymbol;
using llvm::lsp::Hover;
using llvm::lsp::Location;
using llvm::lsp::Position;
using llvm::lsp::TextDocumentContentChangeEvent;
using llvm::lsp::URIForFile;

/// This class implements all of the CSL related functionality necessary for a
/// language server. This class allows keeping the CSL-specific logic separate
/// from the logic that involves LSP server/client communication.
class CSLServer {
public:
  struct Options {
    /// CSL flags (import paths, target arch, etc.)
    CslFlags flags;

    /// When true, filter out error diagnostics that strictly contain a smaller
    /// error diagnostic, showing only the innermost errors.
    bool filterOverlappingErrors = false;
  };

  /// Callback type for proactively publishing diagnostics (e.g., when a layout
  /// change affects an already-open PE program file).
  using DiagnosticsCallback =
      std::function<void(const URIForFile &, llvm::ArrayRef<Diagnostic>)>;

  CSLServer(const Options &options);
  ~CSLServer();

  /// Set the callback used to publish diagnostics proactively.
  /// The LSP server layer calls this to wire up the publishDiagnostics
  /// notification.
  void setDiagnosticsCallback(DiagnosticsCallback cb);

  /// Add a document with the provided version at the given URI.
  void addDocument(const URIForFile &uri, llvm::StringRef contents,
                   int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Update the document at the given URI (full-text sync).
  void updateDocument(const URIForFile &uri,
                      llvm::ArrayRef<TextDocumentContentChangeEvent> changes,
                      int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given URI.
  std::optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return locations of the definition at the given position.
  void getLocationsOf(const URIForFile &uri, const Position &pos,
                      std::vector<Location> &locations);

  /// Find hover information for the given position.
  std::optional<Hover> findHover(const URIForFile &uri, const Position &pos);

  /// Find all document symbols in the given file.
  void findDocumentSymbols(const URIForFile &uri,
                           std::vector<DocumentSymbol> &symbols);

  /// Get code completions at the given position.
  llvm::lsp::CompletionList
  getCodeCompletion(const URIForFile &uri, const Position &completePos,
                    const llvm::lsp::CompletionContext &ctx);

  /// Get signature help at the given position.
  llvm::lsp::SignatureHelp getSignatureHelp(const URIForFile &uri,
                                            const Position &pos);

  /// Get the text content of a file by URI. Used for `csl-sdk://` virtual
  /// documents served via the `csl/getFileContent` custom LSP request.
  std::optional<std::string> getFileContent(const URIForFile &uri);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::csl

#endif // LIB_TOOLS_CSL_LSP_SERVER_CSLSERVER_H
