//===- CSLServer.cpp - CSL Language Server
//---------------------------------===//
//
// Core implementation: document management, parsing, analysis, and queries.
// Supports both independent files and layout-driven multi-file contexts.
//
//===----------------------------------------------------------------------===//

#include "CSLServer.h"
#include "CompilationContext.h"
#include "LspUtils.h"

#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "csl-lsp-server"

// Avoid `using namespace mlir` to prevent ambiguity between mlir::Diagnostic /
// mlir::Location and the LSP equivalents.
namespace lsp = llvm::lsp;

//===----------------------------------------------------------------------===//
// CSLServer::Impl
//===----------------------------------------------------------------------===//

/// Convert a URI to the internal lookup key. For csl-sdk:// URIs (e.g.
/// csl-sdk:///memcpy/memcpy.csl), returns the display name "<memcpy/memcpy>".
/// For file:// URIs, returns the absolute path as-is.
static std::string getInternalKey(const lsp::URIForFile &uri) {
  if (uri.scheme() == mlir::csl::lsp_utils::kCslSdkScheme) {
    llvm::StringRef path = uri.file().drop_front(1); // strip leading /
    // Strip the .csl extension added by uriForPath.
    if (path.ends_with(".csl"))
      path = path.drop_back(4);
    std::string key = ("<" + path + ">").str();
    LLVM_DEBUG(llvm::dbgs() << "getInternalKey: csl-sdk uri='" << uri.uri()
                            << "' -> key='" << key << "'\n");
    return key;
  }
  LLVM_DEBUG(llvm::dbgs() << "getInternalKey: uri='" << uri.uri()
                          << "' -> key='" << uri.file() << "'\n");
  return uri.file().str();
}

struct mlir::csl::CSLServer::Impl {
  explicit Impl(const Options &options) : options(options) {
    LLVM_DEBUG(llvm::dbgs() << "CSLServer::Impl: initializing\n");
    lsp_utils::initLspContext(mlirContext);
    // Create the system module cache once (opens SIF container).
    systemModuleCache = SystemModuleCache::create(options.flags);
    // Create a persistent resolver for reading system module content from SIF.
    if (!options.flags.systemModuleSif.empty())
      systemResolver = std::make_shared<CslModuleResolver>("", options.flags,
                                                           systemModuleCache);
    LLVM_DEBUG({
      llvm::dbgs() << "CSLServer::Impl: systemModuleSif='"
                   << options.flags.systemModuleSif << "'\n";
      llvm::dbgs() << "CSLServer::Impl: importPaths=[";
      for (const auto &p : options.flags.importPaths)
        llvm::dbgs() << "'" << p << "', ";
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "CSLServer::Impl: systemImportPaths=[";
      for (const auto &p : options.flags.systemImportPaths)
        llvm::dbgs() << "'" << p << "', ";
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "CSLServer::Impl: filterOverlappingErrors="
                   << options.filterOverlappingErrors << "\n";
    });
  }

  Options options;

  /// Single shared MLIRContext for all files.
  mlir::MLIRContext mlirContext;

  /// Independent file states (not part of any CompilationContext).
  llvm::StringMap<std::unique_ptr<FileState>> independentFiles;

  /// Compilation contexts (keyed by layout file absolute path).
  llvm::StringMap<std::unique_ptr<CompilationContext>> contexts;

  /// Reverse map: any known file path -> which context owns it.
  llvm::StringMap<CompilationContext *> fileToContext;

  /// Track which files the user has explicitly opened (by absolute path).
  llvm::StringSet<> userOpenedFiles;

  /// In-memory contents for open files (by absolute path).
  llvm::StringMap<std::string> fileContents;

  /// Map from URI string to document version.
  llvm::StringMap<int64_t> fileVersions;

  /// Callback for publishing diagnostics proactively.
  mlir::csl::CSLServer::DiagnosticsCallback diagnosticsCallback;

  /// Cached SIF VFS shared across all resolver instances.
  std::shared_ptr<SystemModuleCache> systemModuleCache;

  /// Persistent resolver for reading system module content from SIF.
  std::shared_ptr<CslModuleResolver> systemResolver;

  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  /// Parse and analyze a standalone CSL file.
  std::unique_ptr<FileState>
  parseIndependentFile(llvm::StringRef absPath, llvm::StringRef contents,
                       std::vector<lsp::Diagnostic> &diagnostics) {
    LLVM_DEBUG(llvm::dbgs() << "parseIndependentFile: absPath='" << absPath
                            << "' contentLen=" << contents.size() << "\n");
    auto state = std::make_unique<FileState>();
    state->filePath = absPath.str();

    llvm::StringMap<std::vector<lsp::Diagnostic>> allDiags;
    auto diagHandler = lsp_utils::createRoutingDiagHandler(
        &mlirContext, absPath, allDiags, options.filterOverlappingErrors);

    auto paramBindings = options.flags.toBindings(&mlirContext);
    // The file being parsed is always "open" — eagerly evaluate it.
    std::string ownPath = absPath.str();
    auto eagerPredicate = [ownPath](llvm::StringRef path) {
      return path == ownPath;
    };
    parseAndAnalyze(&mlirContext, *state, contents, options.flags,
                    paramBindings, /*overrideBuffers=*/{}, systemModuleCache,
                    eagerPredicate);

    // Extract diagnostics for this file.
    auto it = allDiags.find(absPath);
    if (it != allDiags.end())
      diagnostics = std::move(it->second);

    LLVM_DEBUG({
      llvm::dbgs() << "parseIndependentFile: done, hasLayoutBlock="
                   << state->hasLayoutBlock
                   << " parsedOk=" << state->parsedSuccessfully
                   << " diags=" << diagnostics.size() << "\n";
      llvm::dbgs() << "parseIndependentFile: all diag keys:";
      for (const auto &[path, diags] : allDiags)
        llvm::dbgs() << "\n  '" << path << "' (" << diags.size() << " diags)";
      llvm::dbgs() << "\n";
    });
    return state;
  }

  /// Resolve a DocumentView for the given file, trying context first, then
  /// independent files.
  std::optional<DocumentView> resolveView(llvm::StringRef absPath) {
    LLVM_DEBUG(llvm::dbgs() << "resolveView: absPath='" << absPath << "'\n");
    auto ctxIt = fileToContext.find(absPath);
    if (ctxIt != fileToContext.end()) {
      LLVM_DEBUG(llvm::dbgs() << "resolveView: found in context (layout='"
                              << ctxIt->second->getLayoutPath() << "')\n");
      return ctxIt->second->getView(absPath);
    }

    auto fileIt = independentFiles.find(absPath);
    if (fileIt != independentFiles.end()) {
      LLVM_DEBUG(llvm::dbgs() << "resolveView: found as independent file\n");
      return fileIt->second->toView();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "resolveView: NOT FOUND\n";
      llvm::dbgs() << "resolveView: known context files:";
      for (const auto &[path, ctx] : fileToContext)
        llvm::dbgs() << "\n  '" << path << "' (layout='" << ctx->getLayoutPath()
                     << "')";
      llvm::dbgs() << "\nresolveView: known independent files:";
      for (const auto &[path, _] : independentFiles)
        llvm::dbgs() << "\n  '" << path << "'";
      llvm::dbgs() << "\n";
    });
    return std::nullopt;
  }

  /// Rebuild the fileToContext mapping from all contexts.
  void rebuildFileToContextMapping() {
    LLVM_DEBUG(llvm::dbgs() << "rebuildFileToContextMapping: "
                            << contexts.size() << " context(s)\n");
    fileToContext.clear();
    for (auto &[layoutPath, ctx] : contexts) {
      llvm::SmallVector<std::string> paths;
      ctx->getKnownFilePaths(paths);
      LLVM_DEBUG(llvm::dbgs() << "  context layout='" << layoutPath << "' -> "
                              << paths.size() << " file(s):\n");
      for (const auto &path : paths) {
        LLVM_DEBUG(llvm::dbgs() << "    '" << path << "'\n");
        fileToContext[path] = ctx.get();
      }
    }
  }

  /// Publish diagnostics for a file (uses callback if available).
  void publishDiagnostics(llvm::StringRef absPath,
                          llvm::ArrayRef<lsp::Diagnostic> diags) {
    if (!diagnosticsCallback)
      return;
    auto uri = lsp_utils::uriForPath(absPath);
    if (!uri) {
      LLVM_DEBUG(llvm::dbgs() << "publishDiagnostics: failed to create URI "
                                 "for '"
                              << absPath << "'\n");
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "publishDiagnostics: path='" << absPath << "' uri='"
               << uri->uri() << "' count=" << diags.size() << "\n");
    diagnosticsCallback(*uri, diags);
  }

  /// Publish all stored diagnostics from a CompilationContext, except for
  /// `excludePath` (whose diagnostics are returned via the output parameter).
  void publishContextDiagnostics(CompilationContext *ctx,
                                 llvm::StringRef excludePath,
                                 std::vector<lsp::Diagnostic> &diagnostics) {
    const auto &allDiags = ctx->getDiagnostics();

    auto diagsIt = allDiags.find(excludePath);
    if (diagsIt != allDiags.end())
      diagnostics = diagsIt->second;

    for (const auto &[path, diags] : allDiags) {
      if (path != excludePath)
        publishDiagnostics(path, diags);
    }
  }

  /// Get file contents: from in-memory fileContents if available, else disk.
  std::string getContentsForFile(llvm::StringRef absPath) {
    auto it = fileContents.find(absPath);
    if (it != fileContents.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getContentsForFile: '" << absPath << "' -> in-memory ("
                 << it->second.size() << " bytes)\n");
      return it->second;
    }
    auto bufOrErr = llvm::MemoryBuffer::getFile(absPath);
    if (bufOrErr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getContentsForFile: '" << absPath << "' -> disk ("
                 << (*bufOrErr)->getBufferSize() << " bytes)\n");
      return (*bufOrErr)->getBuffer().str();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "getContentsForFile: '" << absPath
               << "' -> NOT FOUND: " << bufOrErr.getError().message() << "\n");
    return {};
  }

  /// Create a CompilationContext for a layout file, compile PE programs,
  /// and update mappings. Returns the context pointer.
  CompilationContext *createContext(llvm::StringRef layoutAbsPath,
                                    llvm::StringRef contents) {
    LLVM_DEBUG({
      llvm::dbgs() << "createContext: layoutAbsPath='" << layoutAbsPath
                   << "' contentLen=" << contents.size() << "\n";
      llvm::dbgs() << "createContext: userOpenedFiles:";
      for (const auto &f : userOpenedFiles)
        llvm::dbgs() << "\n  '" << f.getKey() << "'";
      llvm::dbgs() << "\n";
    });
    auto ctx = std::make_unique<CompilationContext>(
        mlirContext, options.flags, options.filterOverlappingErrors,
        systemModuleCache);
    ctx->setOpenFiles(userOpenedFiles);
    ctx->analyzeLayout(layoutAbsPath, contents);
    CompilationContext *ptr = ctx.get();
    contexts[layoutAbsPath] = std::move(ctx);
    rebuildFileToContextMapping();
    LLVM_DEBUG(llvm::dbgs() << "createContext: done, " << fileToContext.size()
                            << " files in context mapping\n");
    return ptr;
  }

  /// Transition files from independent to contextual and publish diagnostics
  /// for all known files in the context.
  void transitionToContextual(CompilationContext *ctx) {
    LLVM_DEBUG(llvm::dbgs() << "transitionToContextual: layout='"
                            << ctx->getLayoutPath() << "'\n");
    const auto &allDiags = ctx->getDiagnostics();
    llvm::StringSet<> published;
    for (const auto &[path, diags] : allDiags) {
      bool wasOpen = userOpenedFiles.count(path);
      if (wasOpen)
        independentFiles.erase(path);
      LLVM_DEBUG(llvm::dbgs()
                 << "  path='" << path << "' diags=" << diags.size()
                 << " wasOpen=" << wasOpen << "\n");
      publishDiagnostics(path, diags);
      published.insert(path);
    }

    // For user-opened files that are now in the context but had no
    // diagnostics emitted, publish empty diagnostics to clear stale ones
    // from the previous independent parse.
    llvm::SmallVector<std::string> knownPaths;
    ctx->getKnownFilePaths(knownPaths);
    for (const auto &path : knownPaths) {
      if (!published.count(path) && userOpenedFiles.count(path)) {
        independentFiles.erase(path);
        LLVM_DEBUG(llvm::dbgs()
                   << "  clearing stale diags: path='" << path << "'\n");
        publishDiagnostics(path, {});
      }
    }
  }

  /// Revert contextual files back to independent mode. User-opened files are
  /// re-parsed independently; other files get their diagnostics cleared.
  void revertToIndependent(CompilationContext *ctx) {
    LLVM_DEBUG(llvm::dbgs() << "revertToIndependent: layout='"
                            << ctx->getLayoutPath() << "'\n");
    llvm::SmallVector<std::string> knownPaths;
    ctx->getKnownFilePaths(knownPaths);
    for (const auto &path : knownPaths) {
      bool isOpen = userOpenedFiles.count(path);
      LLVM_DEBUG(llvm::dbgs()
                 << "  path='" << path << "' isOpen=" << isOpen << "\n");
      if (isOpen) {
        auto contentsIt = fileContents.find(path);
        if (contentsIt == fileContents.end()) {
          LLVM_DEBUG(llvm::dbgs() << "    no in-memory contents, skipping\n");
          continue;
        }

        std::vector<lsp::Diagnostic> diags;
        independentFiles[path] =
            parseIndependentFile(path, contentsIt->second, diags);
        publishDiagnostics(path, diags);
      } else {
        // Clear diagnostics for non-opened files that were part of the context.
        LLVM_DEBUG(llvm::dbgs() << "    clearing diagnostics\n");
        publishDiagnostics(path, {});
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// CSLServer
//===----------------------------------------------------------------------===//

mlir::csl::CSLServer::CSLServer(const Options &options)
    : impl(std::make_unique<Impl>(options)) {}

mlir::csl::CSLServer::~CSLServer() = default;

void mlir::csl::CSLServer::setDiagnosticsCallback(DiagnosticsCallback cb) {
  impl->diagnosticsCallback = std::move(cb);
}

void mlir::csl::CSLServer::addDocument(
    const lsp::URIForFile &uri, llvm::StringRef contents, int64_t version,
    std::vector<lsp::Diagnostic> &diagnostics) {
  std::string key = getInternalKey(uri);
  LLVM_DEBUG(llvm::dbgs() << "addDocument: key='" << key
                          << "' version=" << version
                          << " contentLen=" << contents.size() << "\n");
  impl->fileVersions[uri.uri()] = version;
  impl->userOpenedFiles.insert(key);
  impl->fileContents[key] = contents.str();

  // Check if this file is already in a compilation context.
  auto ctxIt = impl->fileToContext.find(key);
  if (ctxIt != impl->fileToContext.end()) {
    CompilationContext *ctx = ctxIt->second;
    LLVM_DEBUG(llvm::dbgs() << "addDocument: file is in context (layout='"
                            << ctx->getLayoutPath() << "')\n");

    // If this is an imported module that wasn't previously open, re-analyze
    // its owner so that functions in this file get eagerly evaluated.
    llvm::StringRef ownerPath = ctx->getOwnerOfImportedModule(key);
    LLVM_DEBUG(llvm::dbgs()
               << "addDocument: ownerOfImportedModule='" << ownerPath << "'\n");
    if (!ownerPath.empty()) {
      ctx->setOpenFiles(impl->userOpenedFiles);
      llvm::StringRef layoutPath = ctx->getLayoutPath();
      LLVM_DEBUG(llvm::dbgs()
                 << "addDocument: re-analyzing imported module, "
                    "ownerPath='"
                 << ownerPath << "' layoutPath='" << layoutPath << "'\n");
      if (ownerPath == layoutPath) {
        std::string layoutContents = impl->getContentsForFile(layoutPath);
        ctx->analyzeLayout(layoutPath, layoutContents);
      } else {
        std::string ownerContents = impl->getContentsForFile(ownerPath);
        ctx->reanalyzePEProgram(ownerPath, ownerContents);
      }
      impl->rebuildFileToContextMapping();
      impl->publishContextDiagnostics(ctx, key, diagnostics);
      LLVM_DEBUG(llvm::dbgs() << "addDocument: done (re-analyzed imported "
                                 "module), diags="
                              << diagnostics.size() << "\n");
      return;
    }

    // Return stored diagnostics so the client doesn't clear them on open.
    auto diags = ctx->getDiagnosticsForFile(key);
    diagnostics.assign(diags.begin(), diags.end());
    LLVM_DEBUG(llvm::dbgs() << "addDocument: returning stored context diags="
                            << diagnostics.size() << "\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "addDocument: file not in any context, parsing "
                             "independently\n");
  // Parse independently.
  auto state = impl->parseIndependentFile(key, contents, diagnostics);

  // If this file has a layout block, create a CompilationContext.
  if (state->hasLayoutBlock) {
    LLVM_DEBUG(llvm::dbgs() << "addDocument: file has layout block, creating "
                               "compilation context\n");
    auto *ctx = impl->createContext(key, contents);

    // Layout diagnostics go to the caller.
    auto diags = ctx->getDiagnosticsForFile(key);
    diagnostics.assign(diags.begin(), diags.end());

    // Publish diagnostics for all other files in the context.
    impl->transitionToContextual(ctx);
    LLVM_DEBUG(llvm::dbgs() << "addDocument: done (layout context created), "
                               "diags="
                            << diagnostics.size() << "\n");
    return;
  }

  // Store as independent file.
  LLVM_DEBUG(llvm::dbgs() << "addDocument: storing as independent file\n");
  impl->independentFiles[key] = std::move(state);
}

void mlir::csl::CSLServer::updateDocument(
    const lsp::URIForFile &uri,
    llvm::ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<lsp::Diagnostic> &diagnostics) {
  if (changes.empty())
    return;
  llvm::StringRef newContents = changes.back().text;
  std::string key = getInternalKey(uri);

  impl->fileVersions[uri.uri()] = version;
  impl->fileContents[key] = newContents.str();

  auto ctxIt = impl->fileToContext.find(key);
  LLVM_DEBUG(llvm::dbgs() << "updateDocument: key=" << key << " inContext="
                          << (ctxIt != impl->fileToContext.end())
                          << " contentLen=" << newContents.size() << "\n");
  if (ctxIt != impl->fileToContext.end()) {
    CompilationContext *ctx = ctxIt->second;
    ctx->setOpenFiles(impl->userOpenedFiles);
    llvm::StringRef layoutPath = ctx->getLayoutPath();

    if (key == layoutPath) {
      // Editing the layout file: rebuild entire context.
      ctx->analyzeLayout(key, newContents);
      impl->rebuildFileToContextMapping();
      impl->publishContextDiagnostics(ctx, key, diagnostics);
      return;
    }

    // Check if it's an imported module.
    llvm::StringRef ownerPath = ctx->getOwnerOfImportedModule(key);
    LLVM_DEBUG(llvm::dbgs() << "updateDocument: isLayout=false ownerPath='"
                            << ownerPath << "'\n");
    if (!ownerPath.empty()) {
      llvm::StringMap<std::string> overrideBuffers;
      overrideBuffers[key] = newContents.str();

      if (ownerPath == layoutPath) {
        // Editing a layout-imported module: rebuild entire context.
        std::string layoutContents = impl->getContentsForFile(layoutPath);

        LLVM_DEBUG(llvm::dbgs()
                   << "updateDocument: re-analyzing layout-imported module\n"
                   << "  modulePath=" << key << "\n"
                   << "  layoutPath=" << layoutPath << "\n"
                   << "  layoutContentLen=" << layoutContents.size() << "\n"
                   << "  overrideContentLen=" << newContents.size() << "\n");
        ctx->analyzeLayout(layoutPath, layoutContents, overrideBuffers);
      } else {
        // Editing a PE-program-imported module: re-analyze the owning PE
        // program.
        std::string ownerContents = impl->getContentsForFile(ownerPath);

        LLVM_DEBUG(llvm::dbgs()
                   << "updateDocument: re-analyzing PE-imported module\n"
                   << "  modulePath=" << key << "\n"
                   << "  ownerPath=" << ownerPath << "\n"
                   << "  ownerContentLen=" << ownerContents.size() << "\n"
                   << "  overrideContentLen=" << newContents.size() << "\n");
        ctx->reanalyzePEProgram(ownerPath, ownerContents, overrideBuffers);
      }
      impl->rebuildFileToContextMapping();

      LLVM_DEBUG({
        llvm::dbgs() << "updateDocument: stored diags keys:";
        for (const auto &[path, diags] : ctx->getDiagnostics())
          llvm::dbgs() << "\n  '" << path << "' (" << diags.size() << " diags)";
        llvm::dbgs() << "\n";
      });
      impl->publishContextDiagnostics(ctx, key, diagnostics);
      return;
    }

    // It's a PE program: re-analyze with stored params.
    ctx->reanalyzePEProgram(key, newContents);
    impl->rebuildFileToContextMapping();
    impl->publishContextDiagnostics(ctx, key, diagnostics);
    return;
  }

  // Not in a context: re-parse independently.
  addDocument(uri, newContents, version, diagnostics);
}

std::optional<int64_t>
mlir::csl::CSLServer::removeDocument(const lsp::URIForFile &uri) {
  LLVM_DEBUG(llvm::dbgs() << "removeDocument: uri='" << uri.uri() << "'\n");
  auto versionIt = impl->fileVersions.find(uri.uri());
  if (versionIt == impl->fileVersions.end()) {
    LLVM_DEBUG(llvm::dbgs() << "removeDocument: unknown uri, returning "
                               "nullopt\n");
    return std::nullopt;
  }

  int64_t version = versionIt->second;
  std::string key = getInternalKey(uri);
  LLVM_DEBUG(llvm::dbgs() << "removeDocument: key='" << key
                          << "' version=" << version << "\n");

  impl->fileVersions.erase(versionIt);
  impl->userOpenedFiles.erase(key);
  impl->fileContents.erase(key);

  // Check if this is a layout file with a context.
  auto ctxIt = impl->contexts.find(key);
  if (ctxIt != impl->contexts.end()) {
    LLVM_DEBUG(llvm::dbgs() << "removeDocument: is layout file, reverting "
                               "context to independent\n");
    impl->revertToIndependent(ctxIt->second.get());
    impl->contexts.erase(ctxIt);
    impl->rebuildFileToContextMapping();
    return version;
  }

  // Check if this file is in a context (but not the layout).
  // Return nullopt to prevent onDocumentDidClose from clearing diagnostics —
  // they're owned by the context and remain valid.
  auto fileCtxIt = impl->fileToContext.find(key);
  if (fileCtxIt != impl->fileToContext.end()) {
    LLVM_DEBUG(llvm::dbgs() << "removeDocument: file is in context (layout='"
                            << fileCtxIt->second->getLayoutPath()
                            << "'), returning nullopt to preserve "
                               "diagnostics\n");
    return std::nullopt;
  }

  // Remove independent file.
  LLVM_DEBUG(llvm::dbgs() << "removeDocument: removing independent file\n");
  impl->independentFiles.erase(key);
  return version;
}

void mlir::csl::CSLServer::getLocationsOf(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    std::vector<lsp::Location> &locations) {
  std::string key = getInternalKey(uri);
  auto view = impl->resolveView(key);
  if (view) {
    LLVM_DEBUG(llvm::dbgs() << "getLocationsOf: key='" << key << "' viewFile='"
                            << view->filePath << "'\n");
    view->getLocationsOf(uri, pos, locations);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "getLocationsOf: no view for key='" << key << "'\n");
  }
}

std::optional<lsp::Hover>
mlir::csl::CSLServer::findHover(const lsp::URIForFile &uri,
                                const lsp::Position &pos) {
  std::string key = getInternalKey(uri);
  auto view = impl->resolveView(key);
  if (!view) {
    LLVM_DEBUG(llvm::dbgs() << "findHover: no view for key='" << key << "'\n");
    return std::nullopt;
  }
  return view->findHover(uri, pos);
}

void mlir::csl::CSLServer::findDocumentSymbols(
    const lsp::URIForFile &uri, std::vector<lsp::DocumentSymbol> &symbols) {
  std::string key = getInternalKey(uri);
  auto view = impl->resolveView(key);
  if (!view) {
    LLVM_DEBUG(llvm::dbgs()
               << "findDocumentSymbols: no view for key='" << key << "'\n");
    return;
  }
  view->findDocumentSymbols(uri, symbols);
}

lsp::CompletionList
mlir::csl::CSLServer::getCodeCompletion(const lsp::URIForFile &uri,
                                        const lsp::Position &completePos,
                                        const lsp::CompletionContext &ctx) {
  std::string key = getInternalKey(uri);
  auto view = impl->resolveView(key);
  if (!view) {
    LLVM_DEBUG(llvm::dbgs()
               << "getCodeCompletion: no view for key='" << key << "'\n");
    return {};
  }
  auto contentsIt = impl->fileContents.find(key);
  llvm::StringRef sourceText = (contentsIt != impl->fileContents.end())
                                   ? llvm::StringRef(contentsIt->second)
                                   : llvm::StringRef();
  return view->getCompletions(completePos, ctx, sourceText);
}

lsp::SignatureHelp
mlir::csl::CSLServer::getSignatureHelp(const lsp::URIForFile &uri,
                                       const lsp::Position &pos) {
  std::string key = getInternalKey(uri);
  auto view = impl->resolveView(key);
  if (!view) {
    LLVM_DEBUG(llvm::dbgs()
               << "getSignatureHelp: no view for key='" << key << "'\n");
    return {};
  }
  auto contentsIt = impl->fileContents.find(key);
  llvm::StringRef sourceText = (contentsIt != impl->fileContents.end())
                                   ? llvm::StringRef(contentsIt->second)
                                   : llvm::StringRef();
  return view->getSignatureHelp(pos, sourceText);
}

std::optional<std::string>
mlir::csl::CSLServer::getFileContent(const lsp::URIForFile &uri) {
  std::string key = getInternalKey(uri);
  LLVM_DEBUG(llvm::dbgs() << "getFileContent: key='" << key << "' scheme='"
                          << uri.scheme() << "'\n");

  // Check in-memory contents first (editor overrides).
  auto it = impl->fileContents.find(key);
  if (it != impl->fileContents.end()) {
    LLVM_DEBUG(llvm::dbgs() << "getFileContent: returning in-memory contents ("
                            << it->second.size() << " bytes)\n");
    return it->second;
  }

  // For csl-sdk:// URIs, resolve the system module from the SIF container.
  // The URI path is /memcpy/memcpy which maps to module path "memcpy/memcpy".
  if (uri.scheme() == lsp_utils::kCslSdkScheme && impl->systemResolver) {
    llvm::StringRef modulePath = uri.file().drop_front(1); // strip leading /
    LLVM_DEBUG(llvm::dbgs() << "getFileContent: resolving system module '"
                            << modulePath << "'\n");
    auto result = impl->systemResolver->resolveSystemModule(modulePath);
    if (mlir::succeeded(result)) {
      auto *buf = impl->systemResolver->getSourceMgr()->getMemoryBuffer(
          result->bufferID);
      LLVM_DEBUG(llvm::dbgs() << "getFileContent: resolved system module ("
                              << buf->getBufferSize() << " bytes)\n");
      return buf->getBuffer().str();
    }
    LLVM_DEBUG(llvm::dbgs() << "getFileContent: system module resolution "
                               "failed\n");
    return std::nullopt;
  }

  // Fall back to reading from disk.
  auto bufOrErr = llvm::MemoryBuffer::getFile(key);
  if (bufOrErr) {
    LLVM_DEBUG(llvm::dbgs() << "getFileContent: read from disk ("
                            << (*bufOrErr)->getBufferSize() << " bytes)\n");
    return (*bufOrErr)->getBuffer().str();
  }
  LLVM_DEBUG(llvm::dbgs() << "getFileContent: file not found: "
                          << bufOrErr.getError().message() << "\n");
  return std::nullopt;
}
