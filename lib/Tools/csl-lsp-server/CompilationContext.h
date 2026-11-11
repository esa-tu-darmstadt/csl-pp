//===- CompilationContext.h - Layout-Driven Compilation Context --*- C++
//-*-===//
//
// Manages a layout file and its referenced PE programs within a shared
// MLIRContext for the LSP server.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TOOLS_CSL_LSP_SERVER_COMPILATIONCONTEXT_H
#define LIB_TOOLS_CSL_LSP_SERVER_COMPILATIONCONTEXT_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/SourceMgr.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir::csl {
class CslModuleResolver;
struct SystemModuleCache;
} // namespace mlir::csl

namespace mlir::csl {

/// A lightweight view into a compilation's results for a specific file.
/// Provides LSP query methods (hover, go-to-definition, symbols) without
/// owning any resources. Both independent FileStates and CompilationContexts
/// produce DocumentViews.
struct DocumentView {
  csl::CompilationUnitOp compilationUnit;
  parsed_csl::ComptimeEvaluator *analysis = nullptr; // may be null
  std::string filePath;
  bool parsedSuccessfully = false;

  /// Find the innermost operation whose location contains the position.
  mlir::Operation *findOpAtPosition(const llvm::lsp::Position &pos);

  /// Find the innermost operation of a specific type whose location contains
  /// the position. Useful for finding enclosing ops like CallOp that are
  /// siblings (not parents) of the op at the cursor.
  template <typename OpT>
  OpT findOpAtPosition(const llvm::lsp::Position &pos) {
    auto *op = findOpMatching(
        pos, [](mlir::Operation *o) { return mlir::isa<OpT>(o); });
    return op ? mlir::cast<OpT>(op) : OpT(nullptr);
  }

  /// Get the symbol definition by name from the compilation unit.
  mlir::Operation *resolveSymbol(llvm::StringRef name);

private:
  /// Walk the IR to find the innermost op at `pos` matching `predicate`.
  /// Skips GenericFuncOp bodies and prunes subtrees outside the position.
  mlir::Operation *
  findOpMatching(const llvm::lsp::Position &pos,
                 llvm::function_ref<bool(mlir::Operation *)> predicate);

public:
  /// Format hover info for an operation.
  std::optional<std::string> formatHoverInfo(mlir::Operation *op);

  /// Return locations of the definition at the given position.
  void getLocationsOf(const llvm::lsp::URIForFile &uri,
                      const llvm::lsp::Position &pos,
                      std::vector<llvm::lsp::Location> &locations);

  /// Find hover information for the given position.
  std::optional<llvm::lsp::Hover> findHover(const llvm::lsp::URIForFile &uri,
                                            const llvm::lsp::Position &pos);

  /// Find all document symbols in the given file.
  void findDocumentSymbols(const llvm::lsp::URIForFile &uri,
                           std::vector<llvm::lsp::DocumentSymbol> &symbols);

  /// Get code completions at the given position.
  llvm::lsp::CompletionList
  getCompletions(const llvm::lsp::Position &pos,
                 const llvm::lsp::CompletionContext &ctx,
                 llvm::StringRef sourceText);

  /// Get signature help at the given position.
  llvm::lsp::SignatureHelp getSignatureHelp(const llvm::lsp::Position &pos,
                                            llvm::StringRef sourceText);
};

/// Holds the parsed + analyzed state of a single CSL file.
/// Used for layout files, PE programs, and independent (standalone) files.
struct FileState {
  std::string filePath;
  std::shared_ptr<csl::CslModuleResolver> moduleResolver;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  csl::CompilationUnitOp compilationUnit;
  std::unique_ptr<parsed_csl::ComptimeEvaluator> analysis;
  bool parsedSuccessfully = false;
  bool hasLayoutBlock = false;

  /// PE-program-specific fields (unused for layout/independent files).
  csl::StructValueAttr params;
  std::optional<mlir::Location> paramsLoc;

  /// Produce a DocumentView, optionally filtering to a different file path
  /// (for viewing imported modules through their owner's IR).
  DocumentView toView(llvm::StringRef filterFilePath = {}) const;
};

/// Parse a CSL file and run comptime analysis, populating the given FileState.
/// state.filePath must already be set. The caller must install a
/// ScopedDiagnosticHandler before calling this function.
/// If shouldEagerlyEvaluate is provided, it is forwarded to ComptimeEvalOptions
/// to eagerly evaluate functions/globals in matching files (for LSP support).
void parseAndAnalyze(
    mlir::MLIRContext *ctx, FileState &state, llvm::StringRef contents,
    const CslFlags &flags,
    const llvm::DenseMap<llvm::StringRef, csl::ParamBinding> &paramBindings,
    const llvm::StringMap<std::string> &overrideBuffers = {},
    std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr,
    std::function<bool(llvm::StringRef)> shouldEagerlyEvaluate = nullptr);

/// A compilation context anchored by a layout file.
///
/// Uses a shared MLIRContext (owned externally), the layout's
/// CompilationUnitOp + analysis, and one CompilationUnitOp + analysis per
/// PE program referenced by @set_tile_code calls. Provides DocumentViews
/// for any file in the context.
///
/// Diagnostics from compilation are stored internally and can be queried
/// per-file via getDiagnostics() / getDiagnosticsForFile().
class CompilationContext {
public:
  CompilationContext(
      mlir::MLIRContext &ctx, const CslFlags &flags,
      bool filterOverlappingErrors = false,
      std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr);

  /// Analyze the layout file. Discovers PE programs via @set_tile_code.
  /// Diagnostics are stored internally (accessible via getDiagnostics()).
  /// overrideBuffers: file path -> in-memory contents for imports that the user
  /// has edited (so SourceMgr uses editor contents instead of disk).
  void analyzeLayout(llvm::StringRef filePath, llvm::StringRef contents,
                     const llvm::StringMap<std::string> &overrideBuffers = {});

  /// Check if a file path (absolute) is known in this context.
  bool containsFile(llvm::StringRef absPath) const;

  /// Get a DocumentView for a file. Returns nullopt if file not in context.
  std::optional<DocumentView> getView(llvm::StringRef absPath) const;

  /// Get all known file paths (layout + PE programs + imported modules).
  void getKnownFilePaths(llvm::SmallVectorImpl<std::string> &paths) const;

  /// Re-analyze a PE program after its source or one of its imports changed.
  /// Diagnostics are stored internally (accessible via getDiagnostics()).
  /// overrideBuffers: file path -> in-memory contents for imports that the user
  /// has edited (so SourceMgr uses editor contents instead of disk).
  void
  reanalyzePEProgram(llvm::StringRef peAbsPath, llvm::StringRef newContents,
                     const llvm::StringMap<std::string> &overrideBuffers = {});

  /// Set the set of user-opened file paths. Used to enable eager evaluation
  /// of functions/globals in these files during comptime analysis.
  void setOpenFiles(const llvm::StringSet<> &files) { openFiles_ = files; }

  /// Get the layout file path.
  llvm::StringRef getLayoutPath() const;

  /// Look up which file owns an imported module file.
  /// Returns empty string if not found.
  llvm::StringRef getOwnerOfImportedModule(llvm::StringRef absPath) const;

  /// Get the stored diagnostics map (file path -> diagnostics).
  const llvm::StringMap<std::vector<llvm::lsp::Diagnostic>> &
  getDiagnostics() const {
    return storedDiagnostics;
  }

  /// Get stored diagnostics for a specific file.
  llvm::ArrayRef<llvm::lsp::Diagnostic>
  getDiagnosticsForFile(llvm::StringRef absPath) const {
    auto it = storedDiagnostics.find(absPath);
    if (it != storedDiagnostics.end())
      return it->second;
    return {};
  }

private:
  mlir::MLIRContext &context;
  CslFlags flags;
  bool filterOverlappingErrors;
  std::shared_ptr<SystemModuleCache> systemModuleCache;

  /// User-opened files for eager evaluation.
  llvm::StringSet<> openFiles_;

  std::unique_ptr<FileState> layout;
  llvm::StringMap<std::unique_ptr<FileState>> pePrograms;

  /// Map: imported module absolute source_file -> owning file abs path.
  llvm::StringMap<std::string> importedModuleToOwner;

  /// Stored diagnostics from the last compilation, keyed by file path.
  llvm::StringMap<std::vector<llvm::lsp::Diagnostic>> storedDiagnostics;

  /// Compile all PE programs discovered from @set_tile_code calls.
  void compilePEPrograms();

  /// Compile a single PE program with the given params.
  void compileSinglePEProgram(FileState &pe);

  /// Build the imported module -> owner mapping from all compilation units.
  void buildFileMapping();

  /// Create param bindings from a StructValueAttr.
  llvm::DenseMap<llvm::StringRef, csl::ParamBinding>
  createParamBindings(csl::StructValueAttr params,
                      mlir::Location definitionLoc);
};

} // namespace mlir::csl

#endif // LIB_TOOLS_CSL_LSP_SERVER_COMPILATIONCONTEXT_H
