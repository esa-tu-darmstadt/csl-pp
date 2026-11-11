//===- LspUtils.h - Shared LSP Utilities -------------------------*- C++
//-*-===//
//
// Utility functions shared between CSLServer.cpp and CompilationContext.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TOOLS_CSL_LSP_SERVER_LSPUTILS_H
#define LIB_TOOLS_CSL_LSP_SERVER_LSPUTILS_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::csl {
class CslModuleResolver;
struct SystemModuleCache;
} // namespace mlir::csl

namespace mlir::csl::lsp_utils {

/// URI scheme used for system modules resolved from .sif containers.
inline constexpr const char *kCslSdkScheme = "csl-sdk";

/// Return true if \p path is a system module display name (e.g.
/// "<memcpy/memcpy>"). These are produced by CslModuleResolver when resolving
/// system modules from a SIF container.
inline bool isSystemModuleDisplayName(llvm::StringRef path) {
  return path.starts_with("<") && path.ends_with(">");
}

/// Return a URIForFile for the given path. For system module display names
/// (e.g. "<memcpy/memcpy>"), uses the `csl-sdk` scheme. For regular paths,
/// uses the `file` scheme.
std::optional<llvm::lsp::URIForFile> uriForPath(llvm::StringRef path);

/// Try to extract an LSP Location from an MLIR Location. Automatically uses
/// the `csl-sdk` scheme for system module display names.
std::optional<llvm::lsp::Location> getLocWithScheme(mlir::Location loc);

/// Convert a 1-based MLIR line/col to a 0-based LSP Position.
llvm::lsp::Position mlirToLspPosition(unsigned line, unsigned col);

/// Try to extract an LSP Range from an MLIR Location.
std::optional<llvm::lsp::Range> getRange(mlir::Location loc);

/// Compute the span size of an LSP Range (for comparing specificity).
std::pair<int, int> rangeSize(const llvm::lsp::Range &range);

/// Get the absolute file path from an MLIR location, or empty string.
std::string getFileFromLocation(mlir::Location loc);

/// Convert an MLIR diagnostic to an LSP diagnostic.
llvm::lsp::Diagnostic
mlirDiagToLspDiag(mlir::Diagnostic &diag,
                  const std::optional<llvm::lsp::URIForFile> &uri);

/// Create a ScopedDiagnosticHandler that routes diagnostics by source file
/// into the provided map. Diagnostics with no file location are routed to
/// defaultFilePath.
///
/// When \p filterOverlappingErrors is true, incoming error diagnostics are
/// filtered online: if a new error's range strictly contains an existing
/// error's range the new one is dropped; if an existing error strictly
/// contains the new one the existing one is removed.
mlir::ScopedDiagnosticHandler createRoutingDiagHandler(
    mlir::MLIRContext *ctx, llvm::StringRef defaultFilePath,
    llvm::StringMap<std::vector<llvm::lsp::Diagnostic>> &allDiagnostics,
    bool filterOverlappingErrors = false);

/// Register CSL dialects and configure an MLIRContext for LSP use.
void initLspContext(mlir::MLIRContext &ctx);

/// Create a CslModuleResolver with include directories derived from the file's
/// parent directory and the flags' import/system-import paths.
std::shared_ptr<csl::CslModuleResolver>
createModuleResolver(llvm::StringRef filePath, const CslFlags &flags);

/// Create a CslModuleResolver, reusing a pre-built SystemModuleCache to avoid
/// re-opening the SIF container.
std::shared_ptr<csl::CslModuleResolver>
createModuleResolver(llvm::StringRef filePath, const CslFlags &flags,
                     std::shared_ptr<SystemModuleCache> cache);

} // namespace mlir::csl::lsp_utils

#endif // LIB_TOOLS_CSL_LSP_SERVER_LSPUTILS_H
