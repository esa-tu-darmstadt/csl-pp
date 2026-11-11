//===- LspUtils.cpp - Shared LSP Utilities
//---------------------------------===//
//
// Utility functions shared between CSLServer.cpp and CompilationContext.cpp.
//
//===----------------------------------------------------------------------===//

#include "LspUtils.h"

#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "llvm/ADT/STLExtras.h"

namespace lsp = llvm::lsp;

//===----------------------------------------------------------------------===//
// URI Utilities
//===----------------------------------------------------------------------===//

std::optional<lsp::URIForFile>
mlir::csl::lsp_utils::uriForPath(llvm::StringRef path) {
  // System module display names like "<memcpy/memcpy>" use the csl-sdk scheme.
  // Strip the angle brackets and use the inner path as the URI path.
  if (isSystemModuleDisplayName(path)) {
    llvm::StringRef inner = path.drop_front(1).drop_back(1);
    // Append .csl so VS Code recognizes the language mode from the extension.
    std::string uriPath = ("/" + inner + ".csl").str();
    auto uri = lsp::URIForFile::fromFile(uriPath, kCslSdkScheme);
    if (uri)
      return *uri;
    llvm::consumeError(uri.takeError());
    return std::nullopt;
  }
  auto uri = lsp::URIForFile::fromFile(path);
  if (uri)
    return *uri;
  llvm::consumeError(uri.takeError());
  return std::nullopt;
}

std::optional<lsp::Location>
mlir::csl::lsp_utils::getLocWithScheme(mlir::Location loc) {
  auto range = getRange(loc);
  if (!range)
    return std::nullopt;

  llvm::StringRef filename;
  if (auto flcRange = mlir::dyn_cast<mlir::FileLineColRange>(loc))
    filename = flcRange.getFilename().strref();
  else if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    filename = flc.getFilename().strref();
  else
    return std::nullopt;

  auto uri = uriForPath(filename);
  if (!uri)
    return std::nullopt;
  return lsp::Location(*uri, *range);
}

//===----------------------------------------------------------------------===//
// Location Conversion
//===----------------------------------------------------------------------===//

lsp::Position mlir::csl::lsp_utils::mlirToLspPosition(unsigned line,
                                                      unsigned col) {
  return lsp::Position(line > 0 ? line - 1 : 0, col > 0 ? col - 1 : 0);
}

std::optional<lsp::Range> mlir::csl::lsp_utils::getRange(mlir::Location loc) {
  if (auto range = mlir::dyn_cast<mlir::FileLineColRange>(loc)) {
    lsp::Position start =
        mlirToLspPosition(range.getStartLine(), range.getStartColumn());
    lsp::Position end =
        mlirToLspPosition(range.getEndLine(), range.getEndColumn());
    return lsp::Range(start, end);
  }
  if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    lsp::Position pos = mlirToLspPosition(flc.getLine(), flc.getColumn());
    return lsp::Range(pos, pos);
  }
  return std::nullopt;
}

std::pair<int, int> mlir::csl::lsp_utils::rangeSize(const lsp::Range &range) {
  return {range.end.line - range.start.line,
          range.end.character - range.start.character};
}

std::string mlir::csl::lsp_utils::getFileFromLocation(mlir::Location loc) {
  if (auto range = mlir::dyn_cast<mlir::FileLineColRange>(loc))
    return range.getFilename().getValue().str();
  if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    return flc.getFilename().getValue().str();
  return {};
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

lsp::Diagnostic mlir::csl::lsp_utils::mlirDiagToLspDiag(
    mlir::Diagnostic &diag, const std::optional<lsp::URIForFile> &uri) {
  lsp::Diagnostic lspDiag;
  auto range = getRange(diag.getLocation());
  if (range)
    lspDiag.range = *range;
  lspDiag.source = "csl";

  switch (diag.getSeverity()) {
  case mlir::DiagnosticSeverity::Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case mlir::DiagnosticSeverity::Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case mlir::DiagnosticSeverity::Note:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  case mlir::DiagnosticSeverity::Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Hint;
    break;
  }

  lspDiag.message = diag.str();

  // Add editor-friendly hints to certain compiler diagnostics.
  if (lspDiag.message.find("--system-module-sif") != std::string::npos ||
      lspDiag.message.find("--system-import-path") != std::string::npos) {
    auto replace = [](std::string &s, llvm::StringRef from,
                      llvm::StringRef to) {
      size_t pos = 0;
      while ((pos = s.find(from.data(), pos, from.size())) !=
             std::string::npos) {
        s.replace(pos, from.size(), to.data(), to.size());
        pos += to.size();
      }
    };
    replace(lspDiag.message, "--system-module-sif", "csl.systemModuleSif");
    replace(lspDiag.message, "--system-import-path", "csl.systemImportPath");
    lspDiag.message +=
        "\n\nSet \"csl.systemModuleSif\" to the path of the SDK's .sif "
        "container. Alternatively, \"csl.systemImportPath\" can point to a "
        "directory with system module sources.";
  }

  if (lspDiag.message.find("has no binding or default value") !=
      std::string::npos) {
    lspDiag.message +=
        ".\nFor non-layout files, keep the layout file open so the "
        "language server can infer parameter values from @set_tile_code. "
        "For layout files, provide bindings via the \"csl.params\" "
        "setting in the VS Code extension or add a default value in source.";
  }

  for (auto &note : diag.getNotes()) {
    auto noteRange = getRange(note.getLocation());
    if (!noteRange)
      continue;
    if (!lspDiag.relatedInformation)
      lspDiag.relatedInformation.emplace();
    if (uri) {
      lspDiag.relatedInformation->emplace_back(lsp::Location(*uri, *noteRange),
                                               note.str());
    }
  }

  return lspDiag;
}

mlir::ScopedDiagnosticHandler mlir::csl::lsp_utils::createRoutingDiagHandler(
    mlir::MLIRContext *ctx, llvm::StringRef defaultFilePath,
    llvm::StringMap<std::vector<lsp::Diagnostic>> &allDiagnostics,
    bool filterOverlappingErrors) {
  return mlir::ScopedDiagnosticHandler(
      ctx,
      [&allDiagnostics, defaultPath = defaultFilePath.str(),
       filterOverlappingErrors](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        std::string diagFile = getFileFromLocation(diag.getLocation());
        std::optional<lsp::URIForFile> uri;
        if (!diagFile.empty())
          uri = uriForPath(diagFile);
        std::string key = diagFile.empty() ? defaultPath : diagFile;
        lsp::Diagnostic lspDiag = mlirDiagToLspDiag(diag, uri);

        auto &fileDiags = allDiagnostics[key];

        // Skip exact duplicates (same module imported by multiple files).
        auto isDuplicate = [&](const lsp::Diagnostic &existing) {
          return existing.range == lspDiag.range &&
                 existing.severity == lspDiag.severity &&
                 existing.message == lspDiag.message;
        };
        if (llvm::any_of(fileDiags, isDuplicate))
          return mlir::success();

        if (filterOverlappingErrors &&
            lspDiag.severity == lsp::DiagnosticSeverity::Error) {

          // If an existing error strictly contains the new one, remove it
          // (the new one is more specific).
          llvm::erase_if(fileDiags, [&](const lsp::Diagnostic &existing) {
            return existing.severity == lsp::DiagnosticSeverity::Error &&
                   existing.range.contains(lspDiag.range) &&
                   existing.range != lspDiag.range;
          });

          // If the new error strictly contains an existing error, drop it
          // (a more specific error already exists).
          bool subsumed =
              llvm::any_of(fileDiags, [&](const lsp::Diagnostic &existing) {
                return existing.severity == lsp::DiagnosticSeverity::Error &&
                       lspDiag.range.contains(existing.range) &&
                       lspDiag.range != existing.range;
              });

          if (!subsumed)
            fileDiags.push_back(std::move(lspDiag));
        } else {
          fileDiags.push_back(std::move(lspDiag));
        }

        return mlir::success();
      });
}

//===----------------------------------------------------------------------===//
// Context & Source Manager Setup
//===----------------------------------------------------------------------===//

void mlir::csl::lsp_utils::initLspContext(mlir::MLIRContext &ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::parsed_csl::ParsedCSLDialect,
                  mlir::typed_csl::TypedCSLDialect>();
  mlir::parsed_csl::registerComptimeModels(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  ctx.printOpOnDiagnostic(false);
}

std::shared_ptr<mlir::csl::CslModuleResolver>
mlir::csl::lsp_utils::createModuleResolver(llvm::StringRef filePath,
                                           const CslFlags &flags) {
  return std::make_shared<CslModuleResolver>(filePath, flags);
}

std::shared_ptr<mlir::csl::CslModuleResolver>
mlir::csl::lsp_utils::createModuleResolver(
    llvm::StringRef filePath, const CslFlags &flags,
    std::shared_ptr<SystemModuleCache> cache) {
  return std::make_shared<CslModuleResolver>(filePath, flags, std::move(cache));
}
