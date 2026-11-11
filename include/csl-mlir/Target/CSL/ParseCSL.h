//===- ParseCSL.h - CSL Parser --------------------------------------------===//
//
// Public entry point for parsing CSL source.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>

namespace mlir {
namespace csl {

/// Parse CSL source into an existing region.
///
/// \param sourceMgr Source manager containing the CSL source code
/// \param bufferID Buffer ID within the SourceMgr to parse
/// \param region Region to parse the CSL code into
/// \param paramBindings Parameter bindings for module instantiation
/// \param lineOffset Line offset for split-input-file support
/// \param sourceFilename Override filename for locations
/// \param currentModule Current module context (nullptr if top level)
/// \param lspMode Enable LSP-style error-tolerant parsing
/// \param cslFlags Compiler flags
/// \param isSystemModule Whether parsing a system module
LogicalResult parseCSLIntoRegion(
    const std::shared_ptr<llvm::SourceMgr> &sourceMgr, unsigned bufferID,
    Region &region,
    const llvm::DenseMap<llvm::StringRef, ParamBinding> &paramBindings =
        llvm::DenseMap<llvm::StringRef, ParamBinding>(),
    unsigned lineOffset = 0, llvm::StringRef sourceFilename = "",
    parsed_csl::ModuleOp currentModule = nullptr, bool lspMode = false,
    const CslFlags &cslFlags = {}, bool isSystemModule = false);

} // namespace csl
} // namespace mlir
