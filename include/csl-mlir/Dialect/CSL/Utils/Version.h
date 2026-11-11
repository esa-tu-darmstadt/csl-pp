//===- Version.h - csl++ release and build information ----------*- C++ -*-===//
//
// The csl++ release and commit that the tools were built from, as printed by
// their --version option.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_UTILS_VERSION_H
#define CSL_MLIR_DIALECT_CSL_UTILS_VERSION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::csl {

/// The csl++ release the tools were built as, e.g. "1.0.0", or an empty
/// string for a development build.
llvm::StringRef getReleaseVersion();

/// The commit the tools were built from, with the suffix "-dirty" if the
/// working tree had uncommitted changes, or an empty string if the build did
/// not record it.
llvm::StringRef getGitCommit();

/// Print the version of the csl++ tool `toolName`: its name, the csl++
/// release, the commit and the LLVM version, starting with a line like
/// "csl-pp (csl++) 1.0.0". Tools install it for --version with
/// llvm::cl::SetVersionPrinter().
void printVersion(llvm::raw_ostream &os, llvm::StringRef toolName);

} // namespace mlir::csl

#endif // CSL_MLIR_DIALECT_CSL_UTILS_VERSION_H
