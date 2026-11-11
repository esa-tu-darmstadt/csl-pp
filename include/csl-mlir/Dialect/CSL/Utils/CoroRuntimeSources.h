//===- CoroRuntimeSources.h - Embedded coroutine runtime --------*- C++ -*-===//
//
// The coroutine runtime modules, compiled into the binary.
//
// The runtime is not a library the user imports; it is part of the compiler.
// Its symbols are referenced by name from the lowering passes
// (`__fiber_runtime__*`, `__alloc_runtime__*`), and `StripRuntimeExportsPass`
// removes the export markers that only exist to keep them alive until those
// passes run. Carrying the sources in the binary keeps that relationship
// self-contained: there is no path to configure, nothing to install alongside
// the tool, and no way for a stale checkout to supply a mismatched runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_UTILS_CORORUNTIMESOURCES_H
#define CSL_MLIR_DIALECT_CSL_UTILS_CORORUNTIMESOURCES_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::csl {

/// Import paths under which the runtime modules are served. These are angle
/// bracket (system) imports so they cannot collide with a user file, and they
/// are resolved from memory rather than from any import path.
constexpr llvm::StringLiteral kFiberRuntimeImportPath = "coro/fiber_runtime";
constexpr llvm::StringLiteral kAllocRuntimeImportPath = "coro/resources";

/// Module names the runtime is bound to in the PE program. The lowering passes
/// look symbols up as `<name>__<member>` after module inlining, so these must
/// match `CSLDialect::get{Fiber,Alloc}RuntimePrefix()` without the trailing
/// underscores.
constexpr llvm::StringLiteral kFiberRuntimeModuleName = "__fiber_runtime";
constexpr llvm::StringLiteral kAllocRuntimeModuleName = "__alloc_runtime";

/// The fiber runtime source for \p arch. The two archs differ in their
/// instruction sequences for stack switching, so the choice is made here from
/// the target the compiler was invoked with, not by an `@is_arch` switch in
/// user source.
llvm::StringRef getFiberRuntimeSource(TargetArch arch);

/// The resource-allocation runtime source, shared by both archs (it branches
/// internally on `@is_arch` for the bitmap widths).
llvm::StringRef getAllocRuntimeSource();

/// Whether \p modulePath (angle brackets already stripped) names one of the
/// embedded runtime modules.
bool isCoroRuntimeImportPath(llvm::StringRef modulePath);

/// The embedded source for an import path accepted by
/// `isCoroRuntimeImportPath`. \p arch selects the fiber runtime variant.
llvm::StringRef getCoroRuntimeSource(llvm::StringRef modulePath,
                                     TargetArch arch);

} // namespace mlir::csl

#endif // CSL_MLIR_DIALECT_CSL_UTILS_CORORUNTIMESOURCES_H
