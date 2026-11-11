//===- CslModuleResolver.h - Module File Resolution -------------*- C++ -*-===//
//
// Centralized module file resolution with separate VFS stacks for user and
// system modules.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_UTILS_CSLMODULERESOLVER_H
#define CSL_MLIR_DIALECT_CSL_UTILS_CSLMODULERESOLVER_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mlir::csl {
class CslFlags;

/// Cached SIF-backed system VFS and expanded glob paths. Created once and
/// shared across multiple CslModuleResolver instances to avoid repeatedly
/// opening the SIF container.
struct SystemModuleCache {
  /// The VFS backed by the SIF container.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> sifVFS;
  /// System import paths with glob patterns already expanded against sifVFS.
  std::vector<std::string> expandedSystemPaths;
  /// Decompressed file content cache: absolute path → raw file bytes.
  /// Shared across all resolvers so each system module is decompressed from
  /// the SIF at most once, regardless of how many PE programs are compiled.
  std::mutex contentMutex;
  llvm::StringMap<std::string> fileContentCache;

  /// Create a SystemModuleCache from flags. Returns nullptr when no SIF is
  /// configured or when the SIF fails to open.
  static std::shared_ptr<SystemModuleCache> create(const CslFlags &flags);
};

/// Result of resolving a module file.
struct ResolvedModule {
  /// The SourceMgr buffer ID for the resolved file.
  unsigned bufferID;
  /// The resolved absolute path of the file.
  std::string resolvedPath;
};

/// Centralizes module file resolution with separate VFS stacks for user and
/// system modules. Wraps a single SourceMgr and provides deduplication of
/// resolved buffers.
///
/// User modules are resolved through the SourceMgr's include directory
/// mechanism. System modules (angle bracket imports like "<memcpy/get_params>")
/// are resolved by searching systemImportPaths, optionally through a separate
/// VFS (for future .sif container support).
class CslModuleResolver {
public:
  /// Construct a resolver from a file path and CslFlags.
  /// Computes userIncludeDirs from parent_path(filePath) + flags.importPaths,
  /// gets systemImportPaths from flags, and if flags.hasSystemModuleSif(),
  /// creates a SifFileSystem internally and expands glob patterns.
  CslModuleResolver(llvm::StringRef filePath, const CslFlags &flags);

  /// Construct a resolver from a file path, CslFlags, and a pre-built
  /// SystemModuleCache. Reuses the cached SIF VFS and expanded paths instead
  /// of opening the SIF container again. Falls back to the 2-arg constructor
  /// behavior when cache is null.
  CslModuleResolver(llvm::StringRef filePath, const CslFlags &flags,
                    std::shared_ptr<SystemModuleCache> cache);

  /// Construct a resolver with the given include directories.
  /// \param userIncludeDirs Directories for user module resolution (set on
  ///   SourceMgr).
  /// \param systemImportPaths Directories for system module resolution.
  CslModuleResolver(llvm::ArrayRef<std::string> userIncludeDirs,
                    llvm::ArrayRef<std::string> systemImportPaths);

  /// Resolve a user module (regular import, no angle brackets).
  ///
  /// If \p importerDir is non-empty, the resolver first tries to resolve
  /// \p filename relative to that directory. This lets nested imports inside
  /// libraries find their siblings even when the top-level PE program lives
  /// elsewhere. On miss, falls back to cwd-relative + SourceMgr include
  /// directories.
  mlir::FailureOr<ResolvedModule>
  resolveUserModule(llvm::StringRef filename, llvm::StringRef importerDir = "");

  /// Resolve a system module (angle bracket import, brackets already stripped).
  /// Searches systemImportPaths, adds .csl extension if needed.
  mlir::FailureOr<ResolvedModule>
  resolveSystemModule(llvm::StringRef modulePath);

  /// Get the underlying SourceMgr.
  std::shared_ptr<llvm::SourceMgr> getSourceMgr() const { return sourceMgr_; }

  /// Add a buffer to the SourceMgr. Returns the buffer ID.
  unsigned addBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer);

  /// Apply editor overlay buffers. Creates an InMemoryFileSystem from the
  /// override map and overlays it onto both the SourceMgr's VFS (for user
  /// imports) and the system VFS (for system imports).
  void applyEditorOverlay(const llvm::StringMap<std::string> &overrideBuffers);

  /// Replace the system VFS (for future SifFileSystem support).
  void setSystemVFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs);

  /// Return true if a custom system VFS (e.g. SifFileSystem) is active.
  bool hasCustomSystemVFS() const { return systemVFS_ != nullptr; }

  /// Return true if any system import paths are configured.
  bool hasSystemImportPaths() const { return !systemImportPaths_.empty(); }

  /// Select which fiber runtime variant `<coro/fiber_runtime>` resolves to.
  /// The two-argument constructors take this from CslFlags; the explicit
  /// include-dir constructor (used by tests and the LSP) defaults to WSE2 and
  /// can override it here.
  void setTargetArch(TargetArch arch) { targetArch_ = arch; }

private:
  /// Serve one of the embedded coroutine runtime modules from memory. Returns
  /// std::nullopt when \p modulePath does not name one.
  std::optional<ResolvedModule>
  resolveCoroRuntimeModule(llvm::StringRef modulePath);

  std::shared_ptr<SystemModuleCache> systemModuleCache_;
  std::shared_ptr<llvm::SourceMgr> sourceMgr_;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> systemVFS_;
  std::vector<std::string> systemImportPaths_;
  llvm::StringMap<unsigned> resolvedBuffers_; // dedup cache: path -> bufferID
  TargetArch targetArch_ = TargetArch::WSE2;
};

} // namespace mlir::csl

#endif // CSL_MLIR_DIALECT_CSL_UTILS_CSLMODULERESOLVER_H
