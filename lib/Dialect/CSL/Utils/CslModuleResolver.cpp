//===- CslModuleResolver.cpp - Module File Resolution ---------------------===//
//
// Implementation of centralized module file resolution.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"

#include "csl-mlir/Dialect/CSL/Utils/CoroRuntimeSources.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Support/Sif/SifFileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace mlir::csl;

/// Expand glob patterns (segments containing '*') in system import paths
/// against the given VFS. For example, "/cb/toolchains/cslang/*/*/csl-libs"
/// expands the two wildcard segments by listing directories.
static void expandGlobs(llvm::ArrayRef<std::string> rawPaths,
                        llvm::vfs::FileSystem &fs,
                        std::vector<std::string> &expandedPaths) {

  for (const auto &rawPath : rawPaths) {
    // If no wildcard, keep as-is.
    if (rawPath.find('*') == std::string::npos) {
      expandedPaths.push_back(rawPath);
      continue;
    }

    // Split the path into segments and expand wildcards one level at a time.
    llvm::SmallVector<llvm::StringRef> segments;
    llvm::StringRef(rawPath).split(segments, '/');

    // Start with the root prefix (or empty for relative paths).
    std::vector<std::string> currentPrefixes;
    size_t startIdx = 0;

    // Handle absolute paths: first segment after splitting "/" is empty.
    if (llvm::StringRef(rawPath).starts_with("/")) {
      currentPrefixes.push_back("/");
      startIdx = 1; // skip empty first segment
    } else {
      currentPrefixes.push_back("");
    }

    for (size_t i = startIdx; i < segments.size(); ++i) {
      llvm::StringRef seg = segments[i];
      if (seg.empty())
        continue;

      std::vector<std::string> nextPrefixes;

      if (seg == "*") {
        // Wildcard: list each current prefix directory and collect children.
        for (const auto &prefix : currentPrefixes) {
          std::error_code ec;
          for (auto it = fs.dir_begin(prefix.empty() ? "." : prefix, ec);
               !ec && it != llvm::vfs::directory_iterator(); it.increment(ec)) {
            auto status = fs.status(it->path());
            if (status && status->isDirectory())
              nextPrefixes.push_back(it->path().str());
          }
        }
      } else {
        // Literal segment: append to each prefix.
        for (const auto &prefix : currentPrefixes) {
          llvm::SmallString<256> joined(prefix);
          llvm::sys::path::append(joined, seg);
          nextPrefixes.push_back(std::string(joined));
        }
      }

      currentPrefixes = std::move(nextPrefixes);
    }

    // All surviving prefixes are expanded paths.
    for (auto &p : currentPrefixes)
      expandedPaths.push_back(std::move(p));
  }
}

std::shared_ptr<SystemModuleCache>
SystemModuleCache::create(const CslFlags &flags) {
  if (flags.systemModuleSif.empty())
    return nullptr;

  auto sifFSOrErr = ::csl::sif::SifFileSystem::create(flags.systemModuleSif);
  if (!sifFSOrErr) {
    llvm::consumeError(sifFSOrErr.takeError());
    return nullptr;
  }

  auto cache = std::make_shared<SystemModuleCache>();
  cache->sifVFS = std::move(*sifFSOrErr);

  std::vector<std::string> rawPaths(flags.systemImportPaths.begin(),
                                    flags.systemImportPaths.end());
  expandGlobs(rawPaths, *cache->sifVFS, cache->expandedSystemPaths);
  return cache;
}

/// Set up user include dirs on the SourceMgr from filePath and flags.
static void setupUserIncludeDirs(llvm::SourceMgr &sourceMgr,
                                 llvm::StringRef filePath,
                                 const CslFlags &flags) {
  std::vector<std::string> userIncludeDirs;
  llvm::StringRef parentPath = llvm::sys::path::parent_path(filePath);
  if (!parentPath.empty())
    userIncludeDirs.push_back(parentPath.str());
  for (const auto &path : flags.importPaths)
    userIncludeDirs.push_back(path);
  sourceMgr.setIncludeDirs(userIncludeDirs);
}

CslModuleResolver::CslModuleResolver(llvm::StringRef filePath,
                                     const CslFlags &flags)
    : sourceMgr_(std::make_shared<llvm::SourceMgr>()),
      targetArch_(flags.targetArch) {
  setupUserIncludeDirs(*sourceMgr_, filePath, flags);

  // If a SIF container is specified, create the VFS from it.
  if (!flags.systemModuleSif.empty()) {
    auto sifFSOrErr = ::csl::sif::SifFileSystem::create(flags.systemModuleSif);
    if (sifFSOrErr) {
      systemVFS_ = std::move(*sifFSOrErr);

      // Expand glob patterns in system import paths against the SIF VFS.
      std::vector<std::string> rawPaths(flags.systemImportPaths.begin(),
                                        flags.systemImportPaths.end());
      expandGlobs(rawPaths, *systemVFS_, systemImportPaths_);
    } else {
      llvm::consumeError(sifFSOrErr.takeError());
      // Fall back to non-SIF paths.
      systemImportPaths_.assign(flags.systemImportPaths.begin(),
                                flags.systemImportPaths.end());
    }
  } else {
    systemImportPaths_.assign(flags.systemImportPaths.begin(),
                              flags.systemImportPaths.end());
  }
}

CslModuleResolver::CslModuleResolver(llvm::StringRef filePath,
                                     const CslFlags &flags,
                                     std::shared_ptr<SystemModuleCache> cache)
    : systemModuleCache_(std::move(cache)),
      sourceMgr_(std::make_shared<llvm::SourceMgr>()),
      targetArch_(flags.targetArch) {
  setupUserIncludeDirs(*sourceMgr_, filePath, flags);

  if (systemModuleCache_) {
    // Reuse the cached SIF VFS and expanded paths.
    systemVFS_ = systemModuleCache_->sifVFS;
    systemImportPaths_ = systemModuleCache_->expandedSystemPaths;
  } else if (!flags.systemModuleSif.empty()) {
    // No cache but SIF is configured — fall back to opening it directly.
    auto sifFSOrErr = ::csl::sif::SifFileSystem::create(flags.systemModuleSif);
    if (sifFSOrErr) {
      systemVFS_ = std::move(*sifFSOrErr);
      std::vector<std::string> rawPaths(flags.systemImportPaths.begin(),
                                        flags.systemImportPaths.end());
      expandGlobs(rawPaths, *systemVFS_, systemImportPaths_);
    } else {
      llvm::consumeError(sifFSOrErr.takeError());
      systemImportPaths_.assign(flags.systemImportPaths.begin(),
                                flags.systemImportPaths.end());
    }
  } else {
    systemImportPaths_.assign(flags.systemImportPaths.begin(),
                              flags.systemImportPaths.end());
  }
}

CslModuleResolver::CslModuleResolver(
    llvm::ArrayRef<std::string> userIncludeDirs,
    llvm::ArrayRef<std::string> systemImportPaths)
    : sourceMgr_(std::make_shared<llvm::SourceMgr>()),
      systemImportPaths_(systemImportPaths.begin(), systemImportPaths.end()) {
  std::vector<std::string> dirs(userIncludeDirs.begin(), userIncludeDirs.end());
  sourceMgr_->setIncludeDirs(dirs);
}

mlir::FailureOr<ResolvedModule>
CslModuleResolver::resolveUserModule(llvm::StringRef filename,
                                     llvm::StringRef importerDir) {
  // If the importing file's directory is known and the import path is
  // relative, try resolving it from that directory first. This lets a
  // library's internal modules import each other by a plain sibling name
  // regardless of where the top-level PE program lives.
  if (!importerDir.empty() && !llvm::sys::path::is_absolute(filename)) {
    llvm::SmallString<256> candidate(importerDir);
    llvm::sys::path::append(candidate, filename);

    auto bufferOrErr = llvm::MemoryBuffer::getFile(candidate);
    if (bufferOrErr) {
      unsigned bufferID = sourceMgr_->AddNewSourceBuffer(
          std::move(*bufferOrErr), llvm::SMLoc());
      llvm::SmallString<256> absPath(candidate);
      llvm::sys::fs::make_absolute(absPath);
      llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
      return ResolvedModule{bufferID, std::string(absPath)};
    }
  }

  // Fall back to LLVM's include resolution (tries cwd-relative, then the
  // configured include directories).
  std::string includedFrom;
  unsigned bufferID =
      sourceMgr_->AddIncludeFile(filename.str(), llvm::SMLoc(), includedFrom);
  if (bufferID == 0)
    return mlir::failure();

  llvm::SmallString<256> absPath(includedFrom);
  llvm::sys::fs::make_absolute(absPath);
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
  return ResolvedModule{bufferID, std::string(absPath)};
}

std::optional<ResolvedModule>
CslModuleResolver::resolveCoroRuntimeModule(llvm::StringRef modulePath) {
  llvm::StringRef stem = modulePath;
  if (stem.ends_with(".csl"))
    stem = stem.drop_back(4);
  if (!isCoroRuntimeImportPath(stem))
    return std::nullopt;

  std::string displayName = ("<" + stem + ">").str();

  // Dedup on the display name: the same runtime module is imported by every PE
  // program, and each resolver instance should buffer it at most once.
  auto cached = resolvedBuffers_.find(displayName);
  if (cached != resolvedBuffers_.end())
    return ResolvedModule{cached->second, displayName};

  // The buffer identifier becomes the filename in FileLineColLoc, so a
  // diagnostic raised inside the runtime points at "<coro/fiber_runtime>"
  // rather than at a path that does not exist.
  unsigned bufferID = addBuffer(llvm::MemoryBuffer::getMemBuffer(
      getCoroRuntimeSource(stem, targetArch_), displayName));
  resolvedBuffers_[displayName] = bufferID;
  return ResolvedModule{bufferID, displayName};
}

mlir::FailureOr<ResolvedModule>
CslModuleResolver::resolveSystemModule(llvm::StringRef modulePath) {
  // The coroutine runtime is part of the compiler, not of any import path, so
  // it is served from memory before the filesystem search. This also keeps it
  // working when system modules come from a SIF container, which has no place
  // to put compiler-owned sources.
  if (auto embedded = resolveCoroRuntimeModule(modulePath))
    return *embedded;

  // Add .csl extension if not present.
  llvm::SmallString<256> modulePathWithExt(modulePath);
  if (!modulePath.ends_with(".csl"))
    modulePathWithExt.append(".csl");

  // For SIF-backed modules, use a display name like "<memcpy/memcpy>" as the
  // buffer identifier and resolved path. This flows through to FileLineColLoc
  // filenames and diagnostic output. Without SIF, use the absolute path.
  std::string displayName;
  if (systemVFS_) {
    llvm::StringRef stem = modulePath;
    if (stem.ends_with(".csl"))
      stem = stem.drop_back(4);
    displayName = ("<" + stem + ">").str();
  }

  // Search system paths.
  for (const auto &sysPath : systemImportPaths_) {
    llvm::SmallString<256> fullPath(sysPath);
    llvm::sys::path::append(fullPath, modulePathWithExt);

    // Check dedup cache.
    auto cacheIt = resolvedBuffers_.find(fullPath);
    if (cacheIt != resolvedBuffers_.end()) {
      return ResolvedModule{cacheIt->second,
                            systemVFS_ ? displayName : std::string(fullPath)};
    }

    // Check the shared content cache before hitting the SIF. Each unique
    // system module needs to be decompressed at most once across all parallel
    // PE compilations.
    if (systemModuleCache_) {
      std::string cachedContent;
      bool found = false;
      {
        std::lock_guard<std::mutex> lock(systemModuleCache_->contentMutex);
        auto it = systemModuleCache_->fileContentCache.find(fullPath);
        if (it != systemModuleCache_->fileContentCache.end()) {
          cachedContent = it->second;
          found = true;
        }
      }
      if (found) {
        llvm::StringRef bufName =
            systemVFS_ ? llvm::StringRef(displayName) : fullPath.str();
        auto buffer =
            llvm::MemoryBuffer::getMemBufferCopy(cachedContent, bufName);
        unsigned bufferID =
            sourceMgr_->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
        resolvedBuffers_[fullPath] = bufferID;
        return ResolvedModule{bufferID,
                              systemVFS_ ? displayName : std::string(fullPath)};
      }
    }

    // Use the system VFS when available, otherwise use the SourceMgr's VFS
    // (which may have an editor overlay), and finally fall back to disk.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
        std::error_code(ENOENT, std::generic_category());

    if (systemVFS_) {
      bufferOrErr = systemVFS_->getBufferForFile(fullPath);
    } else if (sourceMgr_->getVirtualFileSystem()) {
      bufferOrErr =
          sourceMgr_->getVirtualFileSystem()->getBufferForFile(fullPath);
    } else {
      bufferOrErr = llvm::MemoryBuffer::getFile(fullPath);
    }

    if (bufferOrErr) {
      // Store raw content in the shared cache so future resolvers skip the SIF.
      if (systemModuleCache_) {
        std::lock_guard<std::mutex> lock(systemModuleCache_->contentMutex);
        (void)systemModuleCache_->fileContentCache.try_emplace(
            fullPath.str(), (*bufferOrErr)->getBuffer().str());
      }
      // For SIF-backed modules, re-wrap the buffer with the display name as
      // identifier so that diagnostic output shows "<memcpy/memcpy>" instead
      // of the absolute path inside the container.
      if (systemVFS_) {
        *bufferOrErr = llvm::MemoryBuffer::getMemBufferCopy(
            (*bufferOrErr)->getBuffer(), displayName);
      }
      unsigned bufferID = sourceMgr_->AddNewSourceBuffer(
          std::move(*bufferOrErr), llvm::SMLoc());
      resolvedBuffers_[fullPath] = bufferID;
      return ResolvedModule{bufferID,
                            systemVFS_ ? displayName : std::string(fullPath)};
    }
  }

  return mlir::failure();
}

unsigned
CslModuleResolver::addBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  return sourceMgr_->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
}

void CslModuleResolver::applyEditorOverlay(
    const llvm::StringMap<std::string> &overrideBuffers) {
  if (overrideBuffers.empty())
    return;

  auto inMemFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  for (const auto &[overridePath, overrideContents] : overrideBuffers) {
    inMemFS->addFile(
        overridePath, /*ModificationTime=*/0,
        llvm::MemoryBuffer::getMemBufferCopy(overrideContents, overridePath));
  }

  // Use the existing SourceMgr VFS as base so that multiple overlay calls
  // compose correctly.
  auto baseFS = sourceMgr_->getVirtualFileSystem();
  if (!baseFS)
    baseFS = llvm::vfs::getRealFileSystem();
  auto overlay =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(baseFS);
  overlay->pushOverlay(std::move(inMemFS));

  // Apply to SourceMgr for user imports.
  sourceMgr_->setVirtualFileSystem(overlay);

  // Apply to system VFS if not already set to a custom VFS (like
  // SifFileSystem).
  if (!systemVFS_)
    systemVFS_ = overlay;
}

void CslModuleResolver::setSystemVFS(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
  systemVFS_ = std::move(vfs);
}
