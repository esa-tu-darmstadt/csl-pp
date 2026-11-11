//===- SifFileSystem.h - VFS over SIF container -------------------*- C++ -*-=//
//
// A read-only llvm::vfs::FileSystem backed by a SIF container's Squashfs
// partition. Can be plugged into CslModuleResolver::setSystemVFS().
//
//===------------------------------------------------------------------------===//

#ifndef CSL_MLIR_SUPPORT_SIF_SIFFILESYSTEM_H
#define CSL_MLIR_SUPPORT_SIF_SIFFILESYSTEM_H

#include "csl-mlir/Support/Sif/SifFile.h"
#include "csl-mlir/Support/Sif/SquashfsReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>
#include <string>

namespace csl::sif {

/// A read-only virtual filesystem backed by a SIF container's Squashfs
/// partition. Opens the SIF file once, locates the Squashfs partition,
/// and serves all subsequent requests from the cached reader.
///
/// Usage:
///   auto sifFS = SifFileSystem::create("/path/to/sdk.sif");
///   resolver->setSystemVFS(std::move(sifFS));
class SifFileSystem : public llvm::vfs::FileSystem {
public:
  /// Create a SifFileSystem from a SIF container file.
  static llvm::Expected<llvm::IntrusiveRefCntPtr<SifFileSystem>>
  create(llvm::StringRef sifPath);

  llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &path) override;

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const llvm::Twine &path) override;

  llvm::vfs::directory_iterator dir_begin(const llvm::Twine &dir,
                                          std::error_code &ec) override;

  std::error_code setCurrentWorkingDirectory(const llvm::Twine &path) override;
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override;
  bool exists(const llvm::Twine &path) override;

private:
  SifFileSystem(std::unique_ptr<SifFile> sif,
                std::unique_ptr<SquashfsReader> reader);

  std::unique_ptr<SifFile> sif_;
  std::unique_ptr<SquashfsReader> reader_;
  std::string workingDirectory_;
};

} // namespace csl::sif

#endif // CSL_MLIR_SUPPORT_SIF_SIFFILESYSTEM_H
