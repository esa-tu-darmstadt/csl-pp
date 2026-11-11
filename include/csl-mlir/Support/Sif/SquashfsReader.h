//===- SquashfsReader.h - Read files from embedded Squashfs --------*- C++
//-*-=//
//
// Wraps libsquashfs to read files and directories from a Squashfs filesystem
// image embedded at an offset within a host file (e.g., a SIF container).
//
//===------------------------------------------------------------------------===//

#ifndef CSL_MLIR_SUPPORT_SIF_SQUASHFSREADER_H
#define CSL_MLIR_SUPPORT_SIF_SQUASHFSREADER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace csl::sif {

/// Reads files and directories from a Squashfs filesystem embedded at a
/// given offset within a host file.
///
/// Caches all libsquashfs handles (file, superblock, compressor, dir_reader,
/// data_reader) for the lifetime of the reader so that repeated operations
/// do not re-initialize the Squashfs state.
///
/// Thread safety: individual operations are serialized via an internal mutex.
class SquashfsReader {
public:
  /// Information about a filesystem entry.
  struct EntryInfo {
    std::string name;
    uint64_t size = 0;
    bool isDirectory = false;
    uint32_t modTime = 0;
  };

  /// Open a Squashfs image embedded at the given byte offset and size within
  /// the file at `hostPath`.
  static llvm::Expected<std::unique_ptr<SquashfsReader>>
  open(llvm::StringRef hostPath, int64_t offset, int64_t size);

  ~SquashfsReader();

  // Not copyable or movable (owns C resources via pimpl).
  SquashfsReader(const SquashfsReader &) = delete;
  SquashfsReader &operator=(const SquashfsReader &) = delete;

  /// Read the entire contents of a file within the Squashfs image.
  /// `path` should be absolute within the image (e.g., "/foo/bar.csl").
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  readFile(llvm::StringRef path) const;

  /// List the entries in a directory within the Squashfs image.
  llvm::Expected<std::vector<EntryInfo>>
  listDirectory(llvm::StringRef path) const;

  /// Check whether a path exists in the Squashfs image.
  bool exists(llvm::StringRef path) const;

  /// Get information about a path (file or directory).
  llvm::Expected<EntryInfo> stat(llvm::StringRef path) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  SquashfsReader();
};

} // namespace csl::sif

#endif // CSL_MLIR_SUPPORT_SIF_SQUASHFSREADER_H
