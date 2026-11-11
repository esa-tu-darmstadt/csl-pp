//===- SquashfsReader.cpp - Read files from embedded Squashfs -------------===//
//
// Wraps libsquashfs to provide file and directory reading from a Squashfs
// image embedded at an arbitrary offset within a host file.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Support/Sif/SquashfsReader.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
extern "C" {
#include <sqfs/compressor.h>
#include <sqfs/data_reader.h>
#include <sqfs/dir.h>
#include <sqfs/dir_reader.h>
#include <sqfs/error.h>
#include <sqfs/inode.h>
#include <sqfs/io.h>
#include <sqfs/predef.h>
#include <sqfs/super.h>
}
#pragma GCC diagnostic pop

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

using namespace csl::sif;

//===----------------------------------------------------------------------===//
// OffsetFile — custom sqfs_file_t that translates offsets
//===----------------------------------------------------------------------===//

namespace {

/// A sqfs_file_t implementation that wraps another sqfs_file_t and translates
/// all read offsets by a fixed base amount. This lets libsquashfs read a
/// Squashfs image embedded inside a larger file (the SIF container) without
/// knowing about the container format.
struct OffsetFile {
  sqfs_file_t base;
  sqfs_file_t *hostFile;
  sqfs_u64 baseOffset;
  sqfs_u64 partitionSize;
};

static int offsetFileReadAt(sqfs_file_t *file, sqfs_u64 offset, void *buffer,
                            size_t size) {
  auto *of = reinterpret_cast<OffsetFile *>(file);
  if (offset + size > of->partitionSize)
    return SQFS_ERROR_OUT_OF_BOUNDS;
  return of->hostFile->read_at(of->hostFile, of->baseOffset + offset, buffer,
                               size);
}

static int offsetFileWriteAt(sqfs_file_t * /*file*/, sqfs_u64 /*offset*/,
                             const void * /*buffer*/, size_t /*size*/) {
  return SQFS_ERROR_UNSUPPORTED;
}

static sqfs_u64 offsetFileGetSize(const sqfs_file_t *file) {
  auto *of = reinterpret_cast<const OffsetFile *>(file);
  return of->partitionSize;
}

static int offsetFileTruncate(sqfs_file_t * /*file*/, sqfs_u64 /*size*/) {
  return SQFS_ERROR_UNSUPPORTED;
}

static void offsetFileDestroy(sqfs_object_t *obj) {
  auto *of = reinterpret_cast<OffsetFile *>(obj);
  sqfs_destroy(of->hostFile);
  delete of;
}

/// Create an OffsetFile wrapping `hostFile` at the given offset/size.
/// Takes ownership of `hostFile` (caller must not destroy it).
static sqfs_file_t *createOffsetFile(sqfs_file_t *hostFile, sqfs_u64 offset,
                                     sqfs_u64 size) {
  auto *of = new OffsetFile();
  std::memset(&of->base, 0, sizeof(of->base));

  of->base.base.destroy = offsetFileDestroy;
  of->base.base.copy = nullptr;

  of->base.read_at = offsetFileReadAt;
  of->base.write_at = offsetFileWriteAt;
  of->base.get_size = offsetFileGetSize;
  of->base.truncate = offsetFileTruncate;

  of->hostFile = hostFile;
  of->baseOffset = offset;
  of->partitionSize = size;
  return &of->base;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SquashfsReader::Impl — holds all libsquashfs state
//===----------------------------------------------------------------------===//

struct SquashfsReader::Impl {
  sqfs_file_t *file = nullptr;
  sqfs_super_t super{};
  sqfs_compressor_t *compressor = nullptr;
  sqfs_dir_reader_t *dirReader = nullptr;
  sqfs_data_reader_t *dataReader = nullptr;
  mutable std::mutex mutex;

  ~Impl() {
    if (dataReader)
      sqfs_destroy(dataReader);
    if (dirReader)
      sqfs_destroy(dirReader);
    if (compressor)
      sqfs_destroy(compressor);
    if (file)
      sqfs_destroy(file);
  }
};

//===----------------------------------------------------------------------===//
// SquashfsReader — construction / destruction
//===----------------------------------------------------------------------===//

SquashfsReader::SquashfsReader() : impl_(std::make_unique<Impl>()) {}
SquashfsReader::~SquashfsReader() = default;

llvm::Expected<std::unique_ptr<SquashfsReader>>
SquashfsReader::open(llvm::StringRef hostPath, int64_t offset, int64_t size) {
  // Open the host file (the SIF container).
  sqfs_file_t *hostFile =
      sqfs_open_file(hostPath.str().c_str(), SQFS_FILE_OPEN_READ_ONLY);
  if (!hostFile)
    return llvm::createStringError("failed to open '%s' for squashfs reading",
                                   hostPath.str().c_str());

  // Wrap it in an offset-translating file view.
  // createOffsetFile takes ownership of hostFile.
  sqfs_file_t *offsetFile = createOffsetFile(
      hostFile, static_cast<sqfs_u64>(offset), static_cast<sqfs_u64>(size));

  auto reader = std::unique_ptr<SquashfsReader>(new SquashfsReader());
  auto &impl = *reader->impl_;
  impl.file = offsetFile;

  // Read superblock.
  int err = sqfs_super_read(&impl.super, impl.file);
  if (err)
    return llvm::createStringError(
        "failed to read squashfs superblock: error %d", err);

  // Create decompressor.
  sqfs_compressor_config_t cfg;
  sqfs_compressor_config_init(
      &cfg, static_cast<SQFS_COMPRESSOR>(impl.super.compression_id),
      impl.super.block_size, SQFS_COMP_FLAG_UNCOMPRESS);
  err = sqfs_compressor_create(&cfg, &impl.compressor);
  if (err)
    return llvm::createStringError(
        "failed to create squashfs decompressor: error %d", err);

  // Create directory reader.
  impl.dirReader =
      sqfs_dir_reader_create(&impl.super, impl.compressor, impl.file, 0);
  if (!impl.dirReader)
    return llvm::createStringError(
        "failed to create squashfs directory reader");

  // Create data reader.
  impl.dataReader = sqfs_data_reader_create(impl.file, impl.super.block_size,
                                            impl.compressor, 0);
  if (!impl.dataReader)
    return llvm::createStringError("failed to create squashfs data reader");

  // Load fragment table.
  err = sqfs_data_reader_load_fragment_table(impl.dataReader, &impl.super);
  if (err)
    return llvm::createStringError(
        "failed to load squashfs fragment table: error %d", err);

  return reader;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Resolve a path to a deserialized inode. Caller must sqfs_free() the result.
llvm::Expected<sqfs_inode_generic_t *>
resolveInode(sqfs_dir_reader_t *dirReader, llvm::StringRef path) {
  sqfs_inode_generic_t *root = nullptr;
  int err = sqfs_dir_reader_get_root_inode(dirReader, &root);
  if (err)
    return llvm::createStringError(
        "failed to get squashfs root inode: error %d", err);

  if (path.empty() || path == "/")
    return root;

  sqfs_inode_generic_t *inode = nullptr;
  std::string pathStr = path.str();
  err = sqfs_dir_reader_find_by_path(dirReader, root, pathStr.c_str(), &inode);
  sqfs_free(root);
  if (err)
    return llvm::createStringError(
        "path not found in squashfs: '%s' (error %d)", pathStr.c_str(), err);

  return inode;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SquashfsReader — public API
//===----------------------------------------------------------------------===//

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
SquashfsReader::readFile(llvm::StringRef path) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  auto inodeOrErr = resolveInode(impl_->dirReader, path);
  if (!inodeOrErr)
    return inodeOrErr.takeError();

  sqfs_inode_generic_t *inode = *inodeOrErr;

  sqfs_u64 fileSize = 0;
  int err = sqfs_inode_get_file_size(inode, &fileSize);
  if (err) {
    sqfs_free(inode);
    return llvm::createStringError("'%s' is not a regular file",
                                   path.str().c_str());
  }

  auto buf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
      static_cast<size_t>(fileSize), path);
  if (!buf) {
    sqfs_free(inode);
    return llvm::createStringError(
        "failed to allocate buffer for '%s' (%llu bytes)", path.str().c_str(),
        static_cast<unsigned long long>(fileSize));
  }

  sqfs_u64 bytesRead = 0;
  while (bytesRead < fileSize) {
    sqfs_u32 chunkSize = static_cast<sqfs_u32>(
        std::min<sqfs_u64>(fileSize - bytesRead, 1024 * 1024));
    sqfs_s32 ret =
        sqfs_data_reader_read(impl_->dataReader, inode, bytesRead,
                              buf->getBufferStart() + bytesRead, chunkSize);
    if (ret < 0) {
      sqfs_free(inode);
      return llvm::createStringError(
          "failed to read '%s' at offset %llu: error %d", path.str().c_str(),
          static_cast<unsigned long long>(bytesRead), ret);
    }
    if (ret == 0)
      break;
    bytesRead += static_cast<sqfs_u64>(ret);
  }

  sqfs_free(inode);
  return std::unique_ptr<llvm::MemoryBuffer>(std::move(buf));
}

llvm::Expected<std::vector<SquashfsReader::EntryInfo>>
SquashfsReader::listDirectory(llvm::StringRef path) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  auto inodeOrErr = resolveInode(impl_->dirReader, path);
  if (!inodeOrErr)
    return inodeOrErr.takeError();

  sqfs_inode_generic_t *inode = *inodeOrErr;

  int err = sqfs_dir_reader_open_dir(impl_->dirReader, inode, 0);
  sqfs_free(inode);
  if (err)
    return llvm::createStringError("failed to open directory '%s': error %d",
                                   path.str().c_str(), err);

  std::vector<EntryInfo> entries;
  sqfs_dir_entry_t *entry = nullptr;

  while ((err = sqfs_dir_reader_read(impl_->dirReader, &entry)) == 0) {
    EntryInfo info;
    // entry->name is NOT null-terminated; entry->size stores (length - 1).
    info.name = std::string(reinterpret_cast<const char *>(entry->name),
                            static_cast<size_t>(entry->size) + 1);
    info.isDirectory =
        (entry->type == SQFS_INODE_DIR || entry->type == SQFS_INODE_EXT_DIR);
    info.size = 0;
    info.modTime = 0;
    sqfs_free(entry);
    entries.push_back(std::move(info));
  }

  // err > 0 means end-of-directory; err < 0 means error.
  if (err < 0)
    return llvm::createStringError("error reading directory '%s': error %d",
                                   path.str().c_str(), err);

  return entries;
}

bool SquashfsReader::exists(llvm::StringRef path) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto inodeOrErr = resolveInode(impl_->dirReader, path);
  if (!inodeOrErr) {
    llvm::consumeError(inodeOrErr.takeError());
    return false;
  }
  sqfs_free(*inodeOrErr);
  return true;
}

llvm::Expected<SquashfsReader::EntryInfo>
SquashfsReader::stat(llvm::StringRef path) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  auto inodeOrErr = resolveInode(impl_->dirReader, path);
  if (!inodeOrErr)
    return inodeOrErr.takeError();

  sqfs_inode_generic_t *inode = *inodeOrErr;
  EntryInfo info;
  info.name = llvm::sys::path::filename(path).str();
  info.isDirectory = (inode->base.type == SQFS_INODE_DIR ||
                      inode->base.type == SQFS_INODE_EXT_DIR);
  info.modTime = inode->base.mod_time;
  info.size = 0;
  if (!info.isDirectory) {
    sqfs_u64 sz = 0;
    if (sqfs_inode_get_file_size(inode, &sz) == 0)
      info.size = sz;
  }

  sqfs_free(inode);
  return info;
}
