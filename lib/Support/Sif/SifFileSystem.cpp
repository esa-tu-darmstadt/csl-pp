//===- SifFileSystem.cpp - VFS over SIF container -------------------------===//
//
// Implements llvm::vfs::FileSystem by delegating to SquashfsReader.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Support/Sif/SifFileSystem.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"

#include <functional>

using namespace csl::sif;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Convert an llvm::Error to std::error_code.
std::error_code toErrorCode(llvm::Error err) {
  std::error_code ec;
  llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase &e) {
    ec = e.convertToErrorCode();
    if (!ec)
      ec = llvm::errc::io_error;
  });
  return ec;
}

/// Normalize a path: make absolute relative to workingDir, clean up.
std::string normalizePath(llvm::StringRef path, llvm::StringRef workingDir) {
  llvm::SmallString<256> result;
  if (llvm::sys::path::is_absolute(path)) {
    result = path;
  } else {
    result = workingDir;
    llvm::sys::path::append(result, path);
  }
  // Remove trailing slashes (except root "/").
  while (result.size() > 1 && result.back() == '/')
    result.pop_back();
  return std::string(result);
}

/// An in-memory vfs::File backed by a MemoryBuffer read from squashfs.
class SifVfsFile : public llvm::vfs::File {
public:
  SifVfsFile(std::unique_ptr<llvm::MemoryBuffer> buffer, llvm::vfs::Status stat)
      : buffer_(std::move(buffer)), stat_(std::move(stat)) {}

  llvm::ErrorOr<llvm::vfs::Status> status() override { return stat_; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const llvm::Twine & /*name*/, int64_t /*fileSize*/,
            bool /*requiresNullTerminator*/, bool /*isVolatile*/) override {
    return std::unique_ptr<llvm::MemoryBuffer>(
        llvm::MemoryBuffer::getMemBufferCopy(buffer_->getBuffer(),
                                             buffer_->getBufferIdentifier()));
  }

  std::error_code close() override { return std::error_code(); }

private:
  std::unique_ptr<llvm::MemoryBuffer> buffer_;
  llvm::vfs::Status stat_;
};

/// Directory iterator implementation backed by a pre-fetched entry list.
class SifDirIterImpl : public llvm::vfs::detail::DirIterImpl {
public:
  SifDirIterImpl(std::vector<SquashfsReader::EntryInfo> entries,
                 std::string dirPath)
      : entries_(std::move(entries)), dirPath_(std::move(dirPath)) {
    setEntry();
  }

  std::error_code increment() override {
    ++index_;
    setEntry();
    return std::error_code();
  }

private:
  void setEntry() {
    if (index_ >= entries_.size()) {
      CurrentEntry = llvm::vfs::directory_entry();
      return;
    }
    const auto &info = entries_[index_];
    llvm::SmallString<256> fullPath(dirPath_);
    llvm::sys::path::append(fullPath, info.name);
    auto type = info.isDirectory ? llvm::sys::fs::file_type::directory_file
                                 : llvm::sys::fs::file_type::regular_file;
    CurrentEntry = llvm::vfs::directory_entry(std::string(fullPath), type);
  }

  std::vector<SquashfsReader::EntryInfo> entries_;
  std::string dirPath_;
  size_t index_ = 0;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SifFileSystem
//===----------------------------------------------------------------------===//

SifFileSystem::SifFileSystem(std::unique_ptr<SifFile> sif,
                             std::unique_ptr<SquashfsReader> reader)
    : sif_(std::move(sif)), reader_(std::move(reader)), workingDirectory_("/") {
}

llvm::Expected<llvm::IntrusiveRefCntPtr<SifFileSystem>>
SifFileSystem::create(llvm::StringRef sifPath) {
  auto sifOrErr = SifFile::open(sifPath);
  if (!sifOrErr)
    return sifOrErr.takeError();

  auto &sif = *sifOrErr;
  const SifDescriptor *partition = sif->findSquashfsPartition();
  if (!partition)
    return llvm::createStringError(
        "no Squashfs partition found in SIF file '%s'", sifPath.str().c_str());

  auto readerOrErr =
      SquashfsReader::open(sif->getPath(), partition->offset, partition->size);
  if (!readerOrErr)
    return readerOrErr.takeError();

  auto *fs = new SifFileSystem(std::move(sif), std::move(*readerOrErr));
  return llvm::IntrusiveRefCntPtr<SifFileSystem>(fs);
}

llvm::ErrorOr<llvm::vfs::Status>
SifFileSystem::status(const llvm::Twine &path) {
  std::string normalized = normalizePath(path.str(), workingDirectory_);

  auto infoOrErr = reader_->stat(normalized);
  if (!infoOrErr)
    return toErrorCode(infoOrErr.takeError());

  auto &info = *infoOrErr;
  auto type = info.isDirectory ? llvm::sys::fs::file_type::directory_file
                               : llvm::sys::fs::file_type::regular_file;

  return llvm::vfs::Status(
      normalized,
      llvm::sys::fs::UniqueID(0, std::hash<std::string>{}(normalized)),
      llvm::sys::toTimePoint(info.modTime),
      /*User=*/0, /*Group=*/0, info.size, type, llvm::sys::fs::all_read);
}

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
SifFileSystem::openFileForRead(const llvm::Twine &path) {
  std::string normalized = normalizePath(path.str(), workingDirectory_);

  auto statusOrErr = status(normalized);
  if (!statusOrErr)
    return statusOrErr.getError();

  if (statusOrErr->isDirectory())
    return std::make_error_code(std::errc::is_a_directory);

  auto bufOrErr = reader_->readFile(normalized);
  if (!bufOrErr)
    return toErrorCode(bufOrErr.takeError());

  return std::make_unique<SifVfsFile>(std::move(*bufOrErr), *statusOrErr);
}

llvm::vfs::directory_iterator SifFileSystem::dir_begin(const llvm::Twine &dir,
                                                       std::error_code &ec) {
  std::string normalized = normalizePath(dir.str(), workingDirectory_);

  auto entriesOrErr = reader_->listDirectory(normalized);
  if (!entriesOrErr) {
    ec = toErrorCode(entriesOrErr.takeError());
    return llvm::vfs::directory_iterator();
  }

  ec = std::error_code();
  return llvm::vfs::directory_iterator(
      std::make_shared<SifDirIterImpl>(std::move(*entriesOrErr), normalized));
}

std::error_code
SifFileSystem::setCurrentWorkingDirectory(const llvm::Twine &path) {
  std::string normalized = normalizePath(path.str(), workingDirectory_);
  if (!reader_->exists(normalized))
    return std::make_error_code(std::errc::no_such_file_or_directory);
  workingDirectory_ = std::move(normalized);
  return std::error_code();
}

llvm::ErrorOr<std::string> SifFileSystem::getCurrentWorkingDirectory() const {
  return workingDirectory_;
}

bool SifFileSystem::exists(const llvm::Twine &path) {
  std::string normalized = normalizePath(path.str(), workingDirectory_);
  return reader_->exists(normalized);
}
