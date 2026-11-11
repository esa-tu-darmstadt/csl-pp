//===- SifFile.cpp - SIF Container Reader ---------------------------------===//
//
// Parses SIF (Singularity Image Format) container headers and descriptors.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Support/Sif/SifFile.h"

#include "llvm/Support/MemoryBuffer.h"

#include <cstring>

using namespace csl::sif;

static constexpr char SIF_MAGIC[] = "SIF_MAGIC";
static constexpr size_t SIF_MAGIC_LEN = 10; // includes trailing '\0'

llvm::Expected<std::unique_ptr<SifFile>> SifFile::open(llvm::StringRef path) {
  // Read only the 128-byte header instead of the entire file (which can be
  // multiple GB for SDK SIF containers).
  auto headerOrErr =
      llvm::MemoryBuffer::getFileSlice(path, sizeof(SifHeader), /*Offset=*/0);
  if (!headerOrErr)
    return llvm::errorCodeToError(headerOrErr.getError());

  auto &headerBuf = *headerOrErr;
  if (headerBuf->getBufferSize() < sizeof(SifHeader))
    return llvm::createStringError(
        "file too small to contain SIF header: %zu bytes",
        headerBuf->getBufferSize());

  auto sif = std::unique_ptr<SifFile>(new SifFile());
  sif->path_ = path.str();

  std::memcpy(&sif->header_, headerBuf->getBufferStart(), sizeof(SifHeader));

  // Validate magic bytes.
  if (std::memcmp(sif->header_.magic, SIF_MAGIC, SIF_MAGIC_LEN) != 0)
    return llvm::createStringError("invalid SIF magic in '%s'",
                                   path.str().c_str());

  // Validate version.
  if (sif->header_.version[0] != '0' || sif->header_.version[1] != '1')
    return llvm::createStringError("unsupported SIF version '%c%c' in '%s'",
                                   sif->header_.version[0],
                                   sif->header_.version[1], path.str().c_str());

  int64_t descOffset = sif->header_.descriptorsOffset;
  int64_t descTotal = sif->header_.descriptorsTotal;

  if (descOffset < 0 || descTotal < 0)
    return llvm::createStringError(
        "invalid descriptor offset or count in SIF header");

  // Read only the descriptor table (a few KB) from its offset.
  size_t descSize = static_cast<size_t>(descTotal) * sizeof(SifDescriptor);
  auto descOrErr = llvm::MemoryBuffer::getFileSlice(
      path, descSize, static_cast<uint64_t>(descOffset));
  if (!descOrErr)
    return llvm::errorCodeToError(descOrErr.getError());

  auto &descBuf = *descOrErr;
  if (descBuf->getBufferSize() < descSize)
    return llvm::createStringError("descriptor table extends past end of file");

  const char *descData = descBuf->getBufferStart();
  sif->descriptors_.resize(static_cast<size_t>(descTotal));
  for (int64_t i = 0; i < descTotal; ++i) {
    size_t off = static_cast<size_t>(i) * sizeof(SifDescriptor);
    std::memcpy(&sif->descriptors_[static_cast<size_t>(i)], descData + off,
                sizeof(SifDescriptor));
  }

  return sif;
}

const SifDescriptor *SifFile::findSquashfsPartition() const {
  for (const auto &desc : descriptors_) {
    if (!desc.used)
      continue;
    if (desc.dataType != static_cast<int32_t>(SifDataType::Partition))
      continue;
    SifPartitionExtra partExtra;
    std::memcpy(&partExtra, desc.extra, sizeof(SifPartitionExtra));
    if (partExtra.fsType == static_cast<int32_t>(SifFsType::Squashfs))
      return &desc;
  }
  return nullptr;
}

const SifDescriptor *SifFile::findDescriptorByName(llvm::StringRef name) const {
  for (const auto &desc : descriptors_) {
    if (!desc.used)
      continue;
    llvm::StringRef descName(desc.name, strnlen(desc.name, sizeof(desc.name)));
    if (descName == name)
      return &desc;
  }
  return nullptr;
}
