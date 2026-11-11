//===- SifFile.h - SIF Container Reader ---------------------------*- C++
//-*-===//
//
// Low-level parser for Singularity Image Format (SIF) container files.
// Reads and caches the SIF header and descriptor table, allowing random
// access to individual data objects (partitions, signatures, etc.) without
// re-reading the header each time.
//
//===------------------------------------------------------------------------===//

#ifndef CSL_MLIR_SUPPORT_SIF_SIFFILE_H
#define CSL_MLIR_SUPPORT_SIF_SIFFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace csl::sif {

//===----------------------------------------------------------------------===//
// SIF Binary Format Structures (little-endian, packed)
//===----------------------------------------------------------------------===//

/// SIF global file header. 128 bytes total.
struct __attribute__((packed)) SifHeader {
  char launchScript[32];
  char magic[10];  // "SIF_MAGIC\0"
  char version[3]; // "01\0"
  char arch[3];
  uint8_t uuid[16];
  int64_t createdAt;
  int64_t modifiedAt;
  int64_t descriptorsFree;
  int64_t descriptorsTotal;
  int64_t descriptorsOffset;
  int64_t descriptorsSize;
  int64_t dataOffset;
  int64_t dataSize;
};
static_assert(sizeof(SifHeader) == 128, "SifHeader must be exactly 128 bytes");

/// SIF data object descriptor. 585 bytes total, no alignment padding.
struct __attribute__((packed)) SifDescriptor {
  int32_t dataType;
  uint8_t used;
  uint32_t id;
  uint32_t groupId;
  uint32_t linkedId;
  int64_t offset; // Absolute file offset to the data object.
  int64_t size;
  int64_t sizeWithPadding;
  int64_t createdAt;
  int64_t modifiedAt;
  int64_t uid;
  int64_t gid;
  char name[128];
  char extra[384];
};
static_assert(sizeof(SifDescriptor) == 585,
              "SifDescriptor must be exactly 585 bytes");

/// Partition-specific metadata stored at the start of SifDescriptor::extra.
struct __attribute__((packed)) SifPartitionExtra {
  int32_t fsType;
  int32_t partType;
  char arch[3];
};

/// Known SIF descriptor data types.
enum class SifDataType : int32_t {
  Deffile = 0x4001,
  EnvVar = 0x4002,
  Labels = 0x4003,
  Partition = 0x4004,
  Signature = 0x4005,
  GenericJSON = 0x4006,
  Generic = 0x4007,
  CryptoMessage = 0x4008,
  SBOM = 0x4009,
  OCIRootIndex = 0x400a,
  OCIBlob = 0x400b,
};

/// Known filesystem types within a partition descriptor.
enum class SifFsType : int32_t {
  Squashfs = 1,
  Ext3 = 2,
  ImmuObj = 3,
  Raw = 4,
  EncryptedSquashfs = 5,
  GocryptfsSquashfs = 6,
};

/// Known partition types.
enum class SifPartType : int32_t {
  System = 1,
  PrimSys = 2,
  Data = 3,
  Overlay = 4,
};

//===----------------------------------------------------------------------===//
// SifFile — Parsed SIF container (header + descriptors)
//===----------------------------------------------------------------------===//

/// Opens a SIF file, reads its header and all descriptors once at
/// construction time, then provides cached access to the parsed metadata.
/// Immutable after construction.
class SifFile {
public:
  /// Open and parse a SIF file.
  static llvm::Expected<std::unique_ptr<SifFile>> open(llvm::StringRef path);

  const SifHeader &getHeader() const { return header_; }
  llvm::ArrayRef<SifDescriptor> getDescriptors() const { return descriptors_; }
  llvm::StringRef getPath() const { return path_; }

  /// Find the first Squashfs partition descriptor, or nullptr if none.
  const SifDescriptor *findSquashfsPartition() const;

  /// Find a descriptor by name, or nullptr if not found.
  const SifDescriptor *findDescriptorByName(llvm::StringRef name) const;

private:
  SifFile() = default;

  std::string path_;
  SifHeader header_{};
  std::vector<SifDescriptor> descriptors_;
};

} // namespace csl::sif

#endif // CSL_MLIR_SUPPORT_SIF_SIFFILE_H
