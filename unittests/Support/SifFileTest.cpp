//===- SifFileTest.cpp - Unit tests for SIF reader library ------*- C++ -*-===//
//
// Tests SifFile parsing, SquashfsReader file/directory operations, and
// the SifFileSystem VFS adapter using the "one-group.sif" test fixture
// from the Go reference implementation.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Support/Sif/SifFile.h"
#include "csl-mlir/Support/Sif/SifFileSystem.h"
#include "csl-mlir/Support/Sif/SquashfsReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <string>

using namespace csl::sif;

#ifndef SIF_TEST_DATA_DIR
#error "SIF_TEST_DATA_DIR must be defined by CMake"
#endif

static std::string getTestSifPath() {
  return std::string(SIF_TEST_DATA_DIR) + "/one-group.sif";
}

//===----------------------------------------------------------------------===//
// SifFile tests
//===----------------------------------------------------------------------===//

TEST(SifFileTest, OpenAndParseHeader) {
  auto sifOrErr = SifFile::open(getTestSifPath());
  ASSERT_THAT_EXPECTED(sifOrErr, llvm::Succeeded());

  auto &sif = *sifOrErr;
  const auto &header = sif->getHeader();

  // Verify magic.
  EXPECT_EQ(std::string(header.magic, 10), std::string("SIF_MAGIC\0", 10));

  // Verify version.
  EXPECT_EQ(header.version[0], '0');
  EXPECT_EQ(header.version[1], '1');

  // Should have some descriptors.
  EXPECT_GT(sif->getDescriptors().size(), 0u);
}

TEST(SifFileTest, FindSquashfsPartition) {
  auto sifOrErr = SifFile::open(getTestSifPath());
  ASSERT_THAT_EXPECTED(sifOrErr, llvm::Succeeded());

  const SifDescriptor *partition = (*sifOrErr)->findSquashfsPartition();
  ASSERT_NE(partition, nullptr);

  // Copy fields out of the packed struct to avoid misaligned references
  // in GTest macros (SifDescriptor is __attribute__((packed)), 585 bytes,
  // so vector elements beyond the first have misaligned fields).
  int64_t offset = partition->offset;
  int64_t size = partition->size;
  int32_t dataType = partition->dataType;

  // The partition should have a positive offset and size.
  EXPECT_GT(offset, 0);
  EXPECT_GT(size, 0);

  // Data type should be Partition.
  EXPECT_EQ(dataType, static_cast<int32_t>(SifDataType::Partition));
}

TEST(SifFileTest, OpenInvalidFile) {
  auto sifOrErr = SifFile::open("/nonexistent/path.sif");
  ASSERT_THAT_EXPECTED(sifOrErr, llvm::Failed());
}

//===----------------------------------------------------------------------===//
// SquashfsReader tests
//===----------------------------------------------------------------------===//

class SquashfsReaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto sifOrErr = SifFile::open(getTestSifPath());
    ASSERT_THAT_EXPECTED(sifOrErr, llvm::Succeeded());
    sif = std::move(*sifOrErr);

    const SifDescriptor *partition = sif->findSquashfsPartition();
    ASSERT_NE(partition, nullptr);

    auto readerOrErr = SquashfsReader::open(sif->getPath(), partition->offset,
                                            partition->size);
    ASSERT_THAT_EXPECTED(readerOrErr, llvm::Succeeded());
    reader = std::move(*readerOrErr);
  }

  std::unique_ptr<SifFile> sif;
  std::unique_ptr<SquashfsReader> reader;
};

TEST_F(SquashfsReaderTest, ReadFile) {
  auto bufOrErr = reader->readFile("/hello.txt");
  ASSERT_THAT_EXPECTED(bufOrErr, llvm::Succeeded());

  auto &buf = *bufOrErr;
  llvm::StringRef content(buf->getBufferStart(), buf->getBufferSize());
  // The test fixture contains "Hello from Sylabs!\n" (or without newline).
  EXPECT_TRUE(content.starts_with("Hello from Sylabs!"));
}

TEST_F(SquashfsReaderTest, ListRootDirectory) {
  auto entriesOrErr = reader->listDirectory("/");
  ASSERT_THAT_EXPECTED(entriesOrErr, llvm::Succeeded());

  auto &entries = *entriesOrErr;
  // Should contain at least hello.txt.
  bool foundHello = false;
  for (const auto &entry : entries) {
    if (entry.name == "hello.txt") {
      foundHello = true;
      EXPECT_FALSE(entry.isDirectory);
    }
  }
  EXPECT_TRUE(foundHello) << "Expected hello.txt in root directory";
}

TEST_F(SquashfsReaderTest, Exists) {
  EXPECT_TRUE(reader->exists("/hello.txt"));
  EXPECT_TRUE(reader->exists("/"));
  EXPECT_FALSE(reader->exists("/nonexistent.txt"));
}

TEST_F(SquashfsReaderTest, Stat) {
  auto infoOrErr = reader->stat("/hello.txt");
  ASSERT_THAT_EXPECTED(infoOrErr, llvm::Succeeded());

  auto &info = *infoOrErr;
  EXPECT_EQ(info.name, "hello.txt");
  EXPECT_FALSE(info.isDirectory);
  EXPECT_GT(info.size, 0u);
}

TEST_F(SquashfsReaderTest, StatRoot) {
  auto infoOrErr = reader->stat("/");
  ASSERT_THAT_EXPECTED(infoOrErr, llvm::Succeeded());
  EXPECT_TRUE(infoOrErr->isDirectory);
}

TEST_F(SquashfsReaderTest, ReadNonexistentFile) {
  auto bufOrErr = reader->readFile("/does_not_exist.txt");
  ASSERT_THAT_EXPECTED(bufOrErr, llvm::Failed());
}

//===----------------------------------------------------------------------===//
// SifFileSystem (VFS) tests
//===----------------------------------------------------------------------===//

TEST(SifFileSystemTest, CreateAndReadFile) {
  auto fsOrErr = SifFileSystem::create(getTestSifPath());
  ASSERT_THAT_EXPECTED(fsOrErr, llvm::Succeeded());

  auto &fs = *fsOrErr;

  // Read a file through the VFS interface.
  auto fileOrErr = fs->openFileForRead("/hello.txt");
  ASSERT_FALSE(fileOrErr.getError());

  auto bufOrErr = (*fileOrErr)->getBuffer("hello.txt", -1, false, false);
  ASSERT_FALSE(bufOrErr.getError());

  llvm::StringRef content((*bufOrErr)->getBufferStart(),
                          (*bufOrErr)->getBufferSize());
  EXPECT_TRUE(content.starts_with("Hello from Sylabs!"));
}

TEST(SifFileSystemTest, Status) {
  auto fsOrErr = SifFileSystem::create(getTestSifPath());
  ASSERT_THAT_EXPECTED(fsOrErr, llvm::Succeeded());

  auto statusOrErr = (*fsOrErr)->status("/hello.txt");
  ASSERT_FALSE(statusOrErr.getError());
  EXPECT_FALSE(statusOrErr->isDirectory());
  EXPECT_GT(statusOrErr->getSize(), 0u);
}

TEST(SifFileSystemTest, Exists) {
  auto fsOrErr = SifFileSystem::create(getTestSifPath());
  ASSERT_THAT_EXPECTED(fsOrErr, llvm::Succeeded());

  EXPECT_TRUE((*fsOrErr)->exists("/hello.txt"));
  EXPECT_FALSE((*fsOrErr)->exists("/nonexistent"));
}

TEST(SifFileSystemTest, DirectoryIteration) {
  auto fsOrErr = SifFileSystem::create(getTestSifPath());
  ASSERT_THAT_EXPECTED(fsOrErr, llvm::Succeeded());

  std::error_code ec;
  auto it = (*fsOrErr)->dir_begin("/", ec);
  ASSERT_FALSE(ec);

  bool foundHello = false;
  while (it != llvm::vfs::directory_iterator()) {
    if (it->path().ends_with("hello.txt"))
      foundHello = true;
    it.increment(ec);
    ASSERT_FALSE(ec);
  }
  EXPECT_TRUE(foundHello);
}

//===----------------------------------------------------------------------===//
// SDK SIF integration tests (optional, requires CMake configuration)
//===----------------------------------------------------------------------===//

#ifdef CSL_SDK_SIF_TESTS_ENABLED

TEST(SdkSifTest, ReadTileConfigViaVFS) {
  auto fsOrErr = SifFileSystem::create(CSL_SDK_SIF_PATH);
  ASSERT_THAT_EXPECTED(fsOrErr, llvm::Succeeded());

  auto &fs = *fsOrErr;

  // The CSL libs directory should exist.
  std::string cslLibsPath = CSL_SDK_SIF_CSL_LIBS_PATH;
  ASSERT_TRUE(fs->exists(cslLibsPath))
      << "CSL libs directory not found: " << cslLibsPath;

  auto statusOrErr = fs->status(cslLibsPath);
  ASSERT_FALSE(statusOrErr.getError()) << "Failed to stat CSL libs directory: "
                                       << statusOrErr.getError().message();
  EXPECT_TRUE(statusOrErr->isDirectory());

  // tile_config.csl should exist within the CSL libs directory.
  std::string tileConfigPath = cslLibsPath;
  if (!tileConfigPath.empty() && tileConfigPath.back() != '/')
    tileConfigPath += '/';
  tileConfigPath += "tile_config.csl";

  ASSERT_TRUE(fs->exists(tileConfigPath))
      << "tile_config.csl not found at: " << tileConfigPath;

  // Read the file through the VFS and verify it has content.
  auto fileOrErr = fs->openFileForRead(tileConfigPath);
  ASSERT_FALSE(fileOrErr.getError())
      << "Failed to open tile_config.csl: " << fileOrErr.getError().message();

  auto bufOrErr = (*fileOrErr)->getBuffer("tile_config.csl", -1, false, false);
  ASSERT_FALSE(bufOrErr.getError())
      << "Failed to read tile_config.csl: " << bufOrErr.getError().message();

  llvm::StringRef content((*bufOrErr)->getBufferStart(),
                          (*bufOrErr)->getBufferSize());
  EXPECT_GT(content.size(), 0u) << "tile_config.csl is empty";
}

TEST(SdkSifTest, ListCslLibsDirectory) {
  auto fsOrErr = SifFileSystem::create(CSL_SDK_SIF_PATH);
  ASSERT_THAT_EXPECTED(fsOrErr, llvm::Succeeded());

  auto &fs = *fsOrErr;
  std::string cslLibsPath = CSL_SDK_SIF_CSL_LIBS_PATH;

  std::error_code ec;
  auto it = fs->dir_begin(cslLibsPath, ec);
  ASSERT_FALSE(ec) << "Failed to list CSL libs directory: " << ec.message();

  bool foundTileConfig = false;
  while (it != llvm::vfs::directory_iterator()) {
    if (it->path().ends_with("tile_config.csl"))
      foundTileConfig = true;
    it.increment(ec);
    ASSERT_FALSE(ec);
  }
  EXPECT_TRUE(foundTileConfig)
      << "Expected tile_config.csl in CSL libs directory";
}

#endif // CSL_SDK_SIF_TESTS_ENABLED
