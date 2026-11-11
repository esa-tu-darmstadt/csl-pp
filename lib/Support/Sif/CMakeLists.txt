add_mlir_library(CSLSifReader
  SifFile.cpp
  SquashfsReader.cpp
  SifFileSystem.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Support/Sif
  LINK_COMPONENTS Support
  LINK_LIBS PRIVATE Squashfs::Squashfs)
