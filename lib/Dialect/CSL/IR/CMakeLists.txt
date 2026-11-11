# CSL Dialect

add_mlir_dialect_library(MLIRCSLDialect
  CSLAttrs.cpp
  CSLDialect.cpp
  CSLOps.cpp
  CSLBinaryFolders.cpp
  CSLBitcastFolders.cpp
  CSLBuiltinConfig.cpp
  CSLBuiltinDescriptor.cpp
  CSLBuiltinTypes.cpp
  CSLCastFolders.cpp
  CSLUnaryFolders.cpp
  CSLPrettyPrint.cpp
  NominalTypeRegistry.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/CSL/IR

  DEPENDS
  MLIRCSLDialectIncGen
  MLIRCSLTypeInterfacesIncGen
  MLIRCSLTypesIncGen
  MLIRCSLAttrsIncGen
  MLIRCSLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
