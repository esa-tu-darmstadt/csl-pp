# ParsedCSL Dialect

add_mlir_dialect_library(MLIRParsedCSLDialect
  ParsedCSLCanonicalize.cpp
  ParsedCSLDialect.cpp
  ParsedCSLOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/ParsedCSL/IR

  DEPENDS
  MLIRParsedCSLDialectIncGen
  MLIRParsedCSLTypesIncGen
  MLIRParsedCSLAttrsIncGen
  MLIRParsedCSLInterfacesIncGen
  MLIRParsedCSLOpsIncGen
  MLIRCSLDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRCSLDialect
  )
