add_mlir_library(MLIRParsedCSLToTypedCSL
  LowerToTypedCSL.cpp  # Old implementation (kept for reference)
  # LowerToTypedCSLNew.cpp  # New attribute-based implementation

  PARTIAL_SOURCES_INTENDED  # Allow uncommitted source files in directory

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Conversion/ParsedCSLToTypedCSL

  DEPENDS
  MLIRCSLConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRParsedCSLDialect
  MLIRParsedCSLComptimeEvaluator
  MLIRTypedCSLDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
