add_mlir_dialect_library(MLIRTypedCSLAnalysis
  IndirectCallTargets.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/TypedCSL/Analysis

  DEPENDS
  MLIRTypedCSLDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTypedCSLDialect
  )
