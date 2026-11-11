add_mlir_library(CSLEmitCSL
  EmitCSL.cpp

  ADDITIONAL_HEADER_DIRS
  ${CSL_MLIR_SOURCE_DIR}/include/csl-mlir/Target/CSL/EmitCSL

  DEPENDS
  MLIRCSLConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRCSLDialect
  MLIRTypedCSLDialect
)
