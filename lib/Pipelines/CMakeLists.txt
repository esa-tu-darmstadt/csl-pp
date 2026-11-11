add_mlir_library(MLIRCSLPipelines
  CslPipeline.cpp
  ProcessTileCodePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Pipelines

  DEPENDS
  MLIRCSLPipelinesPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCSLDialect
  MLIRCSLUtils
  MLIRParsedCSLDialect
  MLIRParsedCSLTransforms
  MLIRParsedCSLToTypedCSL
  MLIRTypedCSLDialect
  MLIRTypedCSLTransforms
  MLIRCSLParser
  CSLEmitCSL
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  )
