add_mlir_dialect_library(MLIRParsedCSLTransforms
  ComptimeEvalPass.cpp
  FinalizeInstantiationPass.cpp
  InjectCoroRuntime.cpp
  InlineModulesPass.cpp
  LowerTensorAccessPass.cpp
  NormalizeCasts.cpp
  ResolveReads.cpp
  ValidateParsedCSL.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/ParsedCSL

  DEPENDS
  MLIRParsedCSLTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCSLUtils
  MLIRParsedCSLDialect
  MLIRParsedCSLComptimeEvaluator
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  )
