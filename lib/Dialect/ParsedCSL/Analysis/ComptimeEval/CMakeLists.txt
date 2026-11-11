add_mlir_library(MLIRParsedCSLComptimeEvaluator
  BuiltinEvaluators.cpp
  ComptimeInterfaces.cpp
  ComptimeEvalContext.cpp
  ComptimeModels.cpp
  GlobalEvalContext.cpp
  FunctionEvalContext.cpp
  ComptimeFunctionEvaluator.cpp
  ComptimeEvaluator.cpp
  TensorAccessAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval

  DEPENDS
  MLIRParsedCSLComptimeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCSLDialect
  MLIRCSLUtils
  MLIRParsedCSLDialect
  MLIRCSLParser
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
  )
