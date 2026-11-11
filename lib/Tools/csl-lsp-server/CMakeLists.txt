llvm_add_library(CslLspServerLib
  Completion.cpp
  CompilationContext.cpp
  CslLspServerMain.cpp
  LSPServer.cpp
  LspUtils.cpp
  CSLServer.cpp
  SignatureHelp.cpp

  ADDITIONAL_HEADER_DIRS
  ${CSL_MLIR_SOURCE_DIR}/include/csl-mlir/Tools/csl-lsp-server

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  CSLSifReader
  MLIRCSLParser
  MLIRCSLPipelines
  MLIRCSLUtils
  MLIRParsedCSLDialect
  MLIRParsedCSLTransforms
  MLIRTypedCSLDialect
  MLIRLspServerSupportLib
  MLIRIR
  MLIRSupport
)
