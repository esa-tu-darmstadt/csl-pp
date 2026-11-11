add_mlir_tool(csl-pp
  csl-pp.cpp

  DEPENDS
  
)

target_link_libraries(csl-pp
  PRIVATE
  MLIRCSLParser
  MLIRCSLPipelines
  MLIRCSLUtils
  MLIRParsedCSLDialect
  MLIRParsedCSLTransforms
  MLIRParsedCSLToTypedCSL
  MLIRTypedCSLDialect
  MLIRTypedCSLTransforms
  CSLEmitCSL
  MLIRParser
  MLIRSupport
  MLIRIR
  LLVMSupport
)
