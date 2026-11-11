add_mlir_tool(csl-s2s
  csl-s2s.cpp

  DEPENDS
  
)

target_link_libraries(csl-s2s
  PRIVATE
  MLIRCSLParser
  MLIRCSLPipelines
  MLIRCSLUtils
  MLIRParsedCSLDialect
  MLIRParsedCSLTransforms
  MLIRParsedCSLToTypedCSL
  MLIRTypedCSLDialect
  CSLEmitCSL
  MLIRParser
  MLIRSupport
  MLIRIR
)
