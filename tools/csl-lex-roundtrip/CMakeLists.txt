add_mlir_tool(csl-lex-roundtrip
  csl-lex-roundtrip.cpp

  DEPENDS

)

target_link_libraries(csl-lex-roundtrip
  PRIVATE
  MLIRCSLParser
  MLIRSupport
)
