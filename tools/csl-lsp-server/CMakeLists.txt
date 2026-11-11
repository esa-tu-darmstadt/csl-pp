set(LIBS
  CslLspServerLib
)

add_mlir_tool(csl-lsp-server
  csl-lsp-server.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(csl-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(csl-lsp-server)
