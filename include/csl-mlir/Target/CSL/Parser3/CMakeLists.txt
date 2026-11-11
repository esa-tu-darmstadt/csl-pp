set(LLVM_TARGET_DEFINITIONS TokenKind.td)
mlir_tablegen(TokenKind.h.inc -gen-enum-decls)
mlir_tablegen(TokenKind.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRParser3TokenKindIncGen)

set(LLVM_TARGET_DEFINITIONS SyntaxKind.td)
mlir_tablegen(SyntaxKind.h.inc -gen-enum-decls)
mlir_tablegen(SyntaxKind.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRParser3SyntaxKindIncGen)
