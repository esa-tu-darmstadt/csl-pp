add_mlir_translation_library(MLIRCSLParser
    ../ParseCSL.cpp
    ParseContainerMembers.cpp
    ParseDeclarations.cpp
    ParseExpressions.cpp
    ParseStatements.cpp
    Lexer.cpp
    Parser.cpp

    DEPENDS
    MLIRParser3TokenKindIncGen
    MLIRParser3SyntaxKindIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRParsedCSLDialect
    MLIRCSLUtils
)
