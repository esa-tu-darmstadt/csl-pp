//===- TokenKind.td - Parser3 Token/Trivia Kind Enums -----*- tablegen -*-===//
//
// Enum definitions for the Parser3 hand-written lexer.
//
//===----------------------------------------------------------------------===//

#ifndef PARSER3_TOKEN_KIND
#define PARSER3_TOKEN_KIND

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// TriviaKind
//===----------------------------------------------------------------------===//

def Parser3_TriviaKind : I32EnumAttr<"TriviaKind",
    "Trivia kind for lexer", [
  I32EnumAttrCase<"Whitespace",          0, "whitespace">,
  I32EnumAttrCase<"LineComment",         1, "line_comment">,
  I32EnumAttrCase<"DocComment",          2, "doc_comment">,
  I32EnumAttrCase<"ContainerDocComment", 3, "container_doc_comment">,
]> {
  let cppNamespace = "::mlir::csl::parser3";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// TokenKind
//===----------------------------------------------------------------------===//

def Parser3_TokenKind : I32EnumAttr<"TokenKind",
    "Token kind for lexer", [
  I32EnumAttrCase<"Eof", 0, "eof">,

  // Keywords
  I32EnumAttrCase<"KwAlign",    1, "align">,
  I32EnumAttrCase<"KwAnd",      2, "and">,
  I32EnumAttrCase<"KwAnytype",  3, "anytype">,
  I32EnumAttrCase<"KwAsm",      4, "asm">,
  I32EnumAttrCase<"KwAsync",    37, "async">,
  I32EnumAttrCase<"KwAwait",    38, "await">,
  I32EnumAttrCase<"KwBreak",    5, "break">,
  I32EnumAttrCase<"KwComptime", 6, "comptime">,
  I32EnumAttrCase<"KwConst",    7, "const">,
  I32EnumAttrCase<"KwContinue", 8, "continue">,
  I32EnumAttrCase<"KwElse",     9, "else">,
  I32EnumAttrCase<"KwEnum",     10, "enum">,
  I32EnumAttrCase<"KwExport",   11, "export">,
  I32EnumAttrCase<"KwExtern",   12, "extern">,
  I32EnumAttrCase<"KwFn",       13, "fn">,
  I32EnumAttrCase<"KwFor",      14, "for">,
  I32EnumAttrCase<"KwIf",       15, "if">,
  I32EnumAttrCase<"KwInline",   16, "inline">,
  I32EnumAttrCase<"KwNoinline", 112, "noinline">,
  I32EnumAttrCase<"KwLayout",   17, "layout">,
  I32EnumAttrCase<"KwLinkname", 18, "linkname">,
  I32EnumAttrCase<"KwOr",       19, "or">,
  I32EnumAttrCase<"KwParam",    20, "param">,
  I32EnumAttrCase<"KwReturn",   21, "return">,
  I32EnumAttrCase<"KwStruct",   22, "struct">,
  I32EnumAttrCase<"KwSuspend",  39, "suspend">,
  I32EnumAttrCase<"KwSwitch",   23, "switch">,
  I32EnumAttrCase<"KwTask",     24, "task">,
  I32EnumAttrCase<"KwVar",      25, "var">,
  I32EnumAttrCase<"KwVolatile", 26, "volatile">,
  I32EnumAttrCase<"KwWhile",    27, "while">,
  I32EnumAttrCase<"KwTrue",     28, "true">,
  I32EnumAttrCase<"KwFalse",    29, "false">,

  // Identifiers and literals
  I32EnumAttrCase<"Identifier",       30, "identifier">,
  I32EnumAttrCase<"BuiltinIdentifier", 31, "builtin_identifier">,
  I32EnumAttrCase<"IntegerLiteral",   32, "integer_literal">,
  I32EnumAttrCase<"FloatingLiteral",  33, "floating_literal">,
  I32EnumAttrCase<"CharacterLiteral", 34, "character_literal">,
  I32EnumAttrCase<"StringLiteral",    35, "string_literal">,
  I32EnumAttrCase<"LineString",       36, "line_string">,

  // Brackets
  I32EnumAttrCase<"LParen",   40, "(">,
  I32EnumAttrCase<"RParen",   41, ")">,
  I32EnumAttrCase<"LBrace",   42, "{">,
  I32EnumAttrCase<"RBrace",   43, "}">,
  I32EnumAttrCase<"LBracket", 44, "[">,
  I32EnumAttrCase<"RBracket", 45, "]">,

  // Single-char punctuation
  I32EnumAttrCase<"Semi",        50, ";">,
  I32EnumAttrCase<"Colon",       51, ":">,
  I32EnumAttrCase<"Comma",       52, ",">,
  I32EnumAttrCase<"Dot",         53, ".">,
  I32EnumAttrCase<"Eq",          54, "=">,
  I32EnumAttrCase<"Ampersand",   55, "&">,
  I32EnumAttrCase<"Star",        56, "*">,
  I32EnumAttrCase<"Caret",       57, "^">,
  I32EnumAttrCase<"Pipe",        58, "|">,
  I32EnumAttrCase<"Plus",        59, "+">,
  I32EnumAttrCase<"Minus",       60, "-">,
  I32EnumAttrCase<"Slash",       61, "/">,
  I32EnumAttrCase<"Percent",     62, "%">,
  I32EnumAttrCase<"Tilde",       63, "~">,
  I32EnumAttrCase<"Exclamation", 64, "!">,
  I32EnumAttrCase<"Question",    65, "?">,
  I32EnumAttrCase<"Less",        66, "<">,
  I32EnumAttrCase<"Greater",     67, ">">,

  // Multi-char operators
  I32EnumAttrCase<"Arrow",         70, "->">,
  I32EnumAttrCase<"FatArrow",      71, "=>">,
  I32EnumAttrCase<"Ellipsis",      72, "...">,
  I32EnumAttrCase<"Range",         73, "..">,
  I32EnumAttrCase<"StarStar",      76, "**">,
  I32EnumAttrCase<"PlusPlus",      79, "++">,
  I32EnumAttrCase<"LeftShift",     82, "<<">,
  I32EnumAttrCase<"RightShift",    84, ">>">,
  I32EnumAttrCase<"LessEqual",     85, "<=">,
  I32EnumAttrCase<"GreaterEqual",  86, ">=">,
  I32EnumAttrCase<"EqualEqual",    87, "==">,
  I32EnumAttrCase<"NotEqual",      88, "!=">,
  I32EnumAttrCase<"DotStar",       89, ".*">,
  I32EnumAttrCase<"DotQuestion",   90, ".?">,
  I32EnumAttrCase<"PipePipe",      91, "||">,

  // Compound assignment operators
  I32EnumAttrCase<"PlusEq",          100, "+=">,
  I32EnumAttrCase<"MinusEq",         101, "-=">,
  I32EnumAttrCase<"StarEq",          102, "*=">,
  I32EnumAttrCase<"SlashEq",         103, "/=">,
  I32EnumAttrCase<"PercentEq",       104, "%=">,
  I32EnumAttrCase<"AmpersandEq",     105, "&=">,
  I32EnumAttrCase<"PipeEq",          106, "|=">,
  I32EnumAttrCase<"CaretEq",         107, "^=">,
  I32EnumAttrCase<"LeftShiftEq",     108, "<<=">,
  I32EnumAttrCase<"RightShiftEq",    109, ">>=">,

  // Error
  I32EnumAttrCase<"ErrorToken", 200, "error">,
]> {
  let cppNamespace = "::mlir::csl::parser3";
  let genSpecializedAttr = 0;
}

#endif // PARSER3_TOKEN_KIND