//===- Grammar.h - Internal Grammar Forward Declarations ------------------===//
//
// Forward declarations of grammar production functions shared across the
// split grammar files (ParseExpressions.cpp, ParseStatements.cpp,
// ParseDeclarations.cpp, ParseContainerMembers.cpp).
//
// This is an internal header — not part of the public API.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "csl-mlir/Target/CSL/Parser3/Parser.h"

namespace mlir {
namespace csl {
namespace parser3 {

//===----------------------------------------------------------------------===//
// Expression Parsing (ParseExpressions.cpp)
//===----------------------------------------------------------------------===//

/// Top-level expression entry point.
FailureOr<Value> parseExpr(Parser &p);

/// Parse an expression that is used as a value and read it (Parser::read).
FailureOr<Value> parseValueExpr(Parser &p);

/// Parse a type expression that is used as a value and read it.
FailureOr<Value> parseValueTypeExpr(Parser &p);

/// Precedence-climbing binary operator loop.
FailureOr<Value> parseExprPrecedence(Parser &p, int minPrec);

/// Unary prefix operators: !, -, ~, +, &.
FailureOr<Value> parsePrefixExpr(Parser &p);

/// Control flow expressions (return, break, continue, if-expr, switch-expr,
/// labeled/inline for/while, block) and fall-through to parseCurlySuffixExpr.
FailureOr<Value> parsePrimaryExpr(Parser &p);

/// TypeExpr followed by optional { init } suffix.
FailureOr<Value> parseCurlySuffixExpr(Parser &p);

/// Type prefix operators (*T, [*]T, [N]T) then parseSuffixExpr.
FailureOr<Value> parseTypeExpr(Parser &p);

/// parsePrimaryTypeExpr followed by a loop of suffix ops and calls.
FailureOr<Value> parseSuffixExpr(Parser &p);

/// Single suffix operation: .field, [index], .*, .?
/// Returns the new value, or std::nullopt if no suffix matched.
std::optional<Value> parseSuffixOp(Parser &p, Value lhs);

/// Atoms: literals, identifiers, builtins, fn/task type, struct/enum type,
/// parenthesized expr, anonymous init (.{...}), tensor access, anytype.
FailureOr<Value> parsePrimaryTypeExpr(Parser &p);

/// Parse a bare return: return [expr]. Does NOT consume trailing semicolon.
LogicalResult parseBareReturn(Parser &p);

/// Parse a bare break: break [:label [value]]. No trailing semicolon.
LogicalResult parseBareBreak(Parser &p);

/// Parse a bare continue: continue [:label]. No trailing semicolon.
LogicalResult parseBareContinue(Parser &p);

//===----------------------------------------------------------------------===//
// Statement Parsing (ParseStatements.cpp)
//===----------------------------------------------------------------------===//

/// Parse a list of statements until '}' or EOF.
void parseStatementList(Parser &p);

/// Parse a single statement.
LogicalResult expectStatement(Parser &p);

/// Parse an expression or assignment without trailing semicolon.
LogicalResult parseBareExprOrAssign(Parser &p);

/// Parse an expression statement or assignment: expr; or lhs = rhs;
LogicalResult parseExprOrAssignStmt(Parser &p);

/// Parse an if statement: if (cond) thenBody [else elseBody]
LogicalResult parseIfStmt(Parser &p);

/// Parse a for statement: [label:] [inline] for (...) |elem| body
LogicalResult parseForStmt(Parser &p, StringAttr label = {});

/// Parse a while statement: [label:] while (cond) [: (cont)] body
LogicalResult parseWhileStmt(Parser &p, StringAttr label = {});

/// Parse an asm statement.
LogicalResult parseAsmStmt(Parser &p);

//===----------------------------------------------------------------------===//
// Declaration Parsing (ParseDeclarations.cpp)
//===----------------------------------------------------------------------===//

/// Parse a function or task declaration.
LogicalResult parseFuncDecl(Parser &p);

/// Parse a variable/const declaration.
LogicalResult parseVarDecl(Parser &p, bool isGlobal);

/// Parse a param declaration: param name : type = default;
LogicalResult parseParamDecl(Parser &p);

/// Parse a layout declaration: layout { ... }
LogicalResult parseLayoutDecl(Parser &p);

/// Parse a comptime block: comptime { ... }
LogicalResult parseComptimeBlockDecl(Parser &p);

//===----------------------------------------------------------------------===//
// Top-Level Parsing (ParseContainerMembers.cpp)
//===----------------------------------------------------------------------===//

/// Check whether the current token sequence looks like a top-level item
/// declaration rather than an expression statement.
bool looksLikeItemDecl(Parser &p);

/// Parse the entire file into the given region (renamed from parseFile).
void parseContainerMembers(Parser &p, Region &region);

} // namespace parser3
} // namespace csl
} // namespace mlir
