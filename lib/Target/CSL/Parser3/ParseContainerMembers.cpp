//===- ParseContainerMembers.cpp - Top-Level Parsing ----------------------===//
//
// Top-level entry point for parsing a CSL file. Iterates over declarations
// (functions, variables, params, layout, comptime blocks) until EOF.
//
// Modelled after Zig's parseContainerMembers / parseRoot.
//
//===----------------------------------------------------------------------===//

#include "Grammar.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::csl::parser3;

#define DEBUG_TYPE "parser3"

using TK = TokenKind;

#define TRACE() Parser::TraceScope _trace(p, __func__)

bool mlir::csl::parser3::looksLikeItemDecl(Parser &p) {
  // export, extern are always item-level.
  if (p.at(TK::KwExport) || p.at(TK::KwExtern))
    return true;

  // inline is item-level UNLESS followed by 'for' (inline for is a statement).
  // It can also precede 'async fn' / 'async task'.
  if (p.at(TK::KwInline) && p.nth(1) != TK::KwFor)
    return true;

  // noinline is always item-level (only applies to functions).
  if (p.at(TK::KwNoinline))
    return true;

  // async is item-level when used as a function modifier (it's a soft keyword,
  // so it can also be used as an identifier). Check that it's followed by
  // fn/task or another modifier (inline/noinline).
  if (p.atIdent("async") &&
      (p.nth(1) == TK::KwFn || p.nth(1) == TK::KwTask ||
       p.nth(1) == TK::KwInline || p.nth(1) == TK::KwNoinline))
    return true;

  // param, layout are always item-level.
  if (p.at(TK::KwParam) || p.at(TK::KwLayout))
    return true;

  // fn/task followed by an identifier is a function/task declaration.
  // fn/task followed by '(' is a function/task type expression.
  if ((p.at(TK::KwFn) || p.at(TK::KwTask)) && p.nth(1) == TK::Identifier)
    return true;

  return false;
}

void mlir::csl::parser3::parseContainerMembers(Parser &p, Region &region) {
  TRACE();
  Block *block = &region.emplaceBlock();
  OpBuilder::InsertionGuard guard(p.getBuilder());
  p.getBuilder().setInsertionPointToEnd(block);

  auto containerLoc = p.startLoc();

  while (!p.eof()) {
    // Skip past modifier keywords to find the actual declaration keyword.
    // This determines whether we parse a function, variable, etc.
    unsigned skip = 0;
    while (p.nth(skip) == TK::KwExport || p.nth(skip) == TK::KwExtern ||
           p.nth(skip) == TK::KwInline || p.nth(skip) == TK::KwNoinline ||
           p.nthIsIdent(skip, "async"))
      ++skip;
    TokenKind declKind = p.nth(skip);

    if (declKind == TK::KwFn || declKind == TK::KwTask) {
      (void)parseFuncDecl(p);
    } else if (declKind == TK::KwVar || declKind == TK::KwConst) {
      (void)parseVarDecl(p, /*isGlobal=*/true);
    } else if (declKind == TK::KwParam) {
      (void)parseParamDecl(p);
    } else if (declKind == TK::KwLayout) {
      (void)parseLayoutDecl(p);
    } else if (p.at(TK::KwComptime)) {
      (void)parseComptimeBlockDecl(p);
    } else if (skip > 0) {
      // Had modifiers but no valid declaration keyword after them.
      p.advanceWithError("expected declaration after modifier");
    } else {
      // Unknown token at top level. Skip it.
      p.advanceWithError("expected declaration");
    }
  }

  // Update the containing op's location to span the full file content.
  if (Operation *parentOp = region.getParentOp())
    parentOp->setLoc(containerLoc.toHere());

  // Warn about unused param bindings
  for (const auto &[name, binding] : p.getParamBindings()) {
    if (!p.isParamUsed(name))
      p.emitWarning(binding.location, "unused param binding '" + name + "'");
  }
}
