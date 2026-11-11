//===- ParseStatements.cpp - Statement Parsing ----------------------------===//
//
// Statement-level parsing: statement lists, if/for/while/asm statements,
// expression-or-assignment statements.
//
// Reference: Zig's expectStatement, expectStatementRecoverable, findNextStmt.
//
//===----------------------------------------------------------------------===//

#include "Grammar.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::csl::parser3;

#define DEBUG_TYPE "parser3"

using TK = TokenKind;

#define TRACE() Parser::TraceScope _trace(p, __func__)

//===----------------------------------------------------------------------===//
// parseBareExprOrAssign
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseBareExprOrAssign(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  // In the Zig-style design, return/break/continue are expressions handled
  // by parsePrimaryExpr, so no special-casing needed here.

  auto lhsResult = parseExpr(p);
  if (failed(lhsResult))
    return failure();

  // Check for assignment: = or compound assignment
  if (p.at(TK::Eq)) {
    p.advance(); // consume '='
    auto rhsResult = parseValueExpr(p);
    if (failed(rhsResult))
      return failure();
    parsed_csl::AssignOp::create(builder, loc.toHere(), *lhsResult, *rhsResult);
    return success();
  }

  // Check for compound assignment: +=, -=, *=, etc.
  auto compoundOp = Parser::getCompoundAssignOpKind(p.nth(0));
  if (succeeded(compoundOp)) {
    // Desugar: x op= y -> x = x op y. The target is read before y is
    // evaluated.
    Value lhsValue = p.read(*lhsResult, loc.toHere());
    p.advance(); // consume the compound assign op
    auto rhsResult = parseValueExpr(p);
    if (failed(rhsResult))
      return failure();

    Value rhs = parsed_csl::BinaryOp::create(
                    builder, loc.toHere(), (*lhsResult).getType(), *compoundOp,
                    lhsValue, *rhsResult,
                    /*inferred_type=*/nullptr)
                    .getResult();
    parsed_csl::AssignOp::create(builder, loc.toHere(), *lhsResult, rhs);
    return success();
  }

  // Plain expression statement (discard result)
  return success();
}

//===----------------------------------------------------------------------===//
// parseExprOrAssignStmt
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseExprOrAssignStmt(Parser &p) {
  if (failed(parseBareExprOrAssign(p)))
    return failure();
  (void)p.expect(TK::Semi);
  return success();
}

//===----------------------------------------------------------------------===//
// parseIfStmt
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseIfStmt(Parser &p) {
  TRACE();
  auto loc = p.startLoc();
  p.advance(); // consume 'if'

  auto &builder = p.getBuilder();

  (void)p.expect(TK::LParen);
  auto condResult = parseValueExpr(p);
  if (failed(condResult))
    return failure();
  (void)p.expect(TK::RParen);

  auto ifOp = parsed_csl::IfOp::create(builder, loc.start(), *condResult);

  // Then body — block or bare expression (Zig-style semicolon disambiguation).
  // After the then-body expression, ';' means "no else" and 'else' means
  // "else follows". Block bodies are self-delimiting so else is just optional.
  bool thenIsBlock = p.at(TK::LBrace);
  {
    OpBuilder::InsertionGuard guard(builder);
    Block &thenBlock = ifOp.getThenRegion().emplaceBlock();
    builder.setInsertionPointToStart(&thenBlock);
    Parser::LocalScopeGuard scope(p.getLocalScope());

    if (thenIsBlock) {
      if (failed(expectStatement(p)))
        return failure();
    } else {
      if (failed(parseBareExprOrAssign(p)))
        return failure();
      if (p.eat(TK::Semi)) {
        ifOp->setLoc(loc.toHere());
        return success();
      }
    }
  }

  if (p.eat(TK::KwElse)) {
    OpBuilder::InsertionGuard guard(builder);
    Block &elseBlock = ifOp.getElseRegion().emplaceBlock();
    builder.setInsertionPointToStart(&elseBlock);
    Parser::LocalScopeGuard scope(p.getLocalScope());

    if (failed(expectStatement(p)))
      return failure();
  } else if (!thenIsBlock) {
    p.emitError("expected ';' or 'else'");
  }

  ifOp->setLoc(loc.toHere());
  return success();
}

//===----------------------------------------------------------------------===//
// parseForStmt
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseForStmt(Parser &p, StringAttr label) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  bool isInline = p.eat(TK::KwInline);
  p.advance(); // consume 'for'

  (void)p.expect(TK::LParen);
  auto iterableResult = parseValueExpr(p);
  if (failed(iterableResult))
    return failure();
  (void)p.expect(TK::RParen);

  // Capture list: |elem| or |elem, index|
  (void)p.expect(TK::Pipe);

  StringAttr elemName;
  std::optional<StringAttr> indexName;

  if (p.at(TK::Identifier)) {
    auto elemTok = p.advance();
    elemName = builder.getStringAttr(p.getText(elemTok));
  } else {
    p.emitError("expected element name in for loop capture");
    return failure();
  }

  if (p.eat(TK::Comma)) {
    if (p.at(TK::Identifier)) {
      auto indexTok = p.advance();
      indexName = builder.getStringAttr(p.getText(indexTok));
    } else {
      p.emitError("expected index name after ','");
      return failure();
    }
  }

  (void)p.expect(TK::Pipe);

  auto forOp = parsed_csl::ForOp::create(
      builder, loc.start(), *iterableResult, elemName,
      indexName.value_or(StringAttr()), label, builder.getBoolAttr(isInline),
      /*inferred_elem_type=*/mlir::TypeAttr(),
      /*inferred_index_type=*/mlir::TypeAttr());

  // Body
  {
    OpBuilder::InsertionGuard guard(builder);
    Block &bodyBlock = forOp.getBody().emplaceBlock();
    auto unresolvedTy = p.getUnresolvedTy();

    BlockArgument elemArg = bodyBlock.addArgument(unresolvedTy, loc.start());
    BlockArgument indexArg;
    if (indexName)
      indexArg = bodyBlock.addArgument(unresolvedTy, loc.start());

    builder.setInsertionPointToStart(&bodyBlock);
    Parser::LocalScopeGuard scope(p.getLocalScope());

    if (failed(p.declareLocalSymbol(elemName.getValue(), elemArg, loc.start())))
      return failure();
    if (indexName) {
      if (failed(p.declareLocalSymbol(indexName->getValue(), indexArg,
                                      loc.start())))
        return failure();
    }

    if (failed(expectStatement(p)))
      return failure();
  }

  forOp->setLoc(loc.toHere());
  return success();
}

//===----------------------------------------------------------------------===//
// parseWhileStmt
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseWhileStmt(Parser &p, StringAttr label) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  p.advance(); // consume 'while'

  auto whileOp = parsed_csl::WhileOp::create(builder, loc.start(), label);

  // Condition region
  {
    OpBuilder::InsertionGuard guard(builder);
    Block &condBlock = whileOp.getCondRegion().emplaceBlock();
    builder.setInsertionPointToStart(&condBlock);

    (void)p.expect(TK::LParen);
    auto condResult = parseValueExpr(p);
    if (failed(condResult))
      return failure();
    (void)p.expect(TK::RParen);

    parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(), *condResult);
  }

  // Optional continuation: : (bare_statement)
  // In CSL: while (cond) : (i = i + 1) body
  // Or:     while (cond) : ({i += 1;}) body
  if (p.at(TK::Colon)) {
    p.advance();
    OpBuilder::InsertionGuard guard(builder);
    Block &contBlock = whileOp.getContinuation().emplaceBlock();
    builder.setInsertionPointToStart(&contBlock);

    (void)p.expect(TK::LParen);
    if (failed(parseBareExprOrAssign(p)))
      return failure();
    (void)p.expect(TK::RParen);
  }

  // Body region
  {
    OpBuilder::InsertionGuard guard(builder);
    Block &bodyBlock = whileOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&bodyBlock);
    Parser::LocalScopeGuard scope(p.getLocalScope());

    if (failed(expectStatement(p)))
      return failure();
  }

  whileOp->setLoc(loc.toHere());
  return success();
}

//===----------------------------------------------------------------------===//
// parseAsmStmt
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseAsmStmt(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  p.advance(); // consume 'asm'

  bool isVolatile = p.eat(TK::KwVolatile);

  (void)p.expect(TK::LParen);

  // Asm instructions (comptime string)
  auto asmStrResult = parseValueExpr(p);
  if (failed(asmStrResult))
    return failure();

  SmallVector<Value> outputConstraints;
  SmallVector<Value> outputs;
  SmallVector<Attribute> outputNames;
  SmallVector<Value> inputConstraints;
  SmallVector<Value> inputs;
  SmallVector<Attribute> inputNames;
  SmallVector<Value> clobbers;

  // Sections are separated by ':'.
  // Grammar: asm [volatile] ( expr [: outputList [: inputList [: clobberList]]]
  // )

  // Helper to parse one [name] constraint (variable) item.
  // Parse constraint as a prefix expression only (no postfix) to prevent
  // the following (variable) from being consumed as a call postfix on the
  // constraint string.
  // Outputs are locations that the asm writes; inputs are read.
  auto parseConstraintItem =
      [&](SmallVectorImpl<Attribute> &names,
          SmallVectorImpl<Value> &constraints, SmallVectorImpl<Value> &operands,
          bool isOutput) -> LogicalResult {
    (void)p.expect(TK::LBracket);
    if (p.at(TK::Identifier)) {
      auto nameTok = p.advance();
      names.push_back(builder.getStringAttr(p.getText(nameTok)));
    }
    (void)p.expect(TK::RBracket);

    // Constraint is always a string literal — parse as atom only (like Zig)
    // to prevent the following (variable) from being consumed as a call.
    auto constraintLoc = p.startLoc();
    auto constraintResult = parsePrimaryTypeExpr(p);
    if (failed(constraintResult))
      return failure();
    constraints.push_back(p.read(*constraintResult, constraintLoc.toHere()));

    (void)p.expect(TK::LParen);
    auto valResult = isOutput ? parseExpr(p) : parseValueExpr(p);
    if (failed(valResult))
      return failure();
    operands.push_back(*valResult);
    (void)p.expect(TK::RParen);
    return success();
  };

  // Parse output list (after first ':')
  if (p.eat(TK::Colon)) {
    while (p.at(TK::LBracket)) {
      if (failed(parseConstraintItem(outputNames, outputConstraints, outputs,
                                     /*isOutput=*/true)))
        return failure();
      if (!p.eat(TK::Comma))
        break;
    }

    // Parse input list (after second ':')
    if (p.eat(TK::Colon)) {
      while (p.at(TK::LBracket)) {
        if (failed(parseConstraintItem(inputNames, inputConstraints, inputs,
                                       /*isOutput=*/false)))
          return failure();
        if (!p.eat(TK::Comma))
          break;
      }

      // Parse clobber list (after third ':')
      if (p.eat(TK::Colon)) {
        while (!p.at(TK::RParen) && !p.eof()) {
          if (!clobbers.empty()) {
            if (!p.eat(TK::Comma))
              break;
          }
          auto clobberResult = parseValueExpr(p);
          if (failed(clobberResult))
            return failure();
          clobbers.push_back(*clobberResult);
        }
      }
    }
  }

  (void)p.expect(TK::RParen);

  parsed_csl::InlineAsmOp::create(
      builder, loc.toHere(), *asmStrResult, outputConstraints, outputs,
      inputConstraints, inputs, clobbers, builder.getArrayAttr(outputNames),
      builder.getArrayAttr(inputNames), isVolatile,
      /*inferred_asm_string=*/nullptr,
      /*inferred_output_constraints=*/nullptr,
      /*inferred_input_constraints=*/nullptr,
      /*inferred_clobbers=*/nullptr);

  (void)p.expect(TK::Semi);
  return success();
}

//===----------------------------------------------------------------------===//
// expectStatement
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::expectStatement(Parser &p) {
  TRACE();
  // Variable declaration (possibly with comptime prefix)
  if (p.at(TK::KwVar) || p.at(TK::KwConst) ||
      (p.at(TK::KwComptime) &&
       (p.nth(1) == TK::KwVar || p.nth(1) == TK::KwConst)))
    return parseVarDecl(p, /*isGlobal=*/false);

  // If statement (IfOp — no result value, body is a statement)
  // Note: if-as-expression uses SelectOp and is handled in parsePrimaryExpr.
  if (p.at(TK::KwIf))
    return parseIfStmt(p);

  // For statement (possibly with inline prefix)
  if (p.at(TK::KwFor) || (p.at(TK::KwInline) && p.nth(1) == TK::KwFor))
    return parseForStmt(p);

  // While statement
  if (p.at(TK::KwWhile))
    return parseWhileStmt(p);

  // Asm statement
  if (p.at(TK::KwAsm))
    return parseAsmStmt(p);

  // Suspend statement: "suspend" followed by ";"
  if (p.atIdent("suspend") && p.nth(1) == TK::Semi) {
    auto loc = p.startLoc();
    p.advance(); // consume 'suspend'
    parsed_csl::SuspendOp::create(p.getBuilder(), loc.toHere());
    (void)p.expect(TK::Semi);
    return success();
  }

  // Comptime block as statement
  if (p.at(TK::KwComptime) && p.nth(1) == TK::LBrace)
    return parseComptimeBlockDecl(p);

  // Labeled constructs: label: { block }, label: [inline] for, label: while
  if (p.at(TK::Identifier) && p.nth(1) == TK::Colon) {
    TokenKind afterLabel = p.nth(2);
    if (afterLabel == TK::KwFor ||
        (afterLabel == TK::KwInline && p.nth(3) == TK::KwFor)) {
      auto labelTok = p.advance();
      p.advance(); // consume ':'
      StringAttr label = p.getBuilder().getStringAttr(p.getText(labelTok));
      return parseForStmt(p, label);
    }
    if (afterLabel == TK::KwWhile) {
      auto labelTok = p.advance();
      p.advance(); // consume ':'
      StringAttr label = p.getBuilder().getStringAttr(p.getText(labelTok));
      return parseWhileStmt(p, label);
    }
    if (afterLabel == TK::LBrace) {
      // Labeled block — parse as expression (no trailing ';')
      auto result = parseExpr(p);
      return success(succeeded(result));
    }
  }

  // Self-delimiting expressions that don't need ';' in statement position:
  // blocks, switch.
  if (p.at(TK::LBrace) || p.at(TK::KwSwitch)) {
    auto result = parseExpr(p);
    return success(succeeded(result));
  }

  // Expression or assignment statement
  if (p.atTs(EXPR_FIRST))
    return parseExprOrAssignStmt(p);

  p.emitError("expected statement");
  return failure();
}

//===----------------------------------------------------------------------===//
// parseStatementList
//===----------------------------------------------------------------------===//

void mlir::csl::parser3::parseStatementList(Parser &p) {
  TRACE();
  while (!p.at(TK::RBrace) && !p.eof()) {
    // Check for tokens that look like a top-level item declaration.
    // This means we've run off the end of the function body (missing '}').
    // Break so the caller emits "expected '}'" and the top-level parser
    // sees the item keyword.
    if (looksLikeItemDecl(p)) {
      LLVM_DEBUG(p.getTracePrinter().startLine()
                 << "stmtList: looks like item declaration, breaking\n");
      break;
    }

    p.advancePush();

    if (p.atTs(STMT_FIRST) || p.at(TK::LBrace)) {
      auto result = expectStatement(p);
      if (failed(result)) {
        // expectStatement failed. If we've landed on a closing brace, EOF, or
        // something that looks like a new top-level declaration, stop.
        if (p.at(TK::RBrace) || p.eof() || looksLikeItemDecl(p)) {
          p.advanceDrop();
          break;
        }
        // Skip the problematic token and continue trying
        p.advanceWithError("skipping unexpected token");
        p.advancePop();
        continue;
      }
    } else if (p.atTs(STMT_RECOVERY)) {
      p.advanceDrop();
      break;
    } else {
      p.errRecover("expected statement", STMT_RECOVERY);
    }

    p.advancePop();
  }
}
