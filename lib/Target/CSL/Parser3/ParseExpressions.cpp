//===- ParseExpressions.cpp - Expression Parsing --------------------------===//
//
// Expression parsing with Zig-style call hierarchy:
//
//   parseExpr
//     └─ parseExprPrecedence(minPrec=0)    // Precedence-climbing binary loop
//          ├─ parsePrefixExpr()             // Unary prefix: !, -, ~, +, &
//          │    └─ parsePrimaryExpr()       // Control flow + curly suffix
//          │         ├─ return/break/continue (as expressions)
//          │         ├─ if-expr, switch-expr, block, labeled, comptime
//          │         └─ parseCurlySuffixExpr()
//          │              ├─ parseTypeExpr()     // Type prefix + suffix
//          │              │    ├─ *T, [*]T, [N]T
//          │              │    └─ parseSuffixExpr()
//          │              │         ├─ parsePrimaryTypeExpr()  // Atoms
//          │              │         └─ loop: parseSuffixOp / call
//          │              └─ optional { init } suffix
//          └─ binary op loop
//
// Reference: Zig's Parse.zig expression chain.
//
//===----------------------------------------------------------------------===//

#include "Grammar.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinTypes.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::csl::parser3;

#define DEBUG_TYPE "parser3"

using TK = TokenKind;

#define TRACE() Parser::TraceScope _trace(p, __func__)

//===----------------------------------------------------------------------===//
// Helpers (file-local)
//===----------------------------------------------------------------------===//

/// Parse an integer literal text into an int64_t value.
static int64_t parseIntegerText(llvm::StringRef text) {
  // Remove underscores
  std::string cleaned;
  cleaned.reserve(text.size());
  for (char c : text) {
    if (c != '_')
      cleaned += c;
  }
  llvm::StringRef cleanedRef(cleaned);

  if (cleanedRef.starts_with("0x") || cleanedRef.starts_with("0X"))
    return std::stoll(cleaned.substr(2), nullptr, 16);
  if (cleanedRef.starts_with("0b") || cleanedRef.starts_with("0B"))
    return std::stoll(cleaned.substr(2), nullptr, 2);
  if (cleanedRef.starts_with("0o") || cleanedRef.starts_with("0O"))
    return std::stoll(cleaned.substr(2), nullptr, 8);
  return std::stoll(cleaned);
}

/// Parse a float literal text into a double value.
static double parseFloatText(llvm::StringRef text) {
  std::string cleaned;
  cleaned.reserve(text.size());
  for (char c : text) {
    if (c != '_')
      cleaned += c;
  }
  return std::stod(cleaned);
}

/// Unescape a string literal (removing surrounding quotes and handling
/// escape sequences).
static std::string unescapeString(llvm::StringRef text) {
  // Remove surrounding quotes
  assert(text.size() >= 2 && text.front() == '"' && text.back() == '"');
  text = text.drop_front().drop_back();

  std::string result;
  result.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\\' && i + 1 < text.size()) {
      char next = text[i + 1];
      switch (next) {
      case 'n':
        result += '\n';
        ++i;
        break;
      case 'r':
        result += '\r';
        ++i;
        break;
      case 't':
        result += '\t';
        ++i;
        break;
      case '\\':
        result += '\\';
        ++i;
        break;
      case '"':
        result += '"';
        ++i;
        break;
      case 'x': {
        // \xHH
        if (i + 3 < text.size()) {
          unsigned hexVal;
          llvm::StringRef hexStr(text.data() + i + 2, 2);
          (void)hexStr.getAsInteger(16, hexVal);
          result += static_cast<char>(hexVal);
          i += 3;
        }
        break;
      }
      case 'u': {
        // \u{H+}
        if (i + 3 < text.size() && text[i + 2] == '{') {
          size_t closeBrace = text.find('}', i + 3);
          if (closeBrace != llvm::StringRef::npos) {
            llvm::StringRef hexStr(text.data() + i + 3, closeBrace - (i + 3));
            unsigned codepoint;
            (void)hexStr.getAsInteger(16, codepoint);
            // Encode as UTF-8
            if (codepoint < 0x80) {
              result += static_cast<char>(codepoint);
            } else if (codepoint < 0x800) {
              result += static_cast<char>(0xC0 | (codepoint >> 6));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            } else if (codepoint < 0x10000) {
              result += static_cast<char>(0xE0 | (codepoint >> 12));
              result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            } else {
              result += static_cast<char>(0xF0 | (codepoint >> 18));
              result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
              result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            i = closeBrace;
          }
        }
        break;
      }
      default:
        result += next;
        ++i;
        break;
      }
    } else {
      result += text[i];
    }
  }
  return result;
}

/// Parse a character literal into an integer value.
static uint64_t parseCharLiteral(llvm::StringRef text) {
  assert(text.size() >= 3 && text.front() == '\'' && text.back() == '\'');
  llvm::StringRef inner = text.drop_front().drop_back();

  if (inner[0] != '\\') {
    assert(inner.size() == 1);
    return static_cast<unsigned char>(inner[0]);
  }

  assert(inner.size() >= 2);
  switch (inner[1]) {
  case 'n':
    return '\n';
  case 'r':
    return '\r';
  case 't':
    return '\t';
  case '\\':
    return '\\';
  case '\'':
    return '\'';
  case '"':
    return '"';
  case 'x': {
    assert(inner.size() == 4);
    unsigned hexVal;
    (void)inner.drop_front(2).getAsInteger(16, hexVal);
    return hexVal;
  }
  case 'u': {
    assert(inner.size() >= 4 && inner[2] == '{' && inner.back() == '}');
    llvm::StringRef hexStr = inner.drop_front(3).drop_back(1);
    unsigned codepoint;
    (void)hexStr.getAsInteger(16, codepoint);
    return codepoint;
  }
  default:
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// Operator Precedence Table (Zig-style)
//===----------------------------------------------------------------------===//

namespace {

enum class Assoc { Left, None };

struct OperInfo {
  int prec;             // -1 = not an operator
  csl::BinaryOpKind op; // only meaningful when prec >= 0
  Assoc assoc;
};

/// Map a token to its operator info. Returns prec=-1 for non-operators.
OperInfo getOperInfo(TokenKind kind) {
  switch (kind) {
  // Precedence 10: logical or
  case TK::KwOr:
    return {10, csl::BinaryOpKind::LogicalOr, Assoc::Left};

  // Precedence 20: logical and
  case TK::KwAnd:
    return {20, csl::BinaryOpKind::LogicalAnd, Assoc::Left};

  // Precedence 30: comparisons (non-associative)
  case TK::EqualEqual:
    return {30, csl::BinaryOpKind::Eq, Assoc::None};
  case TK::NotEqual:
    return {30, csl::BinaryOpKind::Ne, Assoc::None};
  case TK::Less:
    return {30, csl::BinaryOpKind::Lt, Assoc::None};
  case TK::LessEqual:
    return {30, csl::BinaryOpKind::Le, Assoc::None};
  case TK::Greater:
    return {30, csl::BinaryOpKind::Gt, Assoc::None};
  case TK::GreaterEqual:
    return {30, csl::BinaryOpKind::Ge, Assoc::None};

  // Precedence 40: bitwise (and, xor, or)
  case TK::Ampersand:
    return {40, csl::BinaryOpKind::BitAnd, Assoc::Left};
  case TK::Caret:
    return {40, csl::BinaryOpKind::BitXor, Assoc::Left};
  case TK::Pipe:
    return {40, csl::BinaryOpKind::BitOr, Assoc::Left};

  // Precedence 50: shifts
  case TK::LeftShift:
    return {50, csl::BinaryOpKind::Shl, Assoc::Left};
  case TK::RightShift:
    return {50, csl::BinaryOpKind::Shr, Assoc::Left};

  // Precedence 60: addition/subtraction
  case TK::Plus:
    return {60, csl::BinaryOpKind::Add, Assoc::Left};
  case TK::Minus:
    return {60, csl::BinaryOpKind::Sub, Assoc::Left};

  // Precedence 70: multiplication/division
  case TK::Star:
    return {70, csl::BinaryOpKind::Mul, Assoc::Left};
  case TK::Slash:
    return {70, csl::BinaryOpKind::Div, Assoc::Left};
  case TK::Percent:
    return {70, csl::BinaryOpKind::Mod, Assoc::Left};

  default:
    return {-1, {}, Assoc::Left};
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// parseBareReturn / parseBareBreak / parseBareContinue
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseBareReturn(Parser &p) {
  auto loc = p.startLoc();
  p.advance(); // consume 'return'
  auto &builder = p.getBuilder();

  if (p.atTs(EXPR_FIRST)) {
    auto exprResult = parseValueExpr(p);
    if (failed(exprResult))
      return failure();
    parsed_csl::ReturnOp::create(builder, loc.toHere(), *exprResult);
  } else {
    parsed_csl::ReturnOp::create(builder, loc.toHere());
  }
  return success();
}

LogicalResult mlir::csl::parser3::parseBareBreak(Parser &p) {
  auto loc = p.startLoc();
  p.advance(); // consume 'break'
  auto &builder = p.getBuilder();

  StringAttr targetLabel;
  Value breakValue;

  // Optional label: :label
  if (p.at(TK::Colon) && p.nth(1) == TK::Identifier) {
    p.advance(); // consume ':'
    auto labelTok = p.advance();
    targetLabel = builder.getStringAttr(p.getText(labelTok));
  }

  // Optional value (only with label)
  if (targetLabel && p.atTs(EXPR_FIRST)) {
    auto valueResult = parseValueExpr(p);
    if (failed(valueResult))
      return failure();
    breakValue = *valueResult;
  }

  parsed_csl::BreakOp::create(builder, loc.toHere(), targetLabel, breakValue);
  return success();
}

LogicalResult mlir::csl::parser3::parseBareContinue(Parser &p) {
  auto loc = p.startLoc();
  p.advance(); // consume 'continue'
  auto &builder = p.getBuilder();

  StringAttr targetLabel;

  // Optional label: :label
  if (p.at(TK::Colon) && p.nth(1) == TK::Identifier) {
    p.advance(); // consume ':'
    auto labelTok = p.advance();
    targetLabel = builder.getStringAttr(p.getText(labelTok));
  }

  parsed_csl::ContinueOp::create(builder, loc.toHere(), targetLabel);
  return success();
}

//===----------------------------------------------------------------------===//
// parseExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parseExpr(Parser &p) {
  TRACE();
  return parseExprPrecedence(p, /*minPrec=*/0);
}

FailureOr<Value> mlir::csl::parser3::parseValueExpr(Parser &p) {
  auto loc = p.startLoc();
  FailureOr<Value> expr = parseExpr(p);
  if (failed(expr))
    return failure();
  return p.read(*expr, loc.toHere());
}

FailureOr<Value> mlir::csl::parser3::parseValueTypeExpr(Parser &p) {
  auto loc = p.startLoc();
  FailureOr<Value> expr = parseTypeExpr(p);
  if (failed(expr))
    return failure();
  return p.read(*expr, loc.toHere());
}

//===----------------------------------------------------------------------===//
// parseExprPrecedence — Precedence-climbing binary operator loop
// Reference: Zig's parseExprPrecedence
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parseExprPrecedence(Parser &p,
                                                         int minPrec) {
  TRACE();
  auto lhsLoc = p.startLoc();
  auto lhs = parsePrefixExpr(p);
  if (failed(lhs))
    return failure();

  auto &builder = p.getBuilder();

  // Track the last non-associative precedence to detect chained comparisons.
  int bannedPrec = -1;

  while (true) {
    TokenKind tokKind = p.nth(0);
    OperInfo info = getOperInfo(tokKind);
    if (info.prec < minPrec)
      break;

    if (info.prec == bannedPrec) {
      p.emitError("comparison operators cannot be chained");
      return failure();
    }

    // The left operand is read before the right one is evaluated.
    *lhs = p.read(*lhs, lhsLoc.toHere());

    p.advancePush();
    Token opTok = p.advance(); // consume operator
    Location loc = p.getLoc(opTok);

    // Handle short-circuit operators (and, or)
    if (tokKind == TK::KwAnd || tokKind == TK::KwOr) {
      auto kind = (tokKind == TK::KwAnd) ? csl::ShortCircuitKind::And
                                         : csl::ShortCircuitKind::Or;
      auto shortCircuitOp = parsed_csl::ShortCircuitOp::create(
          builder, loc, kind, *lhs, /*inferred_type=*/nullptr);

      // Parse RHS lazily inside region
      {
        Block &rhsBlock = shortCircuitOp.getRhsRegion().emplaceBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&rhsBlock);

        auto rhsLoc = p.startLoc();
        auto rhs = parseExprPrecedence(p, info.prec + 1);
        if (failed(rhs)) {
          p.advancePop();
          break;
        }
        parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                    p.read(*rhs, rhsLoc.toHere()));
      }

      *lhs = shortCircuitOp.getResult();
      if (info.assoc == Assoc::None)
        bannedPrec = info.prec;
      p.advancePop();
      continue;
    }

    // Regular binary operator — recurse with higher min precedence
    auto rhsLoc = p.startLoc();
    auto rhs = parseExprPrecedence(p, info.prec + 1);
    if (failed(rhs)) {
      p.advancePop();
      break;
    }

    *lhs = parsed_csl::BinaryOp::create(builder, loc, (*lhs).getType(), info.op,
                                        *lhs, p.read(*rhs, rhsLoc.toHere()),
                                        /*inferred_type=*/nullptr)
               .getResult();

    if (info.assoc == Assoc::None)
      bannedPrec = info.prec;

    p.advancePop();
  }

  return lhs;
}

//===----------------------------------------------------------------------===//
// parsePrefixExpr — Unary prefix operators
// Reference: Zig's parsePrefixExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parsePrefixExpr(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto unresolvedTy = p.getUnresolvedTy();

  // Address-of: &expr
  if (p.at(TK::Ampersand)) {
    auto loc = p.startLoc();
    p.advance();
    auto inner = parsePrefixExpr(p);
    if (failed(inner))
      return failure();
    return parsed_csl::AddressOfOp::create(builder, loc.toHere(), unresolvedTy,
                                           *inner,
                                           /*inferred_type=*/nullptr)
        .getResult();
  }

  // Unary operators: !, ~, -, +
  if (p.at(TK::Exclamation) || p.at(TK::Tilde) || p.at(TK::Minus) ||
      p.at(TK::Plus)) {
    auto loc = p.startLoc();
    auto opTok = p.advance();
    auto innerLoc = p.startLoc();
    auto inner = parsePrefixExpr(p);
    if (failed(inner))
      return failure();

    if (opTok.kind == TK::Plus) {
      // Unary plus is a no-op
      return *inner;
    }

    auto opcode = llvm::StringSwitch<std::optional<csl::UnaryOpKind>>(
                      llvm::StringRef(p.getText(opTok)))
                      .Case("!", csl::UnaryOpKind::LogicalNot)
                      .Case("~", csl::UnaryOpKind::BitNot)
                      .Case("-", csl::UnaryOpKind::Neg)
                      .Default(std::nullopt);

    if (!opcode) {
      p.emitError(loc.start(), "unknown unary operator");
      return failure();
    }

    return parsed_csl::UnaryOp::create(builder, loc.toHere(),
                                       (*inner).getType(), *opcode,
                                       p.read(*inner, innerLoc.toHere()),
                                       /*inferred_type=*/nullptr)
        .getResult();
  }

  // Not a prefix operator — delegate to parsePrimaryExpr
  return parsePrimaryExpr(p);
}

//===----------------------------------------------------------------------===//
// parsePrimaryExpr — Control flow expressions + fall-through to curly suffix
// Reference: Zig's parsePrimaryExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parsePrimaryExpr(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();

  // return [expr]
  if (p.at(TK::KwReturn)) {
    if (failed(parseBareReturn(p)))
      return failure();
    // Return is a terminator that produces void for expression purposes
    return parsed_csl::ConstantOp::create(builder, p.getLocPrev(),
                                          csl::VoidAttr::get(&p.getContext()))
        .getResult();
  }

  // break [:label [value]]
  if (p.at(TK::KwBreak)) {
    if (failed(parseBareBreak(p)))
      return failure();
    return parsed_csl::ConstantOp::create(builder, p.getLocPrev(),
                                          csl::VoidAttr::get(&p.getContext()))
        .getResult();
  }

  // continue [:label]
  if (p.at(TK::KwContinue)) {
    if (failed(parseBareContinue(p)))
      return failure();
    return parsed_csl::ConstantOp::create(builder, p.getLocPrev(),
                                          csl::VoidAttr::get(&p.getContext()))
        .getResult();
  }

  // If expression: if (cond) then_expr else else_expr → SelectOp
  if (p.at(TK::KwIf)) {
    auto loc = p.startLoc();
    p.advance();

    (void)p.expect(TK::LParen);
    auto condResult = parseValueExpr(p);
    if (failed(condResult))
      return failure();
    (void)p.expect(TK::RParen);

    auto selectOp =
        parsed_csl::SelectOp::create(builder, loc.start(), *condResult,
                                     /*inferred_type=*/nullptr);

    // Then branch
    {
      Block &thenBlock = selectOp.getThenRegion().emplaceBlock();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&thenBlock);

      auto thenResult = parseValueExpr(p);
      if (failed(thenResult))
        return failure();
      parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                  *thenResult);
    }

    // Optional else branch
    if (p.eat(TK::KwElse)) {
      Block &elseBlock = selectOp.getElseRegion().emplaceBlock();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&elseBlock);

      auto elseResult = parseValueExpr(p);
      if (failed(elseResult))
        return failure();
      parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                  *elseResult);
    }

    selectOp->setLoc(loc.toHere());
    return selectOp.getResult();
  }

  // Switch expression: switch (input) { cases... }
  if (p.at(TK::KwSwitch)) {
    auto loc = p.startLoc();
    p.advance();

    (void)p.expect(TK::LParen);
    auto inputResult = parseValueExpr(p);
    if (failed(inputResult))
      return failure();
    (void)p.expect(TK::RParen);

    (void)p.expect(TK::LBrace);

    // Parse case bodies into temporary blocks in the current region. Case
    // value ops are emitted at the current insertion point (before the
    // SwitchOp we'll create after the loop). Body ops go into temporary
    // blocks that are spliced into the SwitchOp's regions after creation.
    // We use the current region (not detached blocks) so that getParentOp()
    // works during parsing.
    Region *currentRegion = builder.getInsertionBlock()->getParent();
    SmallVector<Value> flatCaseOperands;
    SmallVector<int32_t> caseOperandSegments;
    SmallVector<Block *> caseBlocks;
    Block *elseBlock = nullptr;

    while (!p.at(TK::RBrace) && !p.eof()) {
      if (p.at(TK::KwElse)) {
        p.advance();
        (void)p.expect(TK::FatArrow);

        elseBlock = new Block();
        currentRegion->push_back(elseBlock);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(elseBlock);
          Parser::LocalScopeGuard scope(p.getLocalScope());

          auto bodyResult = parseValueExpr(p);
          if (succeeded(bodyResult))
            parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                        *bodyResult);
        }

        (void)p.eat(TK::Comma);
        continue;
      }

      // Parse case value list (emitted at current insertion point)
      SmallVector<Value> caseValues;
      while (true) {
        auto valueResult = parseValueExpr(p);
        if (failed(valueResult))
          break;
        caseValues.push_back(*valueResult);
        flatCaseOperands.push_back(*valueResult);
        if (!p.eat(TK::Comma) || p.at(TK::FatArrow))
          break;
      }
      caseOperandSegments.push_back(caseValues.size());

      (void)p.expect(TK::FatArrow);

      // Parse case body into a temporary block in the current region
      auto *caseBlock = new Block();
      currentRegion->push_back(caseBlock);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(caseBlock);
        Parser::LocalScopeGuard scope(p.getLocalScope());

        auto bodyResult = parseValueExpr(p);
        if (succeeded(bodyResult))
          parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                      *bodyResult);
      }
      caseBlocks.push_back(caseBlock);

      (void)p.eat(TK::Comma);
    }

    (void)p.expect(TK::RBrace);

    // Create SwitchOp now that we know the exact number of cases.
    auto switchOp = parsed_csl::SwitchOp::create(
        builder, loc.toHere(), *inputResult, flatCaseOperands,
        caseOperandSegments, static_cast<unsigned>(caseBlocks.size()));

    // Move temporary blocks from the current region into the SwitchOp.
    for (unsigned i = 0; i < caseBlocks.size(); ++i)
      switchOp.getCaseRegions()[i].getBlocks().splice(
          switchOp.getCaseRegions()[i].end(), currentRegion->getBlocks(),
          caseBlocks[i]->getIterator());
    if (elseBlock)
      switchOp.getElseRegion().getBlocks().splice(
          switchOp.getElseRegion().end(), currentRegion->getBlocks(),
          elseBlock->getIterator());

    return switchOp.getResult();
  }

  // Labeled block/for/while: identifier ':' ...
  if (p.at(TK::Identifier) && p.nth(1) == TK::Colon) {
    TokenKind afterLabel = p.nth(2);

    // Labeled block: label: { ... }
    if (afterLabel == TK::LBrace) {
      auto loc = p.startLoc();
      auto labelTok = p.advance(); // consume identifier (label)
      p.advance();                 // consume ':'
      p.advance();                 // consume '{'

      StringAttr label = builder.getStringAttr(p.getText(labelTok));
      auto labeledBlockOp = parsed_csl::LabeledBlockOp::create(
          builder, loc.start(), label, /*inferred_type=*/nullptr);

      {
        Block &bodyBlock = labeledBlockOp.getBody().emplaceBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&bodyBlock);
        Parser::LocalScopeGuard scope(p.getLocalScope());
        parseStatementList(p);
      }
      (void)p.expect(TK::RBrace);
      labeledBlockOp->setLoc(loc.toHere());
      return labeledBlockOp.getResult();
    }

    // Labeled for/while are handled at the statement level only
    // (parsePrimaryExpr falls through to parseCurlySuffixExpr for other cases)
  }

  // Unlabeled block expression: { stmts... }
  // Always yields void.
  if (p.at(TK::LBrace)) {
    auto loc = p.startLoc();
    p.advance(); // consume '{'

    Parser::LocalScopeGuard scope(p.getLocalScope());
    parseStatementList(p);
    (void)p.expect(TK::RBrace);

    return parsed_csl::ConstantOp::create(builder, loc.toHere(),
                                          csl::VoidAttr::get(&p.getContext()))
        .getResult();
  }

  // Comptime expression: comptime expr
  if (p.at(TK::KwComptime)) {
    auto loc = p.startLoc();
    p.advance();

    // comptime { ... } as expression — treat as block
    if (p.at(TK::LBrace)) {
      p.advance(); // consume '{'
      Parser::LocalScopeGuard scope(p.getLocalScope());
      parseStatementList(p);
      (void)p.expect(TK::RBrace);
      return parsed_csl::ConstantOp::create(builder, loc.toHere(),
                                            csl::VoidAttr::get(&p.getContext()))
          .getResult();
    }

    auto comptimeExprOp = parsed_csl::ComptimeExprOp::create(
        builder, loc.start(), /*inferred_type=*/nullptr);
    Block &block = comptimeExprOp.getExprRegion().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    auto exprResult = parseValueExpr(p);
    if (failed(exprResult))
      return failure();
    parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(), *exprResult);

    comptimeExprOp->setLoc(loc.toHere());
    return comptimeExprOp.getResult();
  }

  // Fall through to parseCurlySuffixExpr for everything else
  return parseCurlySuffixExpr(p);
}

//===----------------------------------------------------------------------===//
// parseCurlySuffixExpr — TypeExpr followed by optional { init } suffix
// Reference: Zig's parseCurlySuffixExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parseCurlySuffixExpr(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto unresolvedTy = p.getUnresolvedTy();

  auto lhsLoc = p.startLoc();
  auto lhs = parseTypeExpr(p);
  if (failed(lhs))
    return failure();

  // Check for init suffix: TypeExpr { .field = val, ... } or { val, ... }
  if (!p.at(TK::LBrace))
    return lhs;

  // The type is read before the initializers are evaluated.
  *lhs = p.read(*lhs, lhsLoc.toHere());

  // Struct init: { .field = val, ... }. An element starting with `.{` is an
  // anonymous literal in an array init instead.
  if (p.nth(1) == TK::Dot && p.nth(2) == TK::Identifier) {
    auto loc = p.startLoc(*lhs);
    p.advance(); // consume '{'

    SmallVector<std::string> fieldNameStorage;
    SmallVector<Value> fieldValues;

    while (!p.at(TK::RBrace) && !p.eof()) {
      if (!fieldNameStorage.empty()) {
        if (!p.eat(TK::Comma))
          break;
        if (p.at(TK::RBrace))
          break;
      }
      (void)p.expect(TK::Dot);
      if (!p.at(TK::Identifier)) {
        p.emitError("expected field name");
        break;
      }
      auto nameTok = p.advance();
      fieldNameStorage.push_back(p.getText(nameTok).str());
      (void)p.expect(TK::Eq);
      auto valueResult = parseValueExpr(p);
      if (failed(valueResult))
        return failure();
      fieldValues.push_back(*valueResult);
    }
    (void)p.expect(TK::RBrace);

    SmallVector<StringRef> fieldNameRefs;
    for (const auto &name : fieldNameStorage)
      fieldNameRefs.push_back(name);
    auto fieldNamesAttr = builder.getStrArrayAttr(fieldNameRefs);

    return parsed_csl::ConstructStructOp::create(
               builder, loc.toHere(), unresolvedTy, fieldNamesAttr, *lhs,
               fieldValues, /*inferred_type=*/nullptr)
        .getResult();
  }

  // Array/tuple init: TypeExpr { val, val, ... }
  {
    auto loc = p.startLoc(*lhs);
    p.advance(); // consume '{'
    SmallVector<Value> elements;
    while (!p.at(TK::RBrace) && !p.eof()) {
      if (!elements.empty()) {
        if (!p.eat(TK::Comma))
          break;
        if (p.at(TK::RBrace))
          break;
      }
      auto elemResult = parseValueExpr(p);
      if (failed(elemResult))
        return failure();
      elements.push_back(*elemResult);
    }
    (void)p.expect(TK::RBrace);

    return parsed_csl::ConstructArrayOp::create(builder, loc.toHere(),
                                                unresolvedTy, *lhs, elements,
                                                /*inferred_type=*/nullptr)
        .getResult();
  }
}

//===----------------------------------------------------------------------===//
// parseTypeExpr — Type prefix operators then parseSuffixExpr
// Reference: Zig's parseTypeExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parseTypeExpr(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();

  // Many-item pointer type: [*]T or [*]const T
  if (p.at(TK::LBracket) && p.nth(1) == TK::Star) {
    auto loc = p.startLoc();
    p.advance(); // consume '['
    p.advance(); // consume '*'
    (void)p.expect(TK::RBracket);
    bool isConst = p.eat(TK::KwConst);
    auto pointee = parseValueTypeExpr(p);
    if (failed(pointee))
      return failure();
    return parsed_csl::PtrTypeOp::create(builder, loc.toHere(), *pointee,
                                         isConst,
                                         /*is_many_ptr=*/true)
        .getResult();
  }

  // Single-item pointer: *T or *const T
  if (p.at(TK::Star)) {
    auto loc = p.startLoc();
    p.advance();
    bool isConst = p.eat(TK::KwConst);
    auto pointee = parseValueTypeExpr(p);
    if (failed(pointee))
      return failure();
    return parsed_csl::PtrTypeOp::create(builder, loc.toHere(), *pointee,
                                         isConst,
                                         /*is_many_ptr=*/false)
        .getResult();
  }

  // Array type: [N]T (N can be an expression, possibly multi-dimensional)
  if (p.at(TK::LBracket)) {
    auto loc = p.startLoc();
    p.advance(); // consume '['

    SmallVector<Value> dims;
    while (!p.at(TK::RBracket) && !p.eof()) {
      if (!dims.empty()) {
        if (!p.eat(TK::Comma))
          break;
      }
      auto dimExpr = parseValueExpr(p);
      if (failed(dimExpr))
        return failure();
      dims.push_back(*dimExpr);
    }
    (void)p.expect(TK::RBracket);

    // Element type follows
    auto elemType = parseValueTypeExpr(p);
    if (failed(elemType))
      return failure();

    return parsed_csl::ArrayTypeOp::create(builder, loc.toHere(), *elemType,
                                           dims)
        .getResult();
  }

  // Optional type: ?T
  if (p.at(TK::Question)) {
    p.emitError("optional types not yet implemented");
    return failure();
  }

  // Not a type prefix — delegate to parseSuffixExpr
  return parseSuffixExpr(p);
}

//===----------------------------------------------------------------------===//
// parseSuffixExpr — parsePrimaryTypeExpr + suffix/call loop
// Reference: Zig's parseSuffixExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parseSuffixExpr(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto unresolvedTy = p.getUnresolvedTy();

  auto resLoc = p.startLoc();
  auto res = parsePrimaryTypeExpr(p);
  if (failed(res))
    return failure();

  while (true) {
    // Try suffix op: .field, [index], .*, .?
    auto suffixResult = parseSuffixOp(p, *res);
    if (suffixResult) {
      *res = *suffixResult;
      continue;
    }

    // Function call: expr(args...)
    if (p.at(TK::LParen)) {
      auto loc = p.startLoc(*res);
      // The callee is read before the arguments are evaluated.
      *res = p.read(*res, resLoc.toHere());
      p.advance(); // consume '('
      SmallVector<Value> args;
      while (!p.at(TK::RParen) && !p.eof()) {
        if (looksLikeItemDecl(p))
          break;
        if (!args.empty()) {
          if (!p.eat(TK::Comma))
            break;
        }
        auto argResult = parseValueExpr(p);
        if (failed(argResult))
          break;
        args.push_back(*argResult);
      }
      (void)p.expect(TK::RParen);

      *res = parsed_csl::CallOp::create(builder, loc.toHere(), unresolvedTy,
                                        *res, args, /*inferred_type=*/nullptr)
                 .getResult();
      continue;
    }

    break; // no more suffix operators or calls
  }

  return res;
}

//===----------------------------------------------------------------------===//
// parseSuffixOp — Single suffix operation
// Reference: Zig's parseSuffixOp
//===----------------------------------------------------------------------===//

std::optional<Value> mlir::csl::parser3::parseSuffixOp(Parser &p, Value lhs) {
  auto &builder = p.getBuilder();
  auto unresolvedTy = p.getUnresolvedTy();

  // Field access: expr.field
  if (p.at(TK::Dot) && p.nth(1) == TK::Identifier) {
    auto loc = p.startLoc(lhs);
    p.advance();                 // consume '.'
    auto fieldTok = p.advance(); // consume identifier
    llvm::StringRef fieldName = p.getText(fieldTok);

    return parsed_csl::FieldAccessOp::create(
               builder, loc.toHere(), lhs.getType(), lhs, fieldName,
               /*inferred_type=*/nullptr, /*inferred_base_type=*/nullptr)
        .getResult();
  }

  // Index access: expr[indices]
  if (p.at(TK::LBracket)) {
    auto loc = p.startLoc(lhs);
    p.advance(); // consume '['
    SmallVector<Value> indices;
    while (!p.at(TK::RBracket) && !p.eof()) {
      if (looksLikeItemDecl(p))
        break;
      if (!indices.empty()) {
        if (!p.eat(TK::Comma))
          break;
      }
      auto indexResult = parseValueExpr(p);
      if (failed(indexResult))
        break;
      indices.push_back(*indexResult);
    }
    (void)p.expect(TK::RBracket);

    return parsed_csl::IndexAccessOp::create(builder, loc.toHere(),
                                             lhs.getType(), lhs, indices,
                                             /*inferred_type=*/nullptr,
                                             /*constant_indices=*/nullptr)
        .getResult();
  }

  // Dereference: expr.*
  if (p.at(TK::DotStar)) {
    auto loc = p.startLoc(lhs);
    Value pointer = p.read(lhs, loc.toHere());
    p.advance();
    return parsed_csl::DereferenceOp::create(builder, loc.toHere(),
                                             unresolvedTy, pointer,
                                             /*inferred_type=*/nullptr)
        .getResult();
  }

  // Optional unwrap: expr.?
  if (p.at(TK::DotQuestion)) {
    p.emitError("optional unwrap not yet implemented");
    return std::nullopt;
  }

  return std::nullopt; // no suffix matched
}

//===----------------------------------------------------------------------===//
// parsePrimaryTypeExpr — Atoms
// Reference: Zig's parsePrimaryTypeExpr
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::csl::parser3::parsePrimaryTypeExpr(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto unresolvedTy = p.getUnresolvedTy();

  // Integer literal
  if (p.at(TK::IntegerLiteral)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    int64_t value = parseIntegerText(p.getText(tok));
    APInt apValue(64, value, /*isSigned=*/true);
    auto attr = parsed_csl::ComptimeIntAttr::get(&p.getContext(), apValue);
    return parsed_csl::ConstantOp::create(builder, loc, attr).getResult();
  }

  // Float literal
  if (p.at(TK::FloatingLiteral)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    double value = parseFloatText(p.getText(tok));
    APFloat apValue(value);
    auto attr = parsed_csl::ComptimeFloatAttr::get(&p.getContext(), apValue);
    return parsed_csl::ConstantOp::create(builder, loc, attr).getResult();
  }

  // Bool literal
  if (p.at(TK::KwTrue) || p.at(TK::KwFalse)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    bool value = (tok.kind == TK::KwTrue);
    auto boolType = IntegerType::get(&p.getContext(), 1);
    auto attr = IntegerAttr::get(boolType, APInt(1, value));
    return parsed_csl::ConstantOp::create(builder, loc, attr).getResult();
  }

  // String literal
  if (p.at(TK::StringLiteral)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    llvm::StringRef text = p.getText(tok);
    if (text.size() < 2 || text.front() != '"' || text.back() != '"') {
      p.emitError(loc, "unterminated string literal");
      return failure();
    }
    std::string unescaped = unescapeString(text);
    auto attr = csl::ComptimeStringAttr::get(&p.getContext(), unescaped);
    return parsed_csl::ConstantOp::create(builder, loc, attr).getResult();
  }

  // Character literal
  if (p.at(TK::CharacterLiteral)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    llvm::StringRef text = p.getText(tok);
    if (text.size() < 3 || text.front() != '\'' || text.back() != '\'') {
      p.emitError(loc, "unterminated character literal");
      return failure();
    }
    uint64_t value = parseCharLiteral(text);
    APInt apValue(64, value, /*isSigned=*/false);
    auto attr = parsed_csl::ComptimeIntAttr::get(&p.getContext(), apValue);
    return parsed_csl::ConstantOp::create(builder, loc, attr).getResult();
  }

  // Line string literal
  if (p.at(TK::LineString)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    // Line string: \\content\n - strip the leading \\ and trailing newline
    llvm::StringRef text = p.getText(tok);
    if (!text.starts_with("\\\\")) {
      p.emitError(loc, "malformed line string literal");
      return failure();
    }
    text = text.drop_front(2);
    if (text.ends_with("\n"))
      text = text.drop_back(1);
    auto attr = csl::ComptimeStringAttr::get(&p.getContext(), text.str());
    return parsed_csl::ConstantOp::create(builder, loc, attr).getResult();
  }

  // Identifier
  if (p.at(TK::Identifier)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    llvm::StringRef name = p.getText(tok);

    auto symbolValue = p.lookupSymbol(name, loc);
    if (failed(symbolValue)) {
      p.emitError(loc, "unknown identifier: " + name);
      return failure();
    }
    return *symbolValue;
  }

  // Builtin identifier call: @name(args...)
  if (p.at(TK::BuiltinIdentifier)) {
    auto loc = p.startLoc();
    auto tok = p.advance();
    llvm::StringRef fullName = p.getText(tok);
    llvm::StringRef nameWithoutAt = fullName.drop_front(1); // remove '@'

    // Check if this is followed by parentheses (a call)
    if (!p.at(TK::LParen)) {
      // Bare builtin identifier (without call) - error
      return p.emitError(loc.start(), "builtin '@" + nameWithoutAt +
                                          "' must be called with arguments");
    }

    // Determine comptime scope
    Operation *parentOp = builder.getInsertionBlock()->getParentOp();
    bool alreadyInComptimeScope =
        parentOp->hasTrait<parsed_csl::OpTrait::ComptimeScope>() ||
        parentOp->getParentWithTrait<parsed_csl::OpTrait::ComptimeScope>() !=
            nullptr;

    // Helper: visit an argument expression, optionally wrapping in comptime.
    // An argument is read unless the builtin does not read its value.
    auto visitArg = [&](bool wrapInComptime,
                        bool readsArgument) -> FailureOr<Value> {
      auto parseArg = [&]() {
        return readsArgument ? parseValueExpr(p) : parseExpr(p);
      };
      if (wrapInComptime && !alreadyInComptimeScope) {
        auto argLoc = p.startLoc();
        auto comptimeExprOp = parsed_csl::ComptimeExprOp::create(
            builder, argLoc.start(), /*inferred_type=*/nullptr);
        Block &comptimeExprBlock =
            comptimeExprOp.getExprRegion().emplaceBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&comptimeExprBlock);
        auto argResult = parseArg();
        if (failed(argResult))
          return failure();
        parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                    *argResult);
        comptimeExprOp->setLoc(argLoc.toHere());
        return comptimeExprOp.getResult();
      }
      return parseArg();
    };

    // Handle @as and @ptrcast specially: both become a CastOp
    if (nameWithoutAt == "as" || nameWithoutAt == "ptrcast") {
      (void)p.expect(TK::LParen);
      SmallVector<Value> asArgs;
      unsigned argIdx = 0;
      while (!p.at(TK::RParen) && !p.eof()) {
        if (looksLikeItemDecl(p))
          break;
        if (argIdx > 0) {
          if (!p.eat(TK::Comma))
            break;
        }
        auto argResult = visitArg(/*wrapInComptime=*/argIdx == 0,
                                  /*readsArgument=*/true);
        if (failed(argResult))
          break;
        asArgs.push_back(*argResult);
        argIdx++;
      }
      (void)p.expect(TK::RParen);
      if (asArgs.size() != 2)
        return p.emitError(loc.start(), "@" + nameWithoutAt +
                                            " requires exactly 2 arguments");
      return parsed_csl::CastOp::create(builder, loc.toHere(), asArgs[1],
                                        asArgs[0], /*inferred_type=*/{},
                                        /*implicit=*/false)
          .getResult();
    }

    // Look up builtin descriptor for comptime arg wrapping
    std::optional<csl::BuiltinDescriptor> desc;
    if (auto builtinKind = csl::symbolizeBuiltinKind(nameWithoutAt))
      desc = csl::BuiltinDescriptor::get(*builtinKind, &p.getContext());

    // Parse argument list
    (void)p.expect(TK::LParen);
    SmallVector<Value> args;
    unsigned argIdx = 0;
    while (!p.at(TK::RParen) && !p.eof()) {
      if (looksLikeItemDecl(p))
        break;
      if (argIdx > 0) {
        if (!p.eat(TK::Comma))
          break;
      }
      bool wrapThisArg = desc && desc->shouldWrapArgInComptime(argIdx);
      bool readsArgument = !desc || desc->readsArgument(argIdx);
      auto argResult = visitArg(wrapThisArg, readsArgument);
      if (failed(argResult))
        break;
      args.push_back(*argResult);
      argIdx++;
    }
    (void)p.expect(TK::RParen);

    return parsed_csl::BuiltinCallOp::create(builder, loc.toHere(),
                                             nameWithoutAt, args,
                                             /*inferred_type=*/nullptr)
        .getResult();
  }

  // Function type: fn(params) retType  or  task(params) retType
  //   also: async fn(params) retType
  // fn/task followed by an identifier is a declaration, not a type expression.
  // Refuse to parse it here so that error recovery can reach the top level.
  if (p.at(TK::KwFn) || p.at(TK::KwTask) ||
      (p.atIdent("async") && p.nth(1) == TK::KwFn)) {
    auto loc = p.startLoc();
    bool isAsync = p.eatIdent("async");

    if (p.nth(1) == TK::Identifier)
      return p.emitError("expected expression");

    bool isTask = p.at(TK::KwTask);
    p.advance();

    (void)p.expect(TK::LParen);
    SmallVector<Value> paramTypes;
    while (!p.at(TK::RParen) && !p.eof()) {
      if (!paramTypes.empty()) {
        if (!p.eat(TK::Comma))
          break;
      }
      auto paramType = parseValueTypeExpr(p);
      if (failed(paramType))
        return failure();
      paramTypes.push_back(*paramType);
    }
    (void)p.expect(TK::RParen);

    // Return type
    auto retType = parseValueTypeExpr(p);
    if (failed(retType))
      return failure();

    return parsed_csl::FuncTypeOp::create(builder, loc.toHere(), paramTypes,
                                          *retType, isTask, isAsync)
        .getResult();
  }

  // Struct type: struct { ... }
  if (p.at(TK::KwStruct)) {
    auto loc = p.startLoc();
    p.advance();

    (void)p.expect(TK::LBrace);

    SmallVector<std::string> fieldNames;
    SmallVector<Value> fieldTypes;

    while (!p.at(TK::RBrace) && !p.eof()) {
      if (!fieldNames.empty()) {
        if (!p.eat(TK::Comma))
          break;
        if (p.at(TK::RBrace))
          break;
      }
      if (!p.at(TK::Identifier)) {
        p.emitError("expected field name");
        break;
      }
      auto nameTok = p.advance();
      fieldNames.push_back(p.getText(nameTok).str());
      (void)p.expect(TK::Colon);
      auto typeResult = parseValueExpr(p);
      if (failed(typeResult))
        return failure();
      fieldTypes.push_back(*typeResult);
    }
    (void)p.expect(TK::RBrace);

    SmallVector<StringRef> fieldNameRefs;
    for (const auto &name : fieldNames)
      fieldNameRefs.push_back(name);
    auto fieldNamesAttr = builder.getStrArrayAttr(fieldNameRefs);

    return parsed_csl::NamedStructTypeOp::create(
               builder, loc.toHere(), fieldNamesAttr, fieldTypes, loc.toHere())
        .getResult();
  }

  // Enum type: enum(underlying_type) { ... }
  if (p.at(TK::KwEnum)) {
    auto loc = p.startLoc();
    p.advance();

    (void)p.expect(TK::LParen);
    auto underlyingType = parseValueExpr(p);
    if (failed(underlyingType))
      return failure();
    (void)p.expect(TK::RParen);

    (void)p.expect(TK::LBrace);

    SmallVector<std::string> elementNames;
    SmallVector<Value> explicitValues;
    SmallVector<int64_t> explicitIndices;
    size_t idx = 0;

    while (!p.at(TK::RBrace) && !p.eof()) {
      if (!elementNames.empty()) {
        if (!p.eat(TK::Comma))
          break;
        if (p.at(TK::RBrace))
          break;
      }
      if (!p.at(TK::Identifier)) {
        p.emitError("expected enum element name");
        break;
      }
      auto nameTok = p.advance();
      elementNames.push_back(p.getText(nameTok).str());

      // Optional explicit value: = expr
      if (p.eat(TK::Eq)) {
        auto valueResult = parseValueExpr(p);
        if (failed(valueResult))
          return failure();
        explicitValues.push_back(*valueResult);
        explicitIndices.push_back(idx);
      }
      idx++;
    }
    (void)p.expect(TK::RBrace);

    SmallVector<StringRef> elementNameRefs;
    for (const auto &name : elementNames)
      elementNameRefs.push_back(name);
    auto elementNamesAttr = builder.getStrArrayAttr(elementNameRefs);
    auto explicitIndicesAttr = builder.getI64ArrayAttr(explicitIndices);

    return parsed_csl::EnumTypeOp::create(
               builder, loc.toHere(), *underlyingType, elementNamesAttr,
               explicitValues, explicitIndicesAttr, loc.toHere())
        .getResult();
  }

  // Parenthesized expression: ( expr )
  if (p.at(TK::LParen)) {
    p.advance();
    auto expr = parseExpr(p);
    (void)p.expect(TK::RParen);
    return expr;
  }

  // Anonymous struct literal: .{ ... } or tuple literal: .{ expr, ... }
  if (p.at(TK::Dot) && p.nth(1) == TK::LBrace) {
    auto loc = p.startLoc();
    p.advance(); // consume '.'
    p.advance(); // consume '{'

    // Distinguish between struct literal and tuple literal:
    // .{ .field = expr, ... } is struct
    // .{ expr, expr, ... } is tuple
    if (p.at(TK::Dot) && p.nth(1) == TK::Identifier) {
      // Struct literal
      SmallVector<std::string> fieldNameStorage;
      SmallVector<Value> fieldValues;

      while (!p.at(TK::RBrace) && !p.eof()) {
        if (!fieldNameStorage.empty()) {
          if (!p.eat(TK::Comma))
            break;
          // Allow trailing comma
          if (p.at(TK::RBrace))
            break;
        }
        (void)p.expect(TK::Dot);
        if (!p.at(TK::Identifier)) {
          p.emitError("expected field name");
          break;
        }
        auto nameTok = p.advance();
        fieldNameStorage.push_back(p.getText(nameTok).str());
        (void)p.expect(TK::Eq);
        auto valueResult = parseValueExpr(p);
        if (failed(valueResult))
          return failure();
        fieldValues.push_back(*valueResult);
      }
      (void)p.expect(TK::RBrace);

      SmallVector<StringRef> fieldNameRefs;
      for (const auto &name : fieldNameStorage)
        fieldNameRefs.push_back(name);
      auto fieldNamesAttr = builder.getStrArrayAttr(fieldNameRefs);

      return parsed_csl::ConstructStructOp::create(
                 builder, loc.toHere(), unresolvedTy, fieldNamesAttr,
                 /*struct_type=*/Value(), fieldValues,
                 /*inferred_type=*/nullptr)
          .getResult();
    }

    // Tuple literal or empty struct
    if (p.at(TK::RBrace)) {
      // Empty: .{} - treat as empty struct
      p.advance();
      SmallVector<StringRef> emptyNames;
      auto fieldNamesAttr = builder.getStrArrayAttr(emptyNames);
      return parsed_csl::ConstructStructOp::create(
                 builder, loc.toHere(), unresolvedTy, fieldNamesAttr,
                 /*struct_type=*/Value(), /*field_values=*/{},
                 /*inferred_type=*/nullptr)
          .getResult();
    }

    // Tuple literal: .{ expr, expr, ... }
    SmallVector<Value> values;
    while (!p.at(TK::RBrace) && !p.eof()) {
      if (!values.empty()) {
        if (!p.eat(TK::Comma))
          break;
        if (p.at(TK::RBrace))
          break;
      }
      auto valueResult = parseValueExpr(p);
      if (failed(valueResult))
        return failure();
      values.push_back(*valueResult);
    }
    (void)p.expect(TK::RBrace);

    return parsed_csl::ConstructTupleOp::create(builder, loc.toHere(),
                                                unresolvedTy, values,
                                                /*inferred_type=*/nullptr)
        .getResult();
  }

  // Tensor access expression: |vars|{extents} -> base[indices]
  if (p.at(TK::Pipe)) {
    auto loc = p.startLoc();
    p.advance(); // consume '|'

    // Parse induction variable names: i or i, j, ...
    SmallVector<StringAttr> inductionVarNames;
    while (p.at(TK::Identifier)) {
      auto nameTok = p.advance();
      inductionVarNames.push_back(builder.getStringAttr(p.getText(nameTok)));
      if (!p.eat(TK::Comma))
        break;
    }
    (void)p.expect(TK::Pipe);

    // Parse extents: {expr, expr, ...}
    (void)p.expect(TK::LBrace);
    SmallVector<Value> extents;
    while (!p.at(TK::RBrace) && !p.eof()) {
      if (!extents.empty()) {
        if (!p.eat(TK::Comma))
          break;
      }
      auto extentResult = parseValueExpr(p);
      if (failed(extentResult))
        return failure();
      extents.push_back(*extentResult);
    }
    (void)p.expect(TK::RBrace);

    // Expect '->'
    (void)p.expect(TK::Arrow);

    SmallVector<Attribute> namesAttrs(inductionVarNames.begin(),
                                      inductionVarNames.end());
    auto namesArrayAttr = builder.getArrayAttr(namesAttrs);

    auto tensorAccessOp = parsed_csl::TensorAccessOp::create(
        builder, loc.start(), unresolvedTy, /*base=*/extents[0], // placeholder
        extents, namesArrayAttr, /*inferred_type=*/nullptr);

    Block &body = tensorAccessOp.getBody().emplaceBlock();
    SmallVector<BlockArgument> inductionArgs;
    for (size_t i = 0; i < inductionVarNames.size(); ++i)
      inductionArgs.push_back(body.addArgument(unresolvedTy, loc.start()));

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&body);
      Parser::LocalScopeGuard bodyScope(p.getLocalScope());

      for (size_t i = 0; i < inductionVarNames.size(); ++i) {
        (void)p.declareLocalSymbol(inductionVarNames[i].getValue(),
                                   inductionArgs[i], loc.start());
      }

      // Parse `base[indices]` as a single expression.
      auto fullExpr = parseExpr(p);
      if (failed(fullExpr))
        return failure();

      // The result should be an IndexAccessOp: base[i0, i1, ...]
      auto indexOp = mlir::dyn_cast_or_null<parsed_csl::IndexAccessOp>(
          (*fullExpr).getDefiningOp());
      if (!indexOp) {
        p.emitError(loc.start(), "expected 'base[indices]' in tensor access");
        return failure();
      }

      // Extract base and indices from the IndexAccessOp.
      Value base = indexOp.getBase();
      SmallVector<Value> indices(indexOp.getIndices());

      // Erase the IndexAccessOp — we don't need it.
      indexOp.erase();

      // Move the base op (e.g. get_symbol) before the TensorAccessOp,
      // since it should NOT be inside the body region.
      if (Operation *baseOp = base.getDefiningOp()) {
        baseOp->moveBefore(tensorAccessOp);
      }

      // Fix up the TensorAccessOp's base operand.
      tensorAccessOp.getBaseMutable().assign(base);

      parsed_csl::TensorAccessYieldOp::create(builder, builder.getUnknownLoc(),
                                              indices);
    }

    tensorAccessOp->setLoc(loc.toHere());
    return tensorAccessOp.getResult();
  }

  // Anytype
  if (p.at(TK::KwAnytype)) {
    auto tok = p.advance();
    Location loc = p.getLoc(tok);
    return p.emitError(loc, "'anytype' can only appear in parameter types");
  }

  // Nothing matched
  return failure();
}