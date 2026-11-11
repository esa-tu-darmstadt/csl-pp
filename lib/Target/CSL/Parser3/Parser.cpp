//===- Parser.cpp - Parser3 Infrastructure for CSL -----------------------===//
//
// Implementation of token navigation, error recovery, symbol management,
// and IR emission helpers.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Target/CSL/Parser3/Parser.h"

#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinTypes.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Location.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::csl::parser3;

#define DEBUG_TYPE "parser3"

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

Parser::Parser(MLIRContext &ctx, const TokenStream &tokens,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
               unsigned bufferID,
               llvm::DenseMap<llvm::StringRef, ParamBinding> paramBindings,
               unsigned lineOffset, llvm::StringRef sourceFilename,
               parsed_csl::ModuleOp currentModule, bool lspMode,
               const CslFlags &cslFlags, bool isSystemModule,
               std::vector<SyntaxRange> *syntaxRanges)
    : tokens(tokens), sourceMgr(sourceMgr), bufferID(bufferID),
      lineOffset(lineOffset), sourceFilename(sourceFilename), ctx(ctx),
      builder(&ctx), unresolvedTy(parsed_csl::UnresolvedType::get(&ctx)),
      paramBindings(std::move(paramBindings)), currentModule(currentModule),
      lspMode(lspMode), systemModule(isSystemModule), cslFlags(cslFlags),
      syntaxRanges(syntaxRanges), tracePrinter(llvm::dbgs()) {}

//===----------------------------------------------------------------------===//
// Debug Tracing
//===----------------------------------------------------------------------===//

Parser::TraceScope::TraceScope(Parser &p, llvm::StringRef funcName)
    : p(p), funcName(funcName) {
  LLVM_DEBUG({
    auto &printer = p.getTracePrinter();
    printer.startLine() << funcName << " @ ";
    if (!p.eof()) {
      auto [line, col] = p.getLineAndColumn(p.current().start);
      printer.getOStream() << line << ":" << col << " ["
                           << tokenKindName(p.nth(0));
      llvm::StringRef text = p.getText(p.current());
      if (!text.empty() && text.size() <= 20)
        printer.getOStream() << " '" << text << "'";
      printer.getOStream() << "]\n";
    } else {
      printer.getOStream() << "EOF\n";
    }
    printer.indent();
  });
}

Parser::TraceScope::~TraceScope() {
  LLVM_DEBUG({ p.getTracePrinter().unindent(); });
}

//===----------------------------------------------------------------------===//
// Token Navigation
//===----------------------------------------------------------------------===//

TokenKind Parser::nth(unsigned lookahead) const {
  uint32_t idx = pos + lookahead;
  if (idx >= tokens.tokens.size())
    return TokenKind::Eof;
  return tokens.tokens[idx].kind;
}

//===----------------------------------------------------------------------===//
// Token Consumption
//===----------------------------------------------------------------------===//

Token Parser::advance() {
  assert(!eof() && "advancing past EOF");
  const Token &tok = tokens.tokens[pos];
  LLVM_DEBUG({
    tracePrinter.startLine() << "advance: " << tokenKindName(tok.kind);
    llvm::StringRef text = getText(tok);
    if (!text.empty() && text.size() <= 20)
      tracePrinter.getOStream() << " '" << text << "'";
    tracePrinter.getOStream() << "\n";
  });
  ++pos;
  return tok;
}

bool Parser::eat(TokenKind kind) {
  if (at(kind)) {
    advance();
    return true;
  }
  return false;
}

ParseResult Parser::expect(TokenKind kind) {
  if (eat(kind))
    return success();
  // Emit error but do NOT consume the current token.
  return emitError(llvm::Twine("expected ") + tokenKindName(kind));
}

//===----------------------------------------------------------------------===//
// Location
//===----------------------------------------------------------------------===//

std::pair<unsigned, unsigned> Parser::getLineAndColumn(uint32_t offset) const {
  const char *bufStart = sourceMgr->getMemoryBuffer(bufferID)->getBufferStart();
  auto loc = llvm::SMLoc::getFromPointer(bufStart + offset);
  return sourceMgr->getLineAndColumn(loc, bufferID);
}

Location Parser::getLoc() const { return getLoc(current()); }

Location Parser::getLoc(const Token &tok) const {
  auto [lineStart, colStart] = getLineAndColumn(tok.start);
  auto [lineEnd, colEnd] = getLineAndColumn(tok.start + tok.length);
  llvm::StringRef filename = sourceFilename;
  if (filename.empty())
    filename = sourceMgr->getMemoryBuffer(bufferID)->getBufferIdentifier();
  return FileLineColRange::get(&ctx, filename, lineStart + lineOffset, colStart,
                               lineEnd + lineOffset, colEnd);
}

Location Parser::getLocRange(const Token &from, const Token &to) const {
  auto [lineStart, colStart] = getLineAndColumn(from.start);
  auto [lineEnd, colEnd] = getLineAndColumn(to.start + to.length);
  llvm::StringRef filename = sourceFilename;
  if (filename.empty())
    filename = sourceMgr->getMemoryBuffer(bufferID)->getBufferIdentifier();
  return FileLineColRange::get(&ctx, filename, lineStart + lineOffset, colStart,
                               lineEnd + lineOffset, colEnd);
}

Location Parser::getLocPrev() const {
  assert(pos > 0 && "no previous token");
  return getLoc(tokens.tokens[pos - 1]);
}

Parser::PendingLoc Parser::startLoc(Value val) const {
  Location start = getLoc();
  if (auto *defOp = val.getDefiningOp()) {
    // VariableOp means the value was referenced via an identifier that was just
    // consumed — use the reference token, not the declaration site.
    if (mlir::isa<parsed_csl::VariableOp>(defOp) && pos > 0)
      start = getLocPrev();
    else
      start = defOp->getLoc();
  } else if (mlir::isa<BlockArgument>(val) && pos > 0) {
    start = getLocPrev();
  }
  return PendingLoc(this, start);
}

Location Parser::PendingLoc::toHere() const {
  auto startRange = mlir::dyn_cast<FileLineColRange>(start_);
  if (!startRange || p->pos == 0)
    return start_;
  const Token &endTok = p->tokens.tokens[p->pos - 1];
  auto [lineEnd, colEnd] = p->getLineAndColumn(endTok.start + endTok.length);
  return FileLineColRange::get(
      &p->ctx, startRange.getFilename(), startRange.getStartLine(),
      startRange.getStartColumn(), lineEnd + p->lineOffset, colEnd);
}

//===----------------------------------------------------------------------===//
// Error Handling and Recovery
//===----------------------------------------------------------------------===//

InFlightDiagnostic Parser::emitError(const llvm::Twine &msg) {
  return emitError(getLoc(), msg);
}

InFlightDiagnostic Parser::emitError(Location loc, const llvm::Twine &msg) {
  hadError_ = true;
  return mlir::emitError(loc, msg);
}

InFlightDiagnostic Parser::emitWarning(Location loc, const llvm::Twine &msg) {
  return mlir::emitWarning(loc, msg);
}

bool Parser::errRecover(const llvm::Twine &msg, const TokenSet &recovery) {
  // Rule: NEVER eat braces. They always delimit blocks.
  if (at(TokenKind::LBrace) || at(TokenKind::RBrace)) {
    LLVM_DEBUG(tracePrinter.startLine()
               << "errRecover: brace barrier, not consuming\n");
    emitError(msg);
    return true; // don't consume
  }
  // If the token is in the recovery set, it belongs to the parent.
  if (atTs(recovery)) {
    LLVM_DEBUG(tracePrinter.startLine()
               << "errRecover: token in recovery set, not consuming\n");
    emitError(msg);
    return true; // don't consume
  }
  // Unknown junk token: consume it wrapped in an error.
  // At EOF, emit the error but don't try to bump (matches rust-analyzer's
  // bump_any() which is a no-op at EOF).
  LLVM_DEBUG(tracePrinter.startLine() << "errRecover: consuming junk token\n");
  emitError(msg);
  if (!eof())
    advance();
  return false;
}

void Parser::errorBlock(const llvm::Twine &msg) {
  assert(at(TokenKind::LBrace));
  emitError(msg);
  advance(); // consume '{'
  int depth = 1;
  while (!eof() && depth > 0) {
    if (at(TokenKind::LBrace))
      depth++;
    else if (at(TokenKind::RBrace))
      depth--;
    advance();
  }
}

FailureOr<Token> Parser::nameRecover(const TokenSet &recovery) {
  if (at(TokenKind::Identifier))
    return advance();
  errRecover("expected identifier", recovery);
  return failure();
}

void Parser::delimited(TokenKind bra, TokenKind ket, const TokenSet &elemFirst,
                       llvm::function_ref<bool()> parseElement) {
  (void)expect(bra);
  while (!at(ket) && !eof()) {
    advancePush();

    // Handle double-comma: (a, , b) -> emit error, skip comma, continue
    if (at(TokenKind::Comma)) {
      (void)emitError("unexpected ','");
      advance();
      advancePop();
      continue;
    }

    // Try to parse an element
    if (!parseElement()) {
      advanceDrop();
      break; // element parser couldn't parse -> stop the list
    }

    // Comma handling after element
    if (!eat(TokenKind::Comma)) {
      // No comma: if next looks like another element, emit "expected ','"
      // and continue (missing comma). Otherwise, stop.
      if (!atTs(elemFirst)) {
        advancePop();
        break;
      }
      (void)emitError("expected ','");
    }

    advancePop();
  }
  (void)expect(ket);
}

void Parser::advanceWithError(const llvm::Twine &msg) {
  emitError(msg);
  if (!eof())
    advance();
}

//===----------------------------------------------------------------------===//
// Advance Assertions
//===----------------------------------------------------------------------===//

void Parser::advancePush() { advanceStack.push_back(pos); }

void Parser::advancePop() {
  uint32_t savedPos = advanceStack.pop_back_val();
  assert(pos > savedPos && "parser did not advance — possible infinite loop");
}

void Parser::advanceDrop() { advanceStack.pop_back(); }

//===----------------------------------------------------------------------===//
// Symbol Management
//===----------------------------------------------------------------------===//

bool Parser::isBuiltinTypeName(llvm::StringRef name) {
  return csl::isBuiltinTypeName(name);
}

FailureOr<Value> Parser::emitBuiltinType(llvm::StringRef name) {
  auto typeResult = csl::getBuiltinType(name, &ctx);
  if (failed(typeResult))
    return failure();

  auto typeAttr = parsed_csl::TypeAttr::get(*typeResult);
  return parsed_csl::BuiltinTypeOp::create(builder, getLocPrev(), typeAttr)
      .getResult();
}

bool Parser::isBuiltinConstName(llvm::StringRef name) {
  return name == "NORTH" || name == "SOUTH" || name == "EAST" ||
         name == "WEST" || name == "RAMP";
}

FailureOr<Value> Parser::emitBuiltinConst(llvm::StringRef name) {
  std::optional<csl::Direction> dir;
  if (name == "NORTH")
    dir = csl::Direction::NORTH;
  else if (name == "SOUTH")
    dir = csl::Direction::SOUTH;
  else if (name == "EAST")
    dir = csl::Direction::EAST;
  else if (name == "WEST")
    dir = csl::Direction::WEST;
  else if (name == "RAMP")
    dir = csl::Direction::RAMP;
  if (!dir)
    return failure();

  auto attr = csl::DirectionAttr::get(&ctx, *dir);
  return parsed_csl::ConstantOp::create(builder, getLocPrev(), attr)
      .getResult();
}

Value Parser::read(Value value, Location loc) {
  if (!parsed_csl::mayDenoteLocation(value))
    return value;
  return parsed_csl::ReadOp::create(builder, loc, unresolvedTy, value,
                                    /*inferred_type=*/{})
      .getResult();
}

FailureOr<Value> Parser::lookupSymbol(llvm::StringRef name, Location loc) {
  // 1. Builtin types
  auto builtinType = emitBuiltinType(name);
  if (succeeded(builtinType))
    return builtinType;

  // 2. Builtin constants
  auto builtinConst = emitBuiltinConst(name);
  if (succeeded(builtinConst))
    return builtinConst;

  // 3. Local scope
  if (Value localVal = localScope.lookup(name)) {
    if (lspMode) {
      // In LSP mode, wrap in LocOp for hover/go-to-definition
      return parsed_csl::LocOp::create(builder, loc, unresolvedTy, localVal,
                                       /*inferred_type=*/{})
          .getResult();
    }
    return localVal;
  }

  // 4. Global symbols (emit GetSymbolOp)
  if (globalSymbols.count(name)) {
    StringRef symName =
        std::visit([](auto &&arg) -> StringRef { return arg.getSymName(); },
                   globalSymbols[name]);
    return parsed_csl::GetSymbolOp::create(builder, loc, unresolvedTy,
                                           builder.getStringAttr(symName),
                                           /*lexical_module_identifier=*/{},
                                           /*inferred_type=*/{})
        .getResult();
  }

  // 5. Unknown symbol - emit GetSymbolOp for deferred resolution
  //    (or DeferredErrorOp for system modules with warning mode)
  if (systemModule &&
      cslFlags.diagnostics.unresolvedSystemSymbol == DiagLevel::Warn) {
    emitWarning(loc, "unresolved symbol in system module: " + name);
    return parsed_csl::DeferredErrorOp::create(
               builder, loc, unresolvedTy,
               builder.getStringAttr("unknown identifier: " + name))
        .getResult();
  }

  return parsed_csl::GetSymbolOp::create(builder, loc, unresolvedTy,
                                         builder.getStringAttr(name),
                                         /*lexical_module_identifier=*/{},
                                         /*inferred_type=*/{})
      .getResult();
}

bool Parser::declareGlobalSymbol(llvm::StringRef name,
                                 GlobalSymbolEntry entry) {
  if (isBuiltinTypeName(name)) {
    emitError("symbol '" + name + "' shadows builtin type");
    return false;
  }
  if (globalSymbols.count(name)) {
    emitError("symbol '" + name + "' already declared");
    return false;
  }
  globalSymbols[name] = entry;
  return true;
}

LogicalResult Parser::declareLocalSymbol(llvm::StringRef name, Value value,
                                         Location loc) {
  if (isBuiltinTypeName(name) || isBuiltinConstName(name)) {
    auto diag = emitError(loc, "'" + name + "' shadows a builtin");
    if (lspMode) {
      localScope.insert(name, value);
      return success();
    }
    return failure();
  }
  if (localScope.lookup(name)) {
    auto diag =
        emitError(loc, "redeclaration of local variable '" + name + "'");
    if (lspMode) {
      localScope.insert(name, value);
      return success();
    }
    return failure();
  }
  localScope.insert(name, value);
  return success();
}

//===----------------------------------------------------------------------===//
// Binary/Unary Op Helpers
//===----------------------------------------------------------------------===//

FailureOr<csl::BinaryOpKind> Parser::getBinaryOpKind(TokenKind kind) {
  using BOK = csl::BinaryOpKind;
  switch (kind) {
  case TokenKind::Plus:
    return BOK::Add;
  case TokenKind::Minus:
    return BOK::Sub;
  case TokenKind::Star:
    return BOK::Mul;
  case TokenKind::Slash:
    return BOK::Div;
  case TokenKind::Percent:
    return BOK::Mod;
  case TokenKind::Ampersand:
    return BOK::BitAnd;
  case TokenKind::Pipe:
    return BOK::BitOr;
  case TokenKind::Caret:
    return BOK::BitXor;
  case TokenKind::LeftShift:
    return BOK::Shl;
  case TokenKind::RightShift:
    return BOK::Shr;
  case TokenKind::EqualEqual:
    return BOK::Eq;
  case TokenKind::NotEqual:
    return BOK::Ne;
  case TokenKind::Less:
    return BOK::Lt;
  case TokenKind::Greater:
    return BOK::Gt;
  case TokenKind::LessEqual:
    return BOK::Le;
  case TokenKind::GreaterEqual:
    return BOK::Ge;
  default:
    return failure();
  }
}

FailureOr<csl::BinaryOpKind> Parser::getCompoundAssignOpKind(TokenKind kind) {
  using BOK = csl::BinaryOpKind;
  switch (kind) {
  case TokenKind::PlusEq:
    return BOK::Add;
  case TokenKind::MinusEq:
    return BOK::Sub;
  case TokenKind::StarEq:
    return BOK::Mul;
  case TokenKind::SlashEq:
    return BOK::Div;
  case TokenKind::PercentEq:
    return BOK::Mod;
  case TokenKind::AmpersandEq:
    return BOK::BitAnd;
  case TokenKind::PipeEq:
    return BOK::BitOr;
  case TokenKind::CaretEq:
    return BOK::BitXor;
  case TokenKind::LeftShiftEq:
    return BOK::Shl;
  case TokenKind::RightShiftEq:
    return BOK::Shr;
  default:
    return failure();
  }
}
