//===- Parser.h - Parser3 Infrastructure for CSL -------------------------===//
//
// Hand-written recursive descent parser infrastructure. Provides token
// navigation, error recovery primitives, and the Parser class that grammar
// productions use for parsing and IR emission.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Target/CSL/Parser3/Lexer.h"
#include "csl-mlir/Target/CSL/Parser3/SyntaxKind.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ScopedPrinter.h"

#include <variant>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
namespace csl {
namespace parser3 {

//===----------------------------------------------------------------------===//
// TokenSet - Efficient bitset over TokenKind
//===----------------------------------------------------------------------===//

/// Compact bitset over TokenKind values. Supports O(1) membership test
/// and compile-time construction via constexpr.
class TokenSet {
  // TokenKind values go up to 200 (ErrorToken), so 4 x uint64_t = 256 bits.
  uint64_t bits[4] = {0, 0, 0, 0};

public:
  constexpr TokenSet() = default;

  constexpr TokenSet(std::initializer_list<TokenKind> kinds) {
    for (auto kind : kinds) {
      auto v = static_cast<unsigned>(kind);
      bits[v / 64] |= (uint64_t(1) << (v % 64));
    }
  }

  constexpr bool contains(TokenKind kind) const {
    auto v = static_cast<unsigned>(kind);
    return (bits[v / 64] & (uint64_t(1) << (v % 64))) != 0;
  }

  constexpr TokenSet operator|(const TokenSet &other) const {
    TokenSet result;
    result.bits[0] = bits[0] | other.bits[0];
    result.bits[1] = bits[1] | other.bits[1];
    result.bits[2] = bits[2] | other.bits[2];
    result.bits[3] = bits[3] | other.bits[3];
    return result;
  }
};

//===----------------------------------------------------------------------===//
// Recovery Sets
//===----------------------------------------------------------------------===//

// Atomic sets
constexpr TokenSet LITERAL_FIRST = {
    TokenKind::IntegerLiteral,   TokenKind::FloatingLiteral,
    TokenKind::CharacterLiteral, TokenKind::StringLiteral,
    TokenKind::KwTrue,           TokenKind::KwFalse,
};

constexpr TokenSet ATOM_EXPR_FIRST =
    LITERAL_FIRST | TokenSet{
                        TokenKind::Identifier, TokenKind::BuiltinIdentifier,
                        TokenKind::LParen,     TokenKind::LBracket,
                        TokenKind::LBrace,     TokenKind::Dot,
                        TokenKind::Pipe,       TokenKind::KwStruct,
                        TokenKind::KwEnum,     TokenKind::KwIf,
                        TokenKind::KwSwitch,   TokenKind::KwFn,
                        TokenKind::KwTask,     TokenKind::KwComptime,
                        TokenKind::KwAnytype,  TokenKind::LineString,
                    };

constexpr TokenSet EXPR_FIRST =
    ATOM_EXPR_FIRST | TokenSet{
                          // Prefix operators
                          TokenKind::Star,
                          TokenKind::Ampersand,
                          TokenKind::Exclamation,
                          TokenKind::Tilde,
                          TokenKind::Plus,
                          TokenKind::Minus,
                          TokenKind::Question,
                          // Control flow expressions (Zig-style)
                          TokenKind::KwReturn,
                          TokenKind::KwBreak,
                          TokenKind::KwContinue,
                      };

constexpr TokenSet STMT_FIRST =
    EXPR_FIRST | TokenSet{
                     TokenKind::KwVar, TokenKind::KwConst, TokenKind::KwAsm,
                     TokenKind::KwFor, TokenKind::KwWhile, TokenKind::KwInline,
                 };

constexpr TokenSet ITEM_FIRST = {
    TokenKind::KwFn,       TokenKind::KwTask,   TokenKind::KwVar,
    TokenKind::KwConst,    TokenKind::KwParam,  TokenKind::KwLayout,
    TokenKind::KwComptime, TokenKind::KwExport, TokenKind::KwExtern,
    TokenKind::KwInline,
};

constexpr TokenSet PARAM_FIRST =
    TokenSet{TokenKind::Identifier, TokenKind::KwComptime,
             TokenKind::KwAnytype} |
    EXPR_FIRST;

// Recovery sets
constexpr TokenSet ITEM_RECOVERY = ITEM_FIRST | TokenSet{TokenKind::Semi};

constexpr TokenSet EXPR_RECOVERY = {
    TokenKind::RBrace, TokenKind::RParen, TokenKind::RBracket,
    TokenKind::Comma,  TokenKind::Semi,
};

constexpr TokenSet PARAM_RECOVERY =
    ITEM_RECOVERY | TokenSet{TokenKind::Arrow, TokenKind::RParen};

constexpr TokenSet STMT_RECOVERY = ITEM_RECOVERY;

constexpr TokenSet ARG_RECOVERY = {TokenKind::RParen, TokenKind::Semi};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

class Parser {
public:
  /// Global symbol entry: GlobalOp or GenericFuncOp (symbol-based lookup)
  using GlobalSymbolEntry =
      std::variant<parsed_csl::GlobalOp, parsed_csl::GenericFuncOp>;

  /// Helper typedefs for scoped local symbol table (SSA-based lookup)
  using LocalScopeTy = llvm::ScopedHashTable<llvm::StringRef, mlir::Value>;
  using LocalScopeGuard =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

  Parser(MLIRContext &ctx, const TokenStream &tokens,
         const std::shared_ptr<llvm::SourceMgr> &sourceMgr, unsigned bufferID,
         llvm::DenseMap<llvm::StringRef, ParamBinding> paramBindings,
         unsigned lineOffset, llvm::StringRef sourceFilename,
         parsed_csl::ModuleOp currentModule, bool lspMode,
         const CslFlags &cslFlags, bool isSystemModule,
         std::vector<SyntaxRange> *syntaxRanges);

  //===--------------------------------------------------------------------===//
  // Token Navigation
  //===--------------------------------------------------------------------===//

  /// Get the kind of the current token.
  TokenKind nth(unsigned lookahead = 0) const;

  /// Check if the current token is of the given kind.
  bool at(TokenKind kind) const { return nth(0) == kind; }

  /// Check if the current token is in the given set.
  bool atTs(const TokenSet &set) const { return set.contains(nth(0)); }

  /// Check if the current token is an identifier with the given name.
  /// Used for context-sensitive (soft) keywords like async/await/suspend.
  bool atIdent(llvm::StringRef name) const {
    return at(TokenKind::Identifier) && tokens.text(current()) == name;
  }

  /// Check if the nth lookahead token is an identifier with the given name.
  bool nthIsIdent(unsigned lookahead, llvm::StringRef name) const {
    uint32_t idx = pos + lookahead;
    if (idx >= tokens.tokens.size())
      return false;
    return tokens.tokens[idx].kind == TokenKind::Identifier &&
           tokens.text(tokens.tokens[idx]) == name;
  }

  /// Check if at end of file.
  bool eof() const { return at(TokenKind::Eof); }

  /// Get the current token.
  const Token &current() const { return tokens.tokens[pos]; }

  /// Get the text of a token.
  llvm::StringRef getText(const Token &tok) const { return tokens.text(tok); }

  /// Get the leading trivia of a token.
  llvm::ArrayRef<Trivia> leadingTrivia(const Token &tok) const {
    return tokens.leadingTrivia(tok);
  }

  /// Get the source text of a trivia item.
  llvm::StringRef triviaText(const Trivia &t) const {
    return tokens.source.substr(t.start, t.length);
  }

  //===--------------------------------------------------------------------===//
  // Token Consumption
  //===--------------------------------------------------------------------===//

  /// Consume and return the current token.
  Token advance();

  /// If the current token is of the given kind, consume it and return true.
  bool eat(TokenKind kind);

  /// If the current token is an identifier with the given name, consume it
  /// and return true. Used for soft keywords.
  bool eatIdent(llvm::StringRef name) {
    if (atIdent(name)) {
      advance();
      return true;
    }
    return false;
  }

  /// Expect a token of the given kind. Emits error if not found.
  /// Does NOT consume on failure (critical for recovery).
  ParseResult expect(TokenKind kind);

  //===--------------------------------------------------------------------===//
  // Error Handling and Recovery
  //===--------------------------------------------------------------------===//

  /// Emit an error at the current token location.
  InFlightDiagnostic emitError(const llvm::Twine &msg);

  /// Emit an error at a specific location.
  InFlightDiagnostic emitError(Location loc, const llvm::Twine &msg);

  /// Emit a warning at a specific location.
  InFlightDiagnostic emitWarning(Location loc, const llvm::Twine &msg);

  /// Whether this parser instance emitted an error.
  bool hadError() const { return hadError_; }

  /// The three-way recovery decision (from rust-analyzer).
  /// Emit error. If current token is in recovery set or is { / },
  /// don't consume (return true). Otherwise consume as junk (return false).
  bool errRecover(const llvm::Twine &msg, const TokenSet &recovery);

  /// Consume a balanced { ... } block as an error node.
  void errorBlock(const llvm::Twine &msg);

  /// Parse a required identifier with recovery.
  /// Returns failure if name missing (but error is emitted).
  FailureOr<Token> nameRecover(const TokenSet &recovery);

  /// Parse a comma-separated list between bra/ket delimiters.
  /// parseElement returns true if it parsed an element, false to stop.
  void delimited(TokenKind bra, TokenKind ket, const TokenSet &elemFirst,
                 llvm::function_ref<bool()> parseElement);

  /// Advance and emit an error for an unexpected token.
  void advanceWithError(const llvm::Twine &msg);

  //===--------------------------------------------------------------------===//
  // Advance Assertions (infinite loop prevention)
  //===--------------------------------------------------------------------===//

  /// Push current position onto assertion stack.
  void advancePush();

  /// Pop and assert that position advanced since push.
  void advancePop();

  /// Pop without assertion (for expected non-advance).
  void advanceDrop();

  //===--------------------------------------------------------------------===//
  // Debug Tracing (--debug-only=parser3)
  //===--------------------------------------------------------------------===//

  /// Scoped printer for indented debug output.
  llvm::ScopedPrinter &getTracePrinter() { return tracePrinter; }

  /// RAII helper that traces entry/exit of a parse function with indentation.
  class TraceScope {
    Parser &p;
    llvm::StringRef funcName;

  public:
    TraceScope(Parser &p, llvm::StringRef funcName);
    ~TraceScope();
    TraceScope(const TraceScope &) = delete;
    TraceScope &operator=(const TraceScope &) = delete;
  };

  //===--------------------------------------------------------------------===//
  // Location
  //===--------------------------------------------------------------------===//

  /// Get the MLIR location for the current token.
  Location getLoc() const;

  /// Get the MLIR location for a specific token.
  Location getLoc(const Token &tok) const;

  /// Get the MLIR location spanning from token \p from to token \p to.
  Location getLocRange(const Token &from, const Token &to) const;

  /// Get the MLIR location for the previously consumed token.
  Location getLocPrev() const;

  /// A location whose end is not yet known. Created by startLoc(), finalized
  /// to a spanning FileLineColRange by calling toHere().
  class PendingLoc {
    friend class Parser;
    const Parser *p;
    Location start_;
    PendingLoc(const Parser *p, Location start) : p(p), start_(start) {}

  public:
    /// Get the start location.
    Location start() const { return start_; }

    /// Finalize: span from start to the end of the last consumed token.
    Location toHere() const;
  };

  /// Begin a location span from the current token.
  PendingLoc startLoc() const { return PendingLoc(this, getLoc()); }

  /// Begin a location span from a value's origin. For OpResults, uses the
  /// defining op's location. For BlockArguments, uses the previously consumed
  /// token (the referencing identifier). Must be called right after parsing
  /// the value, before advancing further.
  PendingLoc startLoc(Value val) const;

  //===--------------------------------------------------------------------===//
  // Syntax Range Support
  //===--------------------------------------------------------------------===//

  /// RAII scope guard that records a SyntaxRange on destruction.
  class ScopeGuard {
    friend class Parser;
    Parser *p;
    SyntaxKind kind;
    uint32_t start;
    ScopeGuard(Parser *p, SyntaxKind kind, uint32_t start)
        : p(p), kind(kind), start(start) {}

  public:
    ~ScopeGuard() {
      if (p && p->syntaxRanges)
        p->syntaxRanges->push_back({kind, start, p->currentOffset()});
    }
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;
    ScopeGuard(ScopeGuard &&other)
        : p(other.p), kind(other.kind), start(other.start) {
      other.p = nullptr;
    }
    ScopeGuard &operator=(ScopeGuard &&) = delete;
  };

  /// Create a scope guard for a syntax range.
  ScopeGuard scope(SyntaxKind kind) {
    return ScopeGuard(syntaxRanges ? this : nullptr, kind, currentOffset());
  }

  /// Get the current byte offset in the source.
  uint32_t currentOffset() const { return tokens.tokens[pos].start; }

  //===--------------------------------------------------------------------===//
  // IR Emission Infrastructure
  //===--------------------------------------------------------------------===//

  MLIRContext &getContext() { return ctx; }
  OpBuilder &getBuilder() { return builder; }

  /// Check if running in LSP mode (error-tolerant parsing).
  bool isLspMode() const { return lspMode; }

  /// Check if parsing a system module.
  bool isParsingSystemModule() const { return systemModule; }

  /// Get the compiler flags.
  const CslFlags &getCslFlags() const { return cslFlags; }

  /// Get the unresolved type.
  parsed_csl::UnresolvedType getUnresolvedTy() const { return unresolvedTy; }

  //===--------------------------------------------------------------------===//
  // Context Queries
  //===--------------------------------------------------------------------===//

  bool inFunctionContext() const { return currentFunction != nullptr; }
  parsed_csl::GenericFuncOp getCurrentFunction() { return currentFunction; }
  void setCurrentFunction(parsed_csl::GenericFuncOp f) { currentFunction = f; }

  bool inModuleContext() const { return currentModule != nullptr; }
  parsed_csl::ModuleOp getCurrentModule() { return currentModule; }
  void setCurrentModule(parsed_csl::ModuleOp m) { currentModule = m; }

  //===--------------------------------------------------------------------===//
  // Symbol Management
  //===--------------------------------------------------------------------===//

  /// Check if a name is a builtin type name.
  static bool isBuiltinTypeName(llvm::StringRef name);

  /// Emit a BuiltinTypeOp for the given builtin type name.
  FailureOr<mlir::Value> emitBuiltinType(llvm::StringRef name);

  /// Check if a name is a builtin constant name.
  static bool isBuiltinConstName(llvm::StringRef name);

  /// Emit a ConstantOp for the given builtin constant name.
  FailureOr<mlir::Value> emitBuiltinConst(llvm::StringRef name);

  /// Lookup a symbol by name.
  FailureOr<mlir::Value> lookupSymbol(llvm::StringRef name, Location loc);

  /// The value of `value` at this point of the evaluation: a ReadOp if
  /// `value` may denote a location, `value` itself otherwise. Every construct
  /// that uses an expression as a value calls this right after parsing the
  /// expression, before parsing anything that follows, so the read is where
  /// the source reads the value. `loc` is the source range of the
  /// expression; a reference to a local variable has no op of its own.
  mlir::Value read(mlir::Value value, Location loc);

  /// Declare a global symbol.
  bool declareGlobalSymbol(llvm::StringRef name, GlobalSymbolEntry entry);

  /// Declare a local symbol.
  LogicalResult declareLocalSymbol(llvm::StringRef name, mlir::Value value,
                                   Location loc);

  /// Get the local scope for creating scope guards.
  LocalScopeTy &getLocalScope() { return localScope; }

  /// Get the param bindings map.
  llvm::DenseMap<llvm::StringRef, ParamBinding> &getParamBindings() {
    return paramBindings;
  }

  /// Mark a param binding as used.
  void markParamUsed(llvm::StringRef name) { usedParamBindings.insert(name); }

  /// Check if a param binding was used.
  bool isParamUsed(llvm::StringRef name) const {
    return usedParamBindings.contains(name);
  }

  //===--------------------------------------------------------------------===//
  // Binary/Unary Op Helpers
  //===--------------------------------------------------------------------===//

  /// Maps operator token kind to BinaryOpKind.
  static FailureOr<csl::BinaryOpKind> getBinaryOpKind(TokenKind kind);

  /// Maps operator token kind to the compound-assignment base op.
  /// E.g. PlusEq -> add, StarEq -> mul
  static FailureOr<csl::BinaryOpKind> getCompoundAssignOpKind(TokenKind kind);

private:
  // --- Token stream ---
  const TokenStream &tokens;
  uint32_t pos = 0;

  // --- Source location mapping ---
  const std::shared_ptr<llvm::SourceMgr> &sourceMgr;
  unsigned bufferID;
  unsigned lineOffset;
  llvm::StringRef sourceFilename;

  // --- IR emission infrastructure ---
  MLIRContext &ctx;
  OpBuilder builder;
  parsed_csl::UnresolvedType unresolvedTy;

  // --- Symbol tables ---
  LocalScopeTy localScope;
  llvm::DenseMap<llvm::StringRef, GlobalSymbolEntry> globalSymbols;

  // --- Param bindings ---
  llvm::DenseMap<llvm::StringRef, ParamBinding> paramBindings;
  llvm::DenseSet<llvm::StringRef> usedParamBindings;

  // --- State ---
  parsed_csl::GenericFuncOp currentFunction = nullptr;
  parsed_csl::ModuleOp currentModule = nullptr;
  bool lspMode;
  bool systemModule;
  bool hadError_ = false;
  CslFlags cslFlags;

  // --- Advance assertions ---
  llvm::SmallVector<uint32_t, 8> advanceStack;

  // --- Syntax ranges (optional) ---
  std::vector<SyntaxRange> *syntaxRanges;

  // --- Debug tracing ---
  llvm::ScopedPrinter tracePrinter;

  /// Compute line and column from byte offset.
  std::pair<unsigned, unsigned> getLineAndColumn(uint32_t offset) const;
};

//===----------------------------------------------------------------------===//
// Grammar entry point (implemented in Grammar.cpp)
//===----------------------------------------------------------------------===//

/// Parse the entire file into the given region.
void parseContainerMembers(Parser &p, Region &region);

} // namespace parser3
} // namespace csl
} // namespace mlir
