//===- Lexer.h - Parser3 Lexer for CSL -----------------------------------===//
//
// Hand-written lexer for CSL. Produces a TokenStream with trivia attached to
// tokens following the Swift/Roslyn model: every byte of source is accounted
// for by exactly one token or trivia item.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// TableGen-generated TokenKind and TriviaKind enums.
#include "csl-mlir/Target/CSL/Parser3/TokenKind.h.inc"

namespace mlir {
namespace csl {
namespace parser3 {

/// Return a human-readable name for a token kind (e.g. "'fn'", "identifier").
std::string tokenKindName(TokenKind kind);

//===----------------------------------------------------------------------===//
// Trivia
//===----------------------------------------------------------------------===//

/// A piece of trivia (whitespace or comment). References the source buffer
/// by [start, start+length).
struct Trivia {
  TriviaKind kind;
  uint32_t start;
  uint32_t length;
};

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

/// A token with surrounding trivia.
///
/// Leading trivia: everything from end of previous token's trailing trivia
/// to start of this token's text. Trailing trivia: same-line whitespace/
/// comments after this token, up to (not including) the next newline.
///
/// Trivia items are stored in a shared contiguous array (TokenStream::trivia).
/// Each token carries index ranges into that array.
struct Token {
  TokenKind kind;
  uint32_t start;  // byte offset of token text (excluding leading trivia)
  uint32_t length; // byte length of token text

  // Indices into TokenStream::trivia
  uint32_t leadingTriviaStart;
  uint16_t leadingTriviaCount;
  uint16_t trailingTriviaCount;
  // trailing trivia starts at leadingTriviaStart + leadingTriviaCount
};

//===----------------------------------------------------------------------===//
// TokenStream
//===----------------------------------------------------------------------===//

/// The complete lexed representation of a source file.
/// Owns all tokens and trivia. Every byte of source is accounted for:
///   concat(leading_trivia, text, trailing_trivia for each token) == source
struct TokenStream {
  std::vector<Token> tokens;  // all tokens + Eof sentinel
  std::vector<Trivia> trivia; // shared trivia storage
  llvm::StringRef source;     // the source text (not owned)

  /// Get the text of a token.
  llvm::StringRef text(const Token &tok) const {
    return source.substr(tok.start, tok.length);
  }

  /// Get the leading trivia of a token.
  llvm::ArrayRef<Trivia> leadingTrivia(const Token &tok) const {
    return llvm::ArrayRef(trivia.data() + tok.leadingTriviaStart,
                          tok.leadingTriviaCount);
  }

  /// Get the trailing trivia of a token.
  llvm::ArrayRef<Trivia> trailingTrivia(const Token &tok) const {
    return llvm::ArrayRef(trivia.data() + tok.leadingTriviaStart +
                              tok.leadingTriviaCount,
                          tok.trailingTriviaCount);
  }

  /// Reconstruct the original source by concatenating all trivia and token
  /// text in order. If the lexer is correct, the output is byte-identical
  /// to `source`.
  void reconstruct(llvm::raw_ostream &os) const;

  /// Dump a human-readable annotated token stream.
  /// Shows each token with line:col, kind, text, and attached trivia.
  void dump(llvm::raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

class Lexer {
  llvm::StringRef source;
  uint32_t pos = 0;

public:
  explicit Lexer(llvm::StringRef source);

  /// Lex the entire source into a TokenStream.
  /// Every byte is covered by exactly one token or trivia item.
  /// Unrecognized characters become ErrorToken.
  TokenStream lexAll();

private:
  /// Peek at the current character without consuming.
  char peek() const;

  /// Peek at the character at offset from current position.
  char peekAt(uint32_t offset) const;

  /// Advance by one character.
  char advance();

  /// Check if we're at the end of the source.
  bool atEnd() const;

  /// Skip whitespace and comments, recording them as trivia.
  void lexTrivia(std::vector<Trivia> &triviaVec, bool isLeading);

  /// Lex a single token (after trivia has been consumed).
  Token lexToken(std::vector<Trivia> &triviaVec);

  /// Lex an identifier or keyword.
  TokenKind lexIdentifierOrKeyword();

  /// Lex a builtin identifier (@name).
  void lexBuiltinIdentifier();

  /// Lex a number literal (integer or float).
  TokenKind lexNumber();

  /// Lex a string literal.
  void lexStringLiteral();

  /// Lex a character literal.
  void lexCharacterLiteral();

  /// Lex a line string (\\... to end of line).
  void lexLineString();

  /// Check if a character is an identifier start.
  static bool isIdentStart(char c);

  /// Check if a character is an identifier continuation.
  static bool isIdentContinue(char c);

  /// Check if a character is a decimal digit.
  static bool isDigit(char c);

  /// Check if a character is a hex digit.
  static bool isHexDigit(char c);

  /// Classify an identifier as keyword or plain identifier.
  static TokenKind classifyKeyword(llvm::StringRef word);
};

} // namespace parser3
} // namespace csl
} // namespace mlir
