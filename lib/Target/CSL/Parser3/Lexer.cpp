//===- Lexer.cpp - Parser3 Lexer for CSL ---------------------------------===//
//
// Hand-written lexer for CSL. Produces tokens with trivia following the
// Swift/Roslyn model.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Target/CSL/Parser3/Lexer.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"

// TableGen-generated stringify/symbolize implementations.
#include "csl-mlir/Target/CSL/Parser3/TokenKind.cpp.inc"

using namespace mlir::csl::parser3;
using llvm::StringRef;

//===----------------------------------------------------------------------===//
// tokenKindName
//===----------------------------------------------------------------------===//

std::string mlir::csl::parser3::tokenKindName(TokenKind kind) {
  // Variable-text tokens and special cases get human-readable names.
  switch (kind) {
  case TokenKind::Eof:
    return "end of file";
  case TokenKind::Identifier:
    return "identifier";
  case TokenKind::BuiltinIdentifier:
    return "builtin identifier";
  case TokenKind::IntegerLiteral:
    return "integer literal";
  case TokenKind::FloatingLiteral:
    return "floating-point literal";
  case TokenKind::CharacterLiteral:
    return "character literal";
  case TokenKind::StringLiteral:
    return "string literal";
  case TokenKind::LineString:
    return "line string";
  case TokenKind::ErrorToken:
    return "error";
  default:
    // Fixed-spelling tokens: quote the spelling from stringifyTokenKind.
    return ("'" + llvm::Twine(stringifyTokenKind(kind)) + "'").str();
  }
}

//===----------------------------------------------------------------------===//
// TokenStream::dump
//===----------------------------------------------------------------------===//

/// Compute line:col from a byte offset by scanning the source.
static std::pair<unsigned, unsigned> getLineCol(llvm::StringRef source,
                                                uint32_t offset) {
  unsigned line = 1, col = 1;
  for (uint32_t i = 0; i < offset && i < source.size(); ++i) {
    if (source[i] == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
  }
  return {line, col};
}

/// Write an escaped representation of text suitable for a dump.
/// Newlines become \n, tabs become \t, other control chars become \xHH.
static void writeEscaped(llvm::raw_ostream &os, llvm::StringRef text,
                         unsigned maxLen = 60) {
  bool truncated = text.size() > maxLen;
  llvm::StringRef shown = truncated ? text.take_front(maxLen) : text;
  for (char c : shown) {
    switch (c) {
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    case '\\':
      os << "\\\\";
      break;
    default:
      if (c < 0x20 || c == 0x7f)
        os << llvm::format("\\x%02x", (unsigned char)c);
      else
        os << c;
    }
  }
  if (truncated)
    os << "...";
}

void TokenStream::dump(llvm::raw_ostream &os) const {
  // Trivia is indented to align with the token kind column, making the
  // hierarchy (token → trivia) visually clear.
  auto dumpTrivia = [&](llvm::StringRef label,
                        llvm::ArrayRef<Trivia> triviaItems) {
    for (const Trivia &t : triviaItems) {
      auto [line, col] = getLineCol(source, t.start);
      os << "          "; // indent to token-kind column
      os << llvm::left_justify(label, 9);
      os << llvm::left_justify(stringifyTriviaKind(t.kind), 22);
      os << llvm::format("%u:%u", line, col);
      os << "  \"";
      writeEscaped(os, source.substr(t.start, t.length));
      os << "\"\n";
    }
  };

  for (const Token &tok : tokens) {
    auto [line, col] = getLineCol(source, tok.start);

    // Leading trivia (before the token line, indented)
    dumpTrivia("leading", leadingTrivia(tok));

    // Token line: line:col  Kind  'text'
    os << llvm::format("%4u:%-3u ", line, col);

    // Token kind name (padded)
    std::string kindStr = tokenKindName(tok.kind);
    os << llvm::left_justify(kindStr, 22);

    // Token text (quoted) - only for tokens with variable text.
    // Compare actual text to the stringified kind to detect variable tokens.
    if (tok.kind != TokenKind::Eof &&
        text(tok) != stringifyTokenKind(tok.kind)) {
      os << "'";
      writeEscaped(os, text(tok));
      os << "'";
    }
    os << "\n";

    // Trailing trivia (after the token line, indented)
    dumpTrivia("trailing", trailingTrivia(tok));
  }
}

//===----------------------------------------------------------------------===//
// Lexer implementation
//===----------------------------------------------------------------------===//

Lexer::Lexer(StringRef source) : source(source) {}

char Lexer::peek() const {
  if (pos >= source.size())
    return '\0';
  return source[pos];
}

char Lexer::peekAt(uint32_t offset) const {
  if (pos + offset >= source.size())
    return '\0';
  return source[pos + offset];
}

char Lexer::advance() {
  if (pos >= source.size())
    return '\0';
  return source[pos++];
}

bool Lexer::atEnd() const { return pos >= source.size(); }

bool Lexer::isIdentStart(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool Lexer::isIdentContinue(char c) {
  return isIdentStart(c) || (c >= '0' && c <= '9');
}

bool Lexer::isDigit(char c) { return c >= '0' && c <= '9'; }

bool Lexer::isHexDigit(char c) {
  return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

TokenKind Lexer::classifyKeyword(StringRef word) {
  return llvm::StringSwitch<TokenKind>(word)
      .Case("align", TokenKind::KwAlign)
      .Case("and", TokenKind::KwAnd)
      .Case("anytype", TokenKind::KwAnytype)
      .Case("asm", TokenKind::KwAsm)
      .Case("break", TokenKind::KwBreak)
      .Case("comptime", TokenKind::KwComptime)
      .Case("const", TokenKind::KwConst)
      .Case("continue", TokenKind::KwContinue)
      .Case("else", TokenKind::KwElse)
      .Case("enum", TokenKind::KwEnum)
      .Case("export", TokenKind::KwExport)
      .Case("extern", TokenKind::KwExtern)
      .Case("fn", TokenKind::KwFn)
      .Case("for", TokenKind::KwFor)
      .Case("if", TokenKind::KwIf)
      .Case("inline", TokenKind::KwInline)
      .Case("noinline", TokenKind::KwNoinline)
      .Case("layout", TokenKind::KwLayout)
      .Case("linkname", TokenKind::KwLinkname)
      .Case("or", TokenKind::KwOr)
      .Case("param", TokenKind::KwParam)
      .Case("return", TokenKind::KwReturn)
      .Case("struct", TokenKind::KwStruct)
      .Case("switch", TokenKind::KwSwitch)
      .Case("task", TokenKind::KwTask)
      .Case("var", TokenKind::KwVar)
      .Case("volatile", TokenKind::KwVolatile)
      .Case("while", TokenKind::KwWhile)
      .Case("true", TokenKind::KwTrue)
      .Case("false", TokenKind::KwFalse)
      .Default(TokenKind::Identifier);
}

//===----------------------------------------------------------------------===//
// Trivia lexing
//===----------------------------------------------------------------------===//

void Lexer::lexTrivia(std::vector<Trivia> &triviaVec, bool isLeading) {
  while (!atEnd()) {
    uint32_t triviaStart = pos;
    char c = peek();

    // Whitespace
    if (c == ' ' || c == '\t' || c == '\r' || (c == '\n' && isLeading)) {
      // For leading trivia, consume all whitespace including newlines.
      // For trailing trivia, consume only same-line whitespace.
      while (!atEnd()) {
        c = peek();
        if (c == ' ' || c == '\t' || c == '\r') {
          advance();
        } else if (c == '\n' && isLeading) {
          advance();
        } else {
          break;
        }
      }
      triviaVec.push_back(
          {TriviaKind::Whitespace, triviaStart, pos - triviaStart});
      continue;
    }

    // Newline ends trailing trivia
    if (c == '\n' && !isLeading)
      break;

    // Comments: // ...
    if (c == '/' && peekAt(1) == '/') {
      // Determine comment kind
      TriviaKind commentKind;
      if (peekAt(2) == '!') {
        // //! container doc comment
        commentKind = TriviaKind::ContainerDocComment;
      } else if (peekAt(2) == '/' && peekAt(3) != '/') {
        // /// doc comment (but not ////)
        commentKind = TriviaKind::DocComment;
      } else {
        // // regular comment or //// comment
        commentKind = TriviaKind::LineComment;
      }

      // Consume to end of line (don't consume the newline itself)
      while (!atEnd() && peek() != '\n')
        advance();

      triviaVec.push_back({commentKind, triviaStart, pos - triviaStart});
      continue;
    }

    // Not trivia
    break;
  }
}

//===----------------------------------------------------------------------===//
// Token lexing
//===----------------------------------------------------------------------===//

TokenKind Lexer::lexIdentifierOrKeyword() {
  // Already consumed first char
  while (!atEnd() && isIdentContinue(peek()))
    advance();
  return TokenKind::Identifier; // classified later
}

void Lexer::lexBuiltinIdentifier() {
  // Already consumed '@'
  while (!atEnd() && isIdentContinue(peek()))
    advance();
}

TokenKind Lexer::lexNumber() {
  // We've already consumed the first digit or '0'.
  // Check for base prefixes.
  // peekAt(-1) would be the char we already consumed; check source directly
  char first = source[pos - 1];
  if (first == '0' && !atEnd()) {
    char base = peek();
    if (base == 'b') {
      // Binary: 0b[01_]+
      advance();
      while (!atEnd() && (peek() == '0' || peek() == '1' || peek() == '_'))
        advance();
      return TokenKind::IntegerLiteral;
    }
    if (base == 'o') {
      // Octal: 0o[0-7_]+
      advance();
      while (!atEnd() && ((peek() >= '0' && peek() <= '7') || peek() == '_'))
        advance();
      return TokenKind::IntegerLiteral;
    }
    if (base == 'x') {
      // Hex integer or float: 0x[0-9a-fA-F_]+
      advance();
      while (!atEnd() && (isHexDigit(peek()) || peek() == '_'))
        advance();

      // Check for hex float: 0xHHH.HHH or 0xHHHpEEE
      if (!atEnd() && peek() == '.') {
        // Hex float with dot
        advance();
        while (!atEnd() && (isHexDigit(peek()) || peek() == '_'))
          advance();
        // Optional pN exponent
        if (!atEnd() && (peek() == 'p' || peek() == 'P')) {
          advance();
          if (!atEnd() && (peek() == '+' || peek() == '-'))
            advance();
          while (!atEnd() && (isDigit(peek()) || peek() == '_'))
            advance();
        }
        return TokenKind::FloatingLiteral;
      }
      if (!atEnd() && (peek() == 'p' || peek() == 'P')) {
        // Hex float with exponent only
        advance();
        if (!atEnd() && (peek() == '+' || peek() == '-'))
          advance();
        while (!atEnd() && (isDigit(peek()) || peek() == '_'))
          advance();
        return TokenKind::FloatingLiteral;
      }
      return TokenKind::IntegerLiteral;
    }
  }

  // Decimal: consume remaining digits
  while (!atEnd() && (isDigit(peek()) || peek() == '_'))
    advance();

  // Check for decimal float: NNN.NNN or NNNeNNN
  if (!atEnd() && peek() == '.') {
    // Only if next char is a digit (to avoid confusing with field access)
    if (pos + 1 < source.size() && isDigit(source[pos + 1])) {
      advance(); // consume '.'
      while (!atEnd() && (isDigit(peek()) || peek() == '_'))
        advance();
      // Optional eN exponent
      if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
        advance();
        if (!atEnd() && (peek() == '+' || peek() == '-'))
          advance();
        while (!atEnd() && (isDigit(peek()) || peek() == '_'))
          advance();
      }
      return TokenKind::FloatingLiteral;
    }
  }

  // Check for decimal float with exponent only: NNNeNNN
  if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
    advance();
    if (!atEnd() && (peek() == '+' || peek() == '-'))
      advance();
    while (!atEnd() && (isDigit(peek()) || peek() == '_'))
      advance();
    return TokenKind::FloatingLiteral;
  }

  return TokenKind::IntegerLiteral;
}

void Lexer::lexStringLiteral() {
  // Already consumed opening '"'
  while (!atEnd()) {
    char c = peek();
    if (c == '"') {
      advance();
      return;
    }
    if (c == '\\') {
      advance(); // consume backslash
      if (!atEnd())
        advance(); // consume escaped char
      // For \x and \u, consume additional chars
      if (!atEnd()) {
        char escaped = source[pos - 1];
        if (escaped == 'x') {
          // \xHH
          if (!atEnd() && isHexDigit(peek()))
            advance();
        } else if (escaped == 'u') {
          // \u{HHHH}
          if (!atEnd() && peek() == '{') {
            advance();
            while (!atEnd() && peek() != '}')
              advance();
            if (!atEnd())
              advance(); // consume '}'
          }
        }
      }
      continue;
    }
    if (c == '\n' || c == '\r') {
      // Unterminated string - stop here
      return;
    }
    advance();
  }
}

void Lexer::lexCharacterLiteral() {
  // Already consumed opening '\''
  while (!atEnd()) {
    char c = peek();
    if (c == '\'') {
      advance();
      return;
    }
    if (c == '\\') {
      advance(); // consume backslash
      if (!atEnd())
        advance(); // consume escaped char
      if (!atEnd()) {
        char escaped = source[pos - 1];
        if (escaped == 'x') {
          if (!atEnd() && isHexDigit(peek()))
            advance();
        } else if (escaped == 'u') {
          if (!atEnd() && peek() == '{') {
            advance();
            while (!atEnd() && peek() != '}')
              advance();
            if (!atEnd())
              advance();
          }
        }
      }
      continue;
    }
    if (c == '\n' || c == '\r') {
      return;
    }
    advance();
  }
}

void Lexer::lexLineString() {
  // Already consumed '\\\\'
  // Consume to end of line (including the newline)
  while (!atEnd() && peek() != '\n')
    advance();
  if (!atEnd() && peek() == '\n')
    advance();
}

Token Lexer::lexToken(std::vector<Trivia> &triviaVec) {
  // Record leading trivia start
  uint32_t leadingTriviaStart = triviaVec.size();

  // Lex leading trivia
  lexTrivia(triviaVec, /*isLeading=*/true);

  uint16_t leadingTriviaCount = triviaVec.size() - leadingTriviaStart;

  // Record token start
  uint32_t tokenStart = pos;

  if (atEnd()) {
    // Eof token
    Token tok;
    tok.kind = TokenKind::Eof;
    tok.start = tokenStart;
    tok.length = 0;
    tok.leadingTriviaStart = leadingTriviaStart;
    tok.leadingTriviaCount = leadingTriviaCount;
    tok.trailingTriviaCount = 0;
    return tok;
  }

  char c = advance();
  TokenKind kind;

  switch (c) {
  // Brackets
  case '(':
    kind = TokenKind::LParen;
    break;
  case ')':
    kind = TokenKind::RParen;
    break;
  case '{':
    kind = TokenKind::LBrace;
    break;
  case '}':
    kind = TokenKind::RBrace;
    break;
  case '[':
    kind = TokenKind::LBracket;
    break;
  case ']':
    kind = TokenKind::RBracket;
    break;

  // Single-char punctuation
  case ';':
    kind = TokenKind::Semi;
    break;
  case ':':
    kind = TokenKind::Colon;
    break;
  case ',':
    kind = TokenKind::Comma;
    break;
  case '~':
    kind = TokenKind::Tilde;
    break;

  // Dot variants: . .. ... .* .?
  case '.':
    if (peek() == '.') {
      advance();
      if (peek() == '.') {
        advance();
        kind = TokenKind::Ellipsis;
      } else {
        kind = TokenKind::Range;
      }
    } else if (peek() == '*') {
      advance();
      kind = TokenKind::DotStar;
    } else if (peek() == '?') {
      advance();
      kind = TokenKind::DotQuestion;
    } else {
      kind = TokenKind::Dot;
    }
    break;

  // Star variants: * ** *=
  case '*':
    if (peek() == '*') {
      advance();
      kind = TokenKind::StarStar;
    } else if (peek() == '=') {
      advance();
      kind = TokenKind::StarEq;
    } else {
      kind = TokenKind::Star;
    }
    break;

  // Plus variants: + ++ +=
  case '+':
    if (peek() == '+') {
      advance();
      kind = TokenKind::PlusPlus;
    } else if (peek() == '=') {
      advance();
      kind = TokenKind::PlusEq;
    } else {
      kind = TokenKind::Plus;
    }
    break;

  // Minus variants: - -> -=
  case '-':
    if (peek() == '>') {
      advance();
      kind = TokenKind::Arrow;
    } else if (peek() == '=') {
      advance();
      kind = TokenKind::MinusEq;
    } else {
      kind = TokenKind::Minus;
    }
    break;

  // Less variants: < << <= <<=
  case '<':
    if (peek() == '<') {
      advance();
      if (peek() == '=') {
        advance();
        kind = TokenKind::LeftShiftEq;
      } else {
        kind = TokenKind::LeftShift;
      }
    } else if (peek() == '=') {
      advance();
      kind = TokenKind::LessEqual;
    } else {
      kind = TokenKind::Less;
    }
    break;

  // Greater variants: > >> >= >>=
  case '>':
    if (peek() == '>') {
      advance();
      if (peek() == '=') {
        advance();
        kind = TokenKind::RightShiftEq;
      } else {
        kind = TokenKind::RightShift;
      }
    } else if (peek() == '=') {
      advance();
      kind = TokenKind::GreaterEqual;
    } else {
      kind = TokenKind::Greater;
    }
    break;

  // Equals variants: = == =>
  case '=':
    if (peek() == '=') {
      advance();
      kind = TokenKind::EqualEqual;
    } else if (peek() == '>') {
      advance();
      kind = TokenKind::FatArrow;
    } else {
      kind = TokenKind::Eq;
    }
    break;

  // Exclamation variants: ! !=
  case '!':
    if (peek() == '=') {
      advance();
      kind = TokenKind::NotEqual;
    } else {
      kind = TokenKind::Exclamation;
    }
    break;

  // Ampersand variants: & &=
  case '&':
    if (peek() == '=') {
      advance();
      kind = TokenKind::AmpersandEq;
    } else {
      kind = TokenKind::Ampersand;
    }
    break;

  // Pipe variants: | |= ||
  case '|':
    if (peek() == '|') {
      advance();
      kind = TokenKind::PipePipe;
    } else if (peek() == '=') {
      advance();
      kind = TokenKind::PipeEq;
    } else {
      kind = TokenKind::Pipe;
    }
    break;

  // Caret variants: ^ ^=
  case '^':
    if (peek() == '=') {
      advance();
      kind = TokenKind::CaretEq;
    } else {
      kind = TokenKind::Caret;
    }
    break;

  // Percent variants: % %=
  case '%':
    if (peek() == '=') {
      advance();
      kind = TokenKind::PercentEq;
    } else {
      kind = TokenKind::Percent;
    }
    break;

  // Slash variants: / /=
  case '/':
    if (peek() == '=') {
      advance();
      kind = TokenKind::SlashEq;
    } else {
      kind = TokenKind::Slash;
    }
    break;

  case '?':
    kind = TokenKind::Question;
    break;

  // String literal
  case '"':
    lexStringLiteral();
    kind = TokenKind::StringLiteral;
    break;

  // Character literal
  case '\'':
    lexCharacterLiteral();
    kind = TokenKind::CharacterLiteral;
    break;

  // Builtin identifier: @name
  case '@':
    if (!atEnd() && isIdentStart(peek())) {
      lexBuiltinIdentifier();
      kind = TokenKind::BuiltinIdentifier;
    } else {
      kind = TokenKind::ErrorToken;
    }
    break;

  // Line string: \\...
  case '\\':
    if (peek() == '\\') {
      advance();
      lexLineString();
      kind = TokenKind::LineString;
    } else {
      kind = TokenKind::ErrorToken;
    }
    break;

  default:
    // Identifier or keyword
    if (isIdentStart(c)) {
      lexIdentifierOrKeyword();
      StringRef word = source.substr(tokenStart, pos - tokenStart);
      kind = classifyKeyword(word);
    }
    // Number literal
    else if (isDigit(c)) {
      kind = lexNumber();
    }
    // Unknown character
    else {
      kind = TokenKind::ErrorToken;
    }
    break;
  }

  uint32_t tokenLength = pos - tokenStart;

  // Lex trailing trivia (same-line whitespace/comments only)
  uint32_t trailingTriviaStart = triviaVec.size();
  lexTrivia(triviaVec, /*isLeading=*/false);
  uint16_t trailingTriviaCount = triviaVec.size() - trailingTriviaStart;

  Token tok;
  tok.kind = kind;
  tok.start = tokenStart;
  tok.length = tokenLength;
  tok.leadingTriviaStart = leadingTriviaStart;
  tok.leadingTriviaCount = leadingTriviaCount;
  tok.trailingTriviaCount = trailingTriviaCount;
  return tok;
}

//===----------------------------------------------------------------------===//
// TokenStream::reconstruct
//===----------------------------------------------------------------------===//

void TokenStream::reconstruct(llvm::raw_ostream &os) const {
  for (const Token &tok : tokens) {
    // Emit leading trivia
    for (auto triviaItem : leadingTrivia(tok))
      os << source.substr(triviaItem.start, triviaItem.length);

    // Emit token text
    os << text(tok);

    // Emit trailing trivia
    for (auto triviaItem : trailingTrivia(tok))
      os << source.substr(triviaItem.start, triviaItem.length);
  }
}

//===----------------------------------------------------------------------===//
// lexAll
//===----------------------------------------------------------------------===//

TokenStream Lexer::lexAll() {
  TokenStream stream;
  stream.source = source;

  while (true) {
    Token tok = lexToken(stream.trivia);
    stream.tokens.push_back(tok);
    if (tok.kind == TokenKind::Eof)
      break;
  }

  return stream;
}
