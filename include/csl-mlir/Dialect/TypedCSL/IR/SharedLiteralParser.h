//===- SharedLiteralParser.h - Shared Literal Parsing Utilities -*- C++ -*-===//
//
// Shared utilities for parsing integer and floating-point literals.
// Used by both TypedCSL and ParsedCSL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_IR_SHAREDLITERALPARSER_H
#define CSL_MLIR_DIALECT_TYPEDCSL_IR_SHAREDLITERALPARSER_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace typed_csl {

/// Parse an integer literal from a string.
///
/// Supports decimal, hexadecimal (0x), octal (0o), and binary (0b) formats.
/// Returns failure if the string is not a valid integer.
///
/// Example:
///   parseIntegerLiteral("42")    -> APInt(64, 42)
///   parseIntegerLiteral("0xFF")  -> APInt(64, 255)
///   parseIntegerLiteral("0b101") -> APInt(64, 5)
LogicalResult parseIntegerLiteral(llvm::StringRef str, llvm::APInt &result);

/// Parse a floating-point literal from a string.
///
/// Supports decimal notation with optional exponent.
/// Returns failure if the string is not a valid floating-point number.
///
/// Example:
///   parseFloatLiteral("3.14")    -> APFloat(3.14)
///   parseFloatLiteral("1.5e-10") -> APFloat(1.5e-10)
LogicalResult parseFloatLiteral(llvm::StringRef str, llvm::APFloat &result);

/// Check if a string looks like an integer literal.
/// Does not guarantee it's valid, just a quick check for common formats.
bool looksLikeInteger(llvm::StringRef str);

/// Check if a string looks like a floating-point literal.
/// Does not guarantee it's valid, just a quick check for common formats.
bool looksLikeFloat(llvm::StringRef str);

} // namespace typed_csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_TYPEDCSL_IR_SHAREDLITERALPARSER_H
