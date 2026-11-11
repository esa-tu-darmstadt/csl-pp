//===- SharedLiteralParser.cpp - Shared Literal Parsing --------*- C++ -*-===//
//
// Implementation of shared literal parsing utilities.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/IR/SharedLiteralParser.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::typed_csl;

LogicalResult mlir::typed_csl::parseIntegerLiteral(llvm::StringRef str,
                                                   llvm::APInt &result) {
  if (str.empty())
    return failure();

  // Handle different bases
  unsigned radix = 10;

  // Check for negative sign
  bool isNegative = str.starts_with("-");
  if (isNegative || str.starts_with("+"))
    str = str.drop_front();

  // Check for base prefixes
  if (str.starts_with_insensitive("0x")) {
    radix = 16;
    str = str.drop_front(2);
  } else if (str.starts_with_insensitive("0b")) {
    radix = 2;
    str = str.drop_front(2);
  } else if (str.starts_with_insensitive("0o")) {
    radix = 8;
    str = str.drop_front(2);
  }

  // Parse the integer
  APInt value;
  if (str.getAsInteger(radix, value))
    return failure();

  // Apply negative sign if needed
  if (isNegative)
    value = -value;

  result = value;
  return success();
}

LogicalResult mlir::typed_csl::parseFloatLiteral(llvm::StringRef str,
                                                 llvm::APFloat &result) {
  if (str.empty())
    return failure();

  // Use APFloat's conversion from string
  // Default to IEEE double precision
  APFloat value(APFloat::IEEEdouble());
  auto expectedStatus =
      value.convertFromString(str, APFloat::rmNearestTiesToEven);

  // Handle Expected<opStatus> return type (newer LLVM API)
  if (!expectedStatus)
    return failure();

  APFloat::opStatus status = *expectedStatus;
  if (status != APFloat::opOK && status != APFloat::opInexact)
    return failure();

  result = value;
  return success();
}

bool mlir::typed_csl::looksLikeInteger(llvm::StringRef str) {
  if (str.empty())
    return false;

  // Skip optional sign
  if (str.front() == '-' || str.front() == '+')
    str = str.drop_front();

  // Check for base prefixes
  if (str.starts_with_insensitive("0x") || str.starts_with_insensitive("0b") ||
      str.starts_with_insensitive("0o"))
    return true;

  // Check if it's all digits
  return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
}

bool mlir::typed_csl::looksLikeFloat(llvm::StringRef str) {
  if (str.empty())
    return false;

  // Quick heuristic: contains '.' or 'e'/'E'
  return str.contains('.') || str.contains_insensitive('e');
}
