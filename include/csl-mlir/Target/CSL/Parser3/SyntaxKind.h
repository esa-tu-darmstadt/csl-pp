//===- SyntaxKind.h - Syntax range kinds for formatting -------------------===//
//
// Optional syntax ranges that the parser can emit for future formatting
// support. Zero overhead when disabled (syntaxRanges pointer is null).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

// TableGen-generated SyntaxKind enum.
#include "csl-mlir/Target/CSL/Parser3/SyntaxKind.h.inc"

namespace mlir {
namespace csl {
namespace parser3 {

/// A syntax range marking a syntactic construct in the source.
struct SyntaxRange {
  SyntaxKind kind;
  uint32_t start; // byte offset
  uint32_t end;   // byte offset (exclusive)
};

} // namespace parser3
} // namespace csl
} // namespace mlir
