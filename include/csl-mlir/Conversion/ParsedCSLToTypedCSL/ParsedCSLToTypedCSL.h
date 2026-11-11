//===- ParsedCSLToTypedCSL.h - ParsedCSL to TypedCSL conversion -*- C++ -*-===//
//
// This file declares the pass for converting ParsedCSL dialect to TypedCSL
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_CONVERSION_PARSEDCSLTOTYPEDCSL_H
#define CSL_MLIR_CONVERSION_PARSEDCSLTOTYPEDCSL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class TypeConverter;

// namespace parsed_csl {
// std::unique_ptr<Pass> createLowerToTypedCSLPass();
// } // namespace parsed_csl

} // namespace mlir

#endif // CSL_MLIR_CONVERSION_PARSEDCSLTOTYPEDCSL_H
