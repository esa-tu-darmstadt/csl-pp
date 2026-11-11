//===- CSLBitcastFolders.cpp - Bitcast Folder Helpers -----------*- C++ -*-===//
//
// Helper functions for bitcasting between float and integer representations.
// Handles custom float types (like CB16) whose fltSemantics is not registered
// with LLVM's APFloat bitcast dispatch.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include <cmath>

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Custom float type detection
//===----------------------------------------------------------------------===//

/// Returns true if the given FloatType has custom (non-LLVM-builtin) semantics
/// that APFloat::bitcastToAPInt() / APFloat(sem, APInt) cannot handle.
static bool hasCustomSemantics(FloatType ft) {
  return mlir::isa<csl::CB16Type>(ft);
}

//===----------------------------------------------------------------------===//
// Custom float → APInt
//===----------------------------------------------------------------------===//

/// Bitcast a custom-semantics APFloat to APInt by decoding through double.
/// The custom format must be IEEE-like and fit losslessly in double.
static APInt customFloatToAPInt(const APFloat &value,
                                const llvm::fltSemantics &sem) {
  unsigned sizeInBits = APFloat::getSizeInBits(sem);
  unsigned mantissaBits = sem.precision - 1;
  unsigned expBits = sizeInBits - 1 - mantissaBits;
  int bias = -(sem.minExponent - 1);
  uint64_t expMask = (1ULL << expBits) - 1;
  uint64_t mantMask = (1ULL << mantissaBits) - 1;

  if (value.isZero()) {
    uint64_t bits = 0;
    if (value.isNegative())
      bits |= 1ULL << (sizeInBits - 1);
    return APInt(sizeInBits, bits);
  }
  if (value.isInfinity()) {
    uint64_t bits = expMask << mantissaBits;
    if (value.isNegative())
      bits |= 1ULL << (sizeInBits - 1);
    return APInt(sizeInBits, bits);
  }
  if (value.isNaN()) {
    uint64_t bits = expMask << mantissaBits;
    bits |= 1ULL << (mantissaBits - 1); // quiet NaN bit
    if (value.isNegative())
      bits |= 1ULL << (sizeInBits - 1);
    return APInt(sizeInBits, bits);
  }

  // Finite non-zero: convert to double to extract the value.
  APFloat asDouble(value);
  bool lossy;
  asDouble.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &lossy);
  double d = asDouble.convertToDouble();
  double absVal = std::fabs(d);

  // Decompose: absVal = frac * 2^rawExp, where 0.5 <= frac < 1.
  int rawExp;
  double frac = std::frexp(absVal, &rawExp);
  // We want significand in [1, 2): significand = frac * 2, realExp = rawExp-1.
  int realExp = rawExp - 1;
  int biasedExp = realExp + bias;

  uint64_t mantissa;
  if (biasedExp <= 0) {
    // Denormal.
    biasedExp = 0;
    mantissa = static_cast<uint64_t>(
        std::ldexp(absVal, -sem.minExponent + static_cast<int>(mantissaBits)) +
        0.5);
  } else {
    // Normal: strip the implicit leading 1.
    double significand = frac * 2.0;
    mantissa = static_cast<uint64_t>(
        (significand - 1.0) * (1ULL << mantissaBits) + 0.5);
  }

  uint64_t bits = 0;
  if (value.isNegative())
    bits |= 1ULL << (sizeInBits - 1);
  bits |= (static_cast<uint64_t>(biasedExp) & expMask) << mantissaBits;
  bits |= mantissa & mantMask;
  return APInt(sizeInBits, bits);
}

//===----------------------------------------------------------------------===//
// APInt → custom float
//===----------------------------------------------------------------------===//

/// Construct an APFloat with custom semantics from raw bits by decoding
/// through double and then converting to the target semantics.
static APFloat apIntToCustomFloat(const APInt &bits,
                                  const llvm::fltSemantics &sem) {
  unsigned sizeInBits = APFloat::getSizeInBits(sem);
  unsigned mantissaBits = sem.precision - 1;
  unsigned expBits = sizeInBits - 1 - mantissaBits;
  int bias = -(sem.minExponent - 1);
  uint64_t expMask = (1ULL << expBits) - 1;
  uint64_t mantMask = (1ULL << mantissaBits) - 1;

  uint64_t raw = bits.getZExtValue();
  bool negative = (raw >> (sizeInBits - 1)) & 1;
  uint64_t biasedExp = (raw >> mantissaBits) & expMask;
  uint64_t mantissa = raw & mantMask;

  if (biasedExp == expMask) {
    if (mantissa == 0)
      return APFloat::getInf(sem, negative);
    return APFloat::getNaN(sem, negative, mantissa);
  }

  if (biasedExp == 0 && mantissa == 0)
    return APFloat::getZero(sem, negative);

  double value;
  if (biasedExp == 0) {
    // Denormal.
    value = std::ldexp(static_cast<double>(mantissa),
                       sem.minExponent - static_cast<int>(mantissaBits));
  } else {
    // Normal.
    int realExp = static_cast<int>(biasedExp) - bias;
    double significand = 1.0 + static_cast<double>(mantissa) /
                                   static_cast<double>(1ULL << mantissaBits);
    value = std::ldexp(significand, realExp);
  }
  if (negative)
    value = -value;

  // Create as double, then convert to target semantics (lossless).
  APFloat asDouble(value);
  bool lossy;
  asDouble.convert(sem, APFloat::rmNearestTiesToEven, &lossy);
  return asDouble;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

APInt csl::floatBitcastToAPInt(const APFloat &value, FloatType floatType) {
  if (hasCustomSemantics(floatType))
    return customFloatToAPInt(value, floatType.getFloatSemantics());
  return value.bitcastToAPInt();
}

APFloat csl::apIntBitcastToFloat(const APInt &bits, FloatType floatType) {
  const llvm::fltSemantics &sem = floatType.getFloatSemantics();
  if (hasCustomSemantics(floatType))
    return apIntToCustomFloat(bits, sem);
  return APFloat(sem, bits);
}
