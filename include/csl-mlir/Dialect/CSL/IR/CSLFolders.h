//===- CSLFolders.h - CSL Folder Helper Functions --------------*- C++ -*-===//
//
// Helper functions for constant folding of cast and binary operations.
// These helpers support both comptime (arbitrary precision) and runtime
// (fixed-width) types, and are shared between ParsedCSL and TypedCSL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_IR_CSLFOLDERS_H
#define CSL_MLIR_DIALECT_CSL_IR_CSLFOLDERS_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/LogicalResult.h"
#include <variant>

namespace mlir {
namespace csl {

//===----------------------------------------------------------------------===//
// Binary Operation Helpers
//===----------------------------------------------------------------------===//

/// A binary operand: either just a Type (when the value is not comptime-known)
/// or a TypedAttr (when we have a comptime-known value).
using TypeOrAttr = std::variant<Type, TypedAttr>;

/// Extract the type from a TypeOrAttr.
inline Type getType(TypeOrAttr toa) {
  if (auto *t = std::get_if<Type>(&toa))
    return *t;
  return std::get<TypedAttr>(toa).getType();
}

/// Check if a TypeOrAttr carries a comptime-known value.
inline bool isComptimeKnown(TypeOrAttr toa) {
  return std::holds_alternative<TypedAttr>(toa);
}

/// Result of inferBinaryOp: the target type for each operand after implicit
/// coercion, plus the result (either a Type or a comptime-evaluated TypedAttr).
struct BinaryOpResult {
  Type lhsTarget;    // type to cast LHS operand to
  Type rhsTarget;    // type to cast RHS operand to
  TypeOrAttr result; // Type if not evaluable, TypedAttr if evaluated
};

/// Single source of truth for binary operation type inference and evaluation.
///
/// Accepts operands as Type (type-only) or TypedAttr (comptime-known value).
/// Returns target types for each operand and the result, which is a TypedAttr
/// if both operands are comptime-known and the operation can be evaluated, or
/// a Type otherwise.
///
/// For shift operations, the shift amount (RHS) doesn't need to match the
/// value type (LHS). When the LHS is comptime_int and the RHS is
/// comptime-known, the result stays comptime_int. Shift amounts must be
/// non-negative integers.
FailureOr<BinaryOpResult> inferBinaryOp(BinaryOpKind opcode, TypeOrAttr lhs,
                                        TypeOrAttr rhs,
                                        ::mlir::Diagnostic *diag = nullptr);

/// Get the common type for two types with implicit coercion support.
/// Allows comptime_int → runtime int and comptime_float → runtime float.
/// Returns the more specific (runtime) type when coercion is possible.
/// Note: For binary operations, use inferBinaryOp instead.
FailureOr<Type> getCommonType(Type lhsType, Type rhsType);

//===----------------------------------------------------------------------===//
// Unary Operation Helpers
//===----------------------------------------------------------------------===//

/// Evaluate a unary operation on runtime integer attributes.
/// Checks for overflow/underflow and returns failure on error.
FailureOr<TypedAttr> evaluateRuntimeIntUnary(UnaryOpKind opcode,
                                             IntegerAttr operand,
                                             ::mlir::Diagnostic *diag);

/// Evaluate a unary operation on comptime integer attributes.
/// Uses arbitrary precision arithmetic.
FailureOr<TypedAttr> evaluateComptimeIntUnary(UnaryOpKind opcode,
                                              ComptimeIntAttr operand,
                                              ::mlir::Diagnostic *diag);

/// Evaluate a unary operation on boolean attributes.
FailureOr<TypedAttr> evaluateBoolUnary(UnaryOpKind opcode, IntegerAttr operand,
                                       ::mlir::Diagnostic *diag);

/// Evaluate a unary operation on runtime float attributes.
FailureOr<TypedAttr> evaluateRuntimeFloatUnary(UnaryOpKind opcode,
                                               FloatAttr operand,
                                               ::mlir::Diagnostic *diag);

/// Evaluate a unary operation on comptime float attributes.
FailureOr<TypedAttr> evaluateComptimeFloatUnary(UnaryOpKind opcode,
                                                ComptimeFloatAttr operand,
                                                ::mlir::Diagnostic *diag);

/// Infer the result type of a unary operation.
/// For unary operations, result type is always the same as operand type.
FailureOr<Type> inferUnaryResultType(UnaryOpKind opcode, Type operandType,
                                     ::mlir::Diagnostic *diag = nullptr);

/// Unified unary operation evaluation without implicit casts.
/// This is the main entry point for TypedCSL unary operation folding.
FailureOr<TypedAttr> evaluateUnaryStrict(UnaryOpKind opcode, TypedAttr operand,
                                         ::mlir::Diagnostic *diag = nullptr);

/// Unified unary operation evaluation with implicit cast support.
/// This is the main entry point for ParsedCSL unary operation folding.
FailureOr<TypedAttr> evaluateUnary(UnaryOpKind opcode, TypedAttr operand,
                                   ::mlir::Diagnostic *diag = nullptr);

//===----------------------------------------------------------------------===//
// Cast Operation Helpers
//===----------------------------------------------------------------------===//

/// Check if a cast is allowed as an implicit cast.
/// Only comptime_int → runtime int and comptime_float → runtime float are
/// allowed as implicit casts. All other casts must be explicit.
/// When comptimeKnown is true, also allows integer-to-integer casts of
/// different widths (the actual value check happens in evaluateCast).
bool isImplicitCastAllowed(Type sourceType, Type targetType,
                           bool comptimeKnown = false);

/// Evaluate an implicit cast from source to target type.
/// First checks if the cast is allowed as an implicit cast using
/// isImplicitCastAllowed(). If not allowed, returns failure with diagnostics.
/// Otherwise, performs the cast using evaluateCast().
FailureOr<TypedAttr> evaluateImplicitCast(TypedAttr srcAttr, Type targetType,
                                          ::mlir::Diagnostic *diag = nullptr);

/// Unified cast evaluation function.
/// Dispatches to the appropriate specific cast helper based on source attribute
/// type and target type. This is the main entry point for cast constant
/// folding. Returns the casted attribute on success, or failure with optional
/// diagnostics.
FailureOr<TypedAttr> evaluateCast(TypedAttr srcAttr, Type targetType,
                                  ::mlir::Diagnostic *diag = nullptr);

/// Check if an integer value fits in the target type
/// Handles signed/unsigned conversions and width changes.
LogicalResult checkIntFitsInType(const llvm::APInt &value, bool valueIsSigned,
                                 IntegerType targetType,
                                 ::mlir::Diagnostic *diag);

/// Cast runtime integer to runtime integer
/// Checks for overflow and performs sign/width conversion.
FailureOr<TypedAttr> castRuntimeIntToRuntimeInt(IntegerAttr srcAttr,
                                                IntegerType targetType,
                                                ::mlir::Diagnostic *diag);

/// Cast runtime integer to runtime float (MLIR builtins)
FailureOr<TypedAttr> castRuntimeIntToRuntimeFloat(IntegerAttr srcAttr,
                                                  FloatType targetType,
                                                  ::mlir::Diagnostic *diag);

/// Cast runtime float to runtime integer (MLIR builtins)
FailureOr<TypedAttr> castRuntimeFloatToRuntimeInt(FloatAttr srcAttr,
                                                  IntegerType targetType,
                                                  ::mlir::Diagnostic *diag);

/// Cast runtime float to runtime float (MLIR builtins)
/// Converts between different floating-point precisions.
FailureOr<TypedAttr> castRuntimeFloatToRuntimeFloat(FloatAttr srcAttr,
                                                    FloatType targetType,
                                                    ::mlir::Diagnostic *diag);

/// Cast comptime integer to runtime integer (MLIR builtin)
/// Checks for overflow based on target type width and signedness.
FailureOr<TypedAttr> castComptimeIntToRuntimeInt(ComptimeIntAttr srcAttr,
                                                 IntegerType targetType,
                                                 ::mlir::Diagnostic *diag);

/// Cast comptime integer to runtime float (MLIR builtin)
FailureOr<TypedAttr> castComptimeIntToRuntimeFloat(ComptimeIntAttr srcAttr,
                                                   FloatType targetType,
                                                   ::mlir::Diagnostic *diag);

/// Cast comptime float to runtime float (MLIR builtin)
/// Converts from double precision to target precision.
FailureOr<TypedAttr> castComptimeFloatToRuntimeFloat(ComptimeFloatAttr srcAttr,
                                                     FloatType targetType,
                                                     ::mlir::Diagnostic *diag);

/// Cast comptime float to runtime integer (MLIR builtin, truncating).
FailureOr<TypedAttr> castComptimeFloatToRuntimeInt(ComptimeFloatAttr srcAttr,
                                                   IntegerType targetType,
                                                   ::mlir::Diagnostic *diag);

/// Cast runtime integer to comptime integer.
/// Converts fixed-width integer to arbitrary precision signed comptime_int,
/// preserving the value exactly.
FailureOr<TypedAttr> castRuntimeIntToComptimeInt(IntegerAttr srcAttr,
                                                 ::mlir::Diagnostic *diag);

/// Cast runtime float to comptime float.
/// Converts fixed-precision float to arbitrary precision comptime_float,
/// preserving the value exactly.
FailureOr<TypedAttr> castRuntimeFloatToComptimeFloat(FloatAttr srcAttr,
                                                     ::mlir::Diagnostic *diag);

/// Check if two types are compatible for casting.
/// Returns true if a cast between sourceType and targetType is valid.
bool areCastCompatible(Type sourceType, Type targetType);

//===----------------------------------------------------------------------===//
// Bitcast Operation Helpers
//===----------------------------------------------------------------------===//

/// Bitcast a float APFloat value to an APInt with the raw bit representation.
/// Handles custom float types (like CB16) that LLVM's APFloat::bitcastToAPInt()
/// doesn't support natively, by round-tripping through double.
APInt floatBitcastToAPInt(const APFloat &value, FloatType floatType);

/// Construct an APFloat from raw bits for a given float type.
/// Handles custom float types (like CB16) that LLVM's APFloat(sem, APInt)
/// constructor doesn't support natively, by round-tripping through double.
APFloat apIntBitcastToFloat(const APInt &bits, FloatType floatType);

//===----------------------------------------------------------------------===//
// InFlightDiagnostic convenience overloads
//===----------------------------------------------------------------------===//

inline FailureOr<BinaryOpResult> inferBinaryOp(BinaryOpKind opcode,
                                               TypeOrAttr lhs, TypeOrAttr rhs,
                                               InFlightDiagnostic *diag) {
  return inferBinaryOp(opcode, lhs, rhs,
                       diag ? diag->getUnderlyingDiagnostic() : nullptr);
}

inline FailureOr<Type> inferUnaryResultType(UnaryOpKind opcode,
                                            Type operandType,
                                            InFlightDiagnostic *diag) {
  return inferUnaryResultType(opcode, operandType,
                              diag ? diag->getUnderlyingDiagnostic() : nullptr);
}

inline FailureOr<TypedAttr> evaluateUnaryStrict(UnaryOpKind opcode,
                                                TypedAttr operand,
                                                InFlightDiagnostic *diag) {
  return evaluateUnaryStrict(opcode, operand,
                             diag ? diag->getUnderlyingDiagnostic() : nullptr);
}

inline FailureOr<TypedAttr> evaluateUnary(UnaryOpKind opcode, TypedAttr operand,
                                          InFlightDiagnostic *diag) {
  return evaluateUnary(opcode, operand,
                       diag ? diag->getUnderlyingDiagnostic() : nullptr);
}

inline FailureOr<TypedAttr> evaluateImplicitCast(TypedAttr srcAttr,
                                                 Type targetType,
                                                 InFlightDiagnostic *diag) {
  return evaluateImplicitCast(srcAttr, targetType,
                              diag ? diag->getUnderlyingDiagnostic() : nullptr);
}

inline FailureOr<TypedAttr> evaluateCast(TypedAttr srcAttr, Type targetType,
                                         InFlightDiagnostic *diag) {
  return evaluateCast(srcAttr, targetType,
                      diag ? diag->getUnderlyingDiagnostic() : nullptr);
}

} // namespace csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_CSL_IR_CSLFOLDERS_H
