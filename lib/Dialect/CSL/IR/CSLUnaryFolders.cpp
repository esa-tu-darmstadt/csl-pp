//===- CSLUnaryFolders.cpp - Unary Operation Folder Helpers -----*- C++ -*-===//
//
// Helper functions for constant folding of unary operations on both
// comptime (arbitrary precision) and runtime (fixed-width) types.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Runtime Integer Unary Operations
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::evaluateRuntimeIntUnary(UnaryOpKind opcode,
                                                        IntegerAttr operand,
                                                        Diagnostic *diag) {
  llvm::APInt value = operand.getValue();
  auto intType = mlir::cast<IntegerType>(operand.getType());
  bool isSigned = intType.isSigned();

  switch (opcode) {
  case UnaryOpKind::Neg: {
    if (isSigned) {
      // Check for overflow: negating INT_MIN overflows
      bool overflow = false;
      llvm::APInt result(value.getBitWidth(), 0);
      result = llvm::APInt(value.getBitWidth(), 0).ssub_ov(value, overflow);
      if (overflow) {
        if (diag)
          *diag << "negation overflow (negating minimum signed value)";
        return failure();
      }
      return (TypedAttr)IntegerAttr::get(intType, result);
    } else {
      // Unsigned negation is always defined (wraps around)
      return (TypedAttr)IntegerAttr::get(intType, -value);
    }
  }
  case UnaryOpKind::BitNot: {
    return (TypedAttr)IntegerAttr::get(intType, ~value);
  }
  case UnaryOpKind::LogicalNot: {
    if (diag)
      *diag << "logical not requires bool operand, got integer";
    return failure();
  }
  }
  llvm_unreachable("Unknown unary op kind");
}

//===----------------------------------------------------------------------===//
// Comptime Integer Unary Operations
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
mlir::csl::evaluateComptimeIntUnary(UnaryOpKind opcode, ComptimeIntAttr operand,
                                    Diagnostic *diag) {
  MLIRContext *ctx = operand.getContext();
  llvm::APInt value = operand.getValue();

  switch (opcode) {
  case UnaryOpKind::Neg: {
    // Comptime ints have arbitrary precision, so extend by 1 bit to avoid
    // overflow
    llvm::APInt extended = value.sext(value.getBitWidth() + 1);
    llvm::APInt result = -extended;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  }
  case UnaryOpKind::BitNot: {
    return (TypedAttr)ComptimeIntAttr::get(ctx, ~value);
  }
  case UnaryOpKind::LogicalNot: {
    if (diag)
      *diag << "logical not not supported on comptime_int";
    return failure();
  }
  }
  llvm_unreachable("Unknown unary op kind");
}

//===----------------------------------------------------------------------===//
// Boolean Unary Operations (MLIR builtin i1)
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::evaluateBoolUnary(UnaryOpKind opcode,
                                                  IntegerAttr operand,
                                                  Diagnostic *diag) {
  MLIRContext *ctx = operand.getContext();
  bool value = operand.getValue().getBoolValue();
  auto boolType = IntegerType::get(ctx, 1);

  switch (opcode) {
  case UnaryOpKind::LogicalNot:
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, !value));
  case UnaryOpKind::Neg:
  case UnaryOpKind::BitNot:
    if (diag)
      *diag << "unsupported unary operation on bool";
    return failure();
  }
  llvm_unreachable("Unknown unary op kind");
}

//===----------------------------------------------------------------------===//
// Runtime Float Unary Operations (MLIR builtins)
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::evaluateRuntimeFloatUnary(UnaryOpKind opcode,
                                                          FloatAttr operand,
                                                          Diagnostic *diag) {
  llvm::APFloat value = operand.getValue();
  Type floatType = operand.getType();

  switch (opcode) {
  case UnaryOpKind::Neg: {
    value.changeSign();
    return (TypedAttr)FloatAttr::get(floatType, value);
  }
  case UnaryOpKind::BitNot:
  case UnaryOpKind::LogicalNot:
    if (diag)
      *diag << "unsupported unary operation on float";
    return failure();
  }
  llvm_unreachable("Unknown unary op kind");
}

//===----------------------------------------------------------------------===//
// Comptime Float Unary Operations
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::evaluateComptimeFloatUnary(
    UnaryOpKind opcode, ComptimeFloatAttr operand, Diagnostic *diag) {
  MLIRContext *ctx = operand.getContext();
  llvm::APFloat value = operand.getValue();

  switch (opcode) {
  case UnaryOpKind::Neg: {
    value.changeSign();
    return (TypedAttr)ComptimeFloatAttr::get(ctx, value);
  }
  case UnaryOpKind::BitNot:
  case UnaryOpKind::LogicalNot:
    if (diag)
      *diag << "unsupported unary operation on comptime_float";
    return failure();
  }
  llvm_unreachable("Unknown unary op kind");
}

//===----------------------------------------------------------------------===//
// Type Inference Helper
//===----------------------------------------------------------------------===//

FailureOr<Type> mlir::csl::inferUnaryResultType(UnaryOpKind opcode,
                                                Type operandType,
                                                Diagnostic *diag) {
  switch (opcode) {
  case UnaryOpKind::Neg: {
    // Negation applies to numeric types
    if (mlir::isa<IntegerType>(operandType) ||
        mlir::isa<FloatType>(operandType) ||
        mlir::isa<ComptimeIntType>(operandType) ||
        mlir::isa<ComptimeFloatType>(operandType)) {
      return operandType;
    }
    if (diag)
      *diag << "negation requires numeric operand, got " << operandType;
    return failure();
  }
  case UnaryOpKind::BitNot: {
    // Bitwise not applies to integer types
    if (mlir::isa<IntegerType>(operandType) ||
        mlir::isa<ComptimeIntType>(operandType)) {
      return operandType;
    }
    if (diag)
      *diag << "bitwise not requires integer operand, got " << operandType;
    return failure();
  }
  case UnaryOpKind::LogicalNot: {
    // Logical not applies to bool (i1) and returns bool
    if (auto intType = mlir::dyn_cast<IntegerType>(operandType)) {
      if (intType.getWidth() == 1) {
        return operandType;
      }
    }
    if (diag)
      *diag << "logical not requires bool operand, got " << operandType;
    return failure();
  }
  }
  llvm_unreachable("Unknown unary op kind");
}

//===----------------------------------------------------------------------===//
// Unified Unary Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::evaluateUnaryStrict(UnaryOpKind opcode,
                                                    TypedAttr operand,
                                                    Diagnostic *diag) {
  // Dispatch to appropriate evaluator based on attribute type
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(operand)) {
    auto intType = mlir::dyn_cast<IntegerType>(intAttr.getType());
    if (intType && intType.getWidth() == 1) {
      return evaluateBoolUnary(opcode, intAttr, diag);
    } else {
      return evaluateRuntimeIntUnary(opcode, intAttr, diag);
    }
  }

  if (auto fpAttr = mlir::dyn_cast<FloatAttr>(operand)) {
    return evaluateRuntimeFloatUnary(opcode, fpAttr, diag);
  }

  if (auto comptimeIntAttr = mlir::dyn_cast<ComptimeIntAttr>(operand)) {
    return evaluateComptimeIntUnary(opcode, comptimeIntAttr, diag);
  }

  if (auto comptimeFloatAttr = mlir::dyn_cast<ComptimeFloatAttr>(operand)) {
    return evaluateComptimeFloatUnary(opcode, comptimeFloatAttr, diag);
  }

  if (diag)
    *diag << "unsupported operand type for unary operation";
  return failure();
}

FailureOr<TypedAttr> mlir::csl::evaluateUnary(UnaryOpKind opcode,
                                              TypedAttr operand,
                                              Diagnostic *diag) {
  // For unary operations, there's no implicit coercion needed - just
  // delegate to the strict evaluation.
  return evaluateUnaryStrict(opcode, operand, diag);
}
