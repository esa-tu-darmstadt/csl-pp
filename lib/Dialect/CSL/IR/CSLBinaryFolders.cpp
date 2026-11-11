//===- CSLBinaryFolders.cpp - Binary Operation Folder Helpers ---*- C++ -*-===//
//
// Helper functions for constant folding of binary operations on both
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
// Runtime Integer Binary Operations
//===----------------------------------------------------------------------===//

static FailureOr<TypedAttr> evaluateRuntimeIntBinary(BinaryOpKind opcode,
                                                     IntegerAttr lhsAttr,
                                                     IntegerAttr rhsAttr,
                                                     Diagnostic *diag) {
  MLIRContext *ctx = lhsAttr.getContext();
  llvm::APInt lhsValue = lhsAttr.getValue();
  llvm::APInt rhsValue = rhsAttr.getValue();

  auto lhsType = mlir::cast<IntegerType>(lhsAttr.getType());
  bool isSigned = lhsType.isSigned();

  llvm::APInt result(lhsValue.getBitWidth(), 0);
  bool overflow = false;

  switch (opcode) {
  case BinaryOpKind::Add: {
    result = isSigned ? lhsValue.sadd_ov(rhsValue, overflow)
                      : lhsValue.uadd_ov(rhsValue, overflow);
    if (overflow) {
      if (diag)
        *diag << "addition overflow";
      return failure();
    }
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::Sub: {
    result = isSigned ? lhsValue.ssub_ov(rhsValue, overflow)
                      : lhsValue.usub_ov(rhsValue, overflow);
    if (overflow) {
      if (diag)
        *diag << "subtraction overflow";
      return failure();
    }
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::Mul: {
    result = isSigned ? lhsValue.smul_ov(rhsValue, overflow)
                      : lhsValue.umul_ov(rhsValue, overflow);
    if (overflow) {
      if (diag)
        *diag << "multiplication overflow";
      return failure();
    }
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::Div: {
    if (rhsValue.isZero()) {
      if (diag)
        *diag << "division by zero";
      return failure();
    }
    result = isSigned ? lhsValue.sdiv_ov(rhsValue, overflow)
                      : lhsValue.udiv(rhsValue);
    if (overflow) {
      if (diag)
        *diag << "division overflow (INT_MIN / -1)";
      return failure();
    }
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::Mod: {
    if (rhsValue.isZero()) {
      if (diag)
        *diag << "modulo by zero";
      return failure();
    }
    result = isSigned ? lhsValue.srem(rhsValue) : lhsValue.urem(rhsValue);
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::BitAnd: {
    result = lhsValue & rhsValue;
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::BitOr: {
    result = lhsValue | rhsValue;
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::BitXor: {
    result = lhsValue ^ rhsValue;
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }
  case BinaryOpKind::Eq: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue == rhsValue));
  }
  case BinaryOpKind::Ne: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue != rhsValue));
  }
  case BinaryOpKind::Lt: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = isSigned ? lhsValue.slt(rhsValue) : lhsValue.ult(rhsValue);
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Le: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = isSigned ? lhsValue.sle(rhsValue) : lhsValue.ule(rhsValue);
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Gt: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = isSigned ? lhsValue.sgt(rhsValue) : lhsValue.ugt(rhsValue);
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Ge: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = isSigned ? lhsValue.sge(rhsValue) : lhsValue.uge(rhsValue);
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Shl:
  case BinaryOpKind::Shr:
    llvm_unreachable("shifts are handled by evaluateShift");
  case BinaryOpKind::LogicalAnd:
  case BinaryOpKind::LogicalOr:
    if (diag)
      *diag << "logical operations not supported on integers";
    return failure();
  }

  if (diag)
    *diag << "unsupported runtime integer operation";
  return failure();
}

//===----------------------------------------------------------------------===//
// Comptime Integer Binary Operations
//===----------------------------------------------------------------------===//

static FailureOr<TypedAttr> evaluateComptimeIntBinary(BinaryOpKind opcode,
                                                      ComptimeIntAttr lhsAttr,
                                                      ComptimeIntAttr rhsAttr,
                                                      Diagnostic *diag) {
  MLIRContext *ctx = lhsAttr.getContext();
  llvm::APInt lhsValue = lhsAttr.getValue();
  llvm::APInt rhsValue = rhsAttr.getValue();

  // Comptime ints are always signed and have arbitrary precision
  // Extend to same width if needed
  unsigned lhsWidth = lhsValue.getBitWidth();
  unsigned rhsWidth = rhsValue.getBitWidth();

  if (lhsWidth != rhsWidth) {
    unsigned maxWidth = std::max(lhsWidth, rhsWidth);
    if (lhsWidth < maxWidth)
      lhsValue = lhsValue.sext(maxWidth);
    if (rhsWidth < maxWidth)
      rhsValue = rhsValue.sext(maxWidth);
  }

  // For operations that can grow, extend to prevent overflow
  switch (opcode) {
  case BinaryOpKind::Add:
  case BinaryOpKind::Sub:
    lhsValue = lhsValue.sext(lhsValue.getBitWidth() + 1);
    rhsValue = rhsValue.sext(rhsValue.getBitWidth() + 1);
    break;
  case BinaryOpKind::Mul: {
    unsigned lhsSignificantBits = lhsValue.getSignificantBits();
    unsigned rhsSignificantBits = rhsValue.getSignificantBits();
    unsigned resultWidth = lhsSignificantBits + rhsSignificantBits;
    resultWidth = std::max(resultWidth, lhsValue.getBitWidth());
    lhsValue = lhsValue.sext(resultWidth);
    rhsValue = rhsValue.sext(resultWidth);
    break;
  }
  default:
    break;
  }

  // Perform the operation
  llvm::APInt result(1, 0);
  switch (opcode) {
  case BinaryOpKind::Add:
    result = lhsValue + rhsValue;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::Sub:
    result = lhsValue - rhsValue;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::Mul:
    result = lhsValue * rhsValue;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::Div:
    if (rhsValue.isZero()) {
      if (diag)
        *diag << "division by zero";
      return failure();
    }
    result = lhsValue.sdiv(rhsValue);
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::Mod:
    if (rhsValue.isZero()) {
      if (diag)
        *diag << "modulo by zero";
      return failure();
    }
    result = lhsValue.srem(rhsValue);
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::BitAnd:
    result = lhsValue & rhsValue;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::BitOr:
    result = lhsValue | rhsValue;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::BitXor:
    result = lhsValue ^ rhsValue;
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  case BinaryOpKind::Eq: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue == rhsValue));
  }
  case BinaryOpKind::Ne: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue != rhsValue));
  }
  case BinaryOpKind::Lt: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue.slt(rhsValue)));
  }
  case BinaryOpKind::Le: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue.sle(rhsValue)));
  }
  case BinaryOpKind::Gt: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue.sgt(rhsValue)));
  }
  case BinaryOpKind::Ge: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue.sge(rhsValue)));
  }
  case BinaryOpKind::Shl:
  case BinaryOpKind::Shr:
    llvm_unreachable("shifts are handled by evaluateShift");
  case BinaryOpKind::LogicalAnd:
  case BinaryOpKind::LogicalOr:
    if (diag)
      *diag << "logical operations not supported on comptime_int";
    return failure();
  }

  if (diag)
    *diag << "unsupported comptime integer operation";
  return failure();
}

//===----------------------------------------------------------------------===//
// Boolean Binary Operations (MLIR builtin i1)
//===----------------------------------------------------------------------===//

static FailureOr<TypedAttr> evaluateBoolBinary(BinaryOpKind opcode,
                                               IntegerAttr lhsAttr,
                                               IntegerAttr rhsAttr,
                                               Diagnostic *diag) {
  MLIRContext *ctx = lhsAttr.getContext();
  bool lhsValue = lhsAttr.getValue().getBoolValue();
  bool rhsValue = rhsAttr.getValue().getBoolValue();
  auto boolType = IntegerType::get(ctx, 1);

  switch (opcode) {
  case BinaryOpKind::LogicalAnd:
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue && rhsValue));
  case BinaryOpKind::LogicalOr:
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue || rhsValue));
  case BinaryOpKind::Eq:
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue == rhsValue));
  case BinaryOpKind::Ne:
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue != rhsValue));
  default:
    if (diag)
      *diag << "unsupported boolean operation";
    return failure();
  }
}

//===----------------------------------------------------------------------===//
// Runtime Float Binary Operations (MLIR builtins)
//===----------------------------------------------------------------------===//

static FailureOr<TypedAttr> evaluateRuntimeFloatBinary(BinaryOpKind opcode,
                                                       FloatAttr lhsAttr,
                                                       FloatAttr rhsAttr,
                                                       Diagnostic *diag) {
  MLIRContext *ctx = lhsAttr.getContext();
  llvm::APFloat lhsValue = lhsAttr.getValue();
  llvm::APFloat rhsValue = rhsAttr.getValue();

  // Get the float type from the attribute
  Type floatType = lhsAttr.getType();
  auto boolType = IntegerType::get(ctx, 1);

  switch (opcode) {
  case BinaryOpKind::Add: {
    llvm::APFloat result = lhsValue;
    result.add(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)FloatAttr::get(floatType, result);
  }
  case BinaryOpKind::Sub: {
    llvm::APFloat result = lhsValue;
    result.subtract(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)FloatAttr::get(floatType, result);
  }
  case BinaryOpKind::Mul: {
    llvm::APFloat result = lhsValue;
    result.multiply(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)FloatAttr::get(floatType, result);
  }
  case BinaryOpKind::Div: {
    llvm::APFloat result = lhsValue;
    result.divide(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)FloatAttr::get(floatType, result);
  }
  case BinaryOpKind::Eq: {
    bool cmpResult = lhsValue.compare(rhsValue) == llvm::APFloat::cmpEqual;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Ne: {
    bool cmpResult = lhsValue.compare(rhsValue) != llvm::APFloat::cmpEqual;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Lt: {
    bool cmpResult = lhsValue.compare(rhsValue) == llvm::APFloat::cmpLessThan;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Le: {
    llvm::APFloat::cmpResult cmp = lhsValue.compare(rhsValue);
    bool cmpResult =
        cmp == llvm::APFloat::cmpLessThan || cmp == llvm::APFloat::cmpEqual;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Gt: {
    bool cmpResult =
        lhsValue.compare(rhsValue) == llvm::APFloat::cmpGreaterThan;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Ge: {
    llvm::APFloat::cmpResult cmp = lhsValue.compare(rhsValue);
    bool cmpResult =
        cmp == llvm::APFloat::cmpGreaterThan || cmp == llvm::APFloat::cmpEqual;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  default: {
    if (diag)
      *diag << "unsupported runtime float operation";
    return failure();
  }
  }
}

//===----------------------------------------------------------------------===//
// Comptime Float Binary Operations
//===----------------------------------------------------------------------===//

static FailureOr<TypedAttr>
evaluateComptimeFloatBinary(BinaryOpKind opcode, ComptimeFloatAttr lhsAttr,
                            ComptimeFloatAttr rhsAttr, Diagnostic *diag) {
  MLIRContext *ctx = lhsAttr.getContext();
  llvm::APFloat lhsValue = lhsAttr.getValue();
  llvm::APFloat rhsValue = rhsAttr.getValue();

  llvm::APFloat result = lhsValue;
  switch (opcode) {
  case BinaryOpKind::Add:
    result.add(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)ComptimeFloatAttr::get(ctx, result);
  case BinaryOpKind::Sub:
    result.subtract(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)ComptimeFloatAttr::get(ctx, result);
  case BinaryOpKind::Mul:
    result.multiply(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)ComptimeFloatAttr::get(ctx, result);
  case BinaryOpKind::Div:
    result.divide(rhsValue, llvm::APFloat::rmNearestTiesToEven);
    return (TypedAttr)ComptimeFloatAttr::get(ctx, result);
  case BinaryOpKind::Eq: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = lhsValue.compare(rhsValue) == llvm::APFloat::cmpEqual;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Ne: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = lhsValue.compare(rhsValue) != llvm::APFloat::cmpEqual;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Lt: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = lhsValue.compare(rhsValue) == llvm::APFloat::cmpLessThan;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Le: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult =
        lhsValue.compare(rhsValue) != llvm::APFloat::cmpGreaterThan;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Gt: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult =
        lhsValue.compare(rhsValue) == llvm::APFloat::cmpGreaterThan;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  case BinaryOpKind::Ge: {
    auto boolType = IntegerType::get(ctx, 1);
    bool cmpResult = lhsValue.compare(rhsValue) != llvm::APFloat::cmpLessThan;
    return (TypedAttr)IntegerAttr::get(boolType, APInt(1, cmpResult));
  }
  default:
    if (diag)
      *diag << "operation not supported on comptime_float";
    return failure();
  }
}

//===----------------------------------------------------------------------===//
// Type Inference Helpers
//===----------------------------------------------------------------------===//

FailureOr<Type> mlir::csl::getCommonType(Type lhsType, Type rhsType) {
  // If types are identical, no coercion needed
  if (lhsType == rhsType)
    return lhsType;

  // Normalize: ensure comptime type is on lhs for simpler logic
  bool lhsIsComptime = isComptimeOnlyType(lhsType);
  bool rhsIsComptime = isComptimeOnlyType(rhsType);

  // Swap if rhs is comptime but lhs isn't (order doesn't matter for coercion)
  if (!lhsIsComptime && rhsIsComptime)
    std::swap(lhsType, rhsType);

  // Now if there's valid coercion, lhs is comptime and rhs is runtime
  // comptime_int can coerce to runtime int
  if (mlir::isa<ComptimeIntType>(lhsType) && isAnyRuntimeIntType(rhsType))
    return rhsType;

  // comptime_float can coerce to runtime float
  if (mlir::isa<ComptimeFloatType>(lhsType) && isAnyRuntimeFloatType(rhsType))
    return rhsType;

  return failure();
}

//===----------------------------------------------------------------------===//
// Comptime String Binary Operations
//===----------------------------------------------------------------------===//

static FailureOr<TypedAttr>
evaluateComptimeStringBinary(BinaryOpKind opcode, ComptimeStringAttr lhsAttr,
                             ComptimeStringAttr rhsAttr, Diagnostic *diag) {
  MLIRContext *ctx = lhsAttr.getContext();
  StringRef lhsValue = lhsAttr.getValue();
  StringRef rhsValue = rhsAttr.getValue();

  switch (opcode) {
  case BinaryOpKind::Add: {
    // Concatenate strings
    std::string result = lhsValue.str() + rhsValue.str();
    return (TypedAttr)ComptimeStringAttr::get(ctx, result);
  }
  case BinaryOpKind::Eq: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue == rhsValue));
  }
  case BinaryOpKind::Ne: {
    auto boolType = IntegerType::get(ctx, 1);
    return (TypedAttr)IntegerAttr::get(boolType,
                                       APInt(1, lhsValue != rhsValue));
  }
  default:
    if (diag)
      *diag << "operation not supported on comptime_string";
    return failure();
  }
}

//===----------------------------------------------------------------------===//
// Unified Binary Evaluation
//===----------------------------------------------------------------------===//

/// Evaluate a shift operation. LHS and RHS may have different types.
/// The shift amount is extracted from RHS regardless of its concrete attr type.
/// The result type matches the LHS type.
static FailureOr<TypedAttr> evaluateShift(BinaryOpKind opcode, TypedAttr lhs,
                                          TypedAttr rhs, Diagnostic *diag) {
  assert(opcode == BinaryOpKind::Shl || opcode == BinaryOpKind::Shr);

  // Extract shift amount from RHS, regardless of its attr type.
  llvm::APInt shiftAmountAP;
  if (auto rhsInt = mlir::dyn_cast<IntegerAttr>(rhs))
    shiftAmountAP = rhsInt.getValue();
  else if (auto rhsCI = mlir::dyn_cast<ComptimeIntAttr>(rhs))
    shiftAmountAP = rhsCI.getValue();
  else {
    if (diag)
      *diag << "shift amount must be an integer type";
    return failure();
  }

  if (shiftAmountAP.getActiveBits() > 32) {
    if (diag)
      *diag << "shift amount too large";
    return failure();
  }
  unsigned shiftAmount = shiftAmountAP.getLimitedValue();

  // Runtime integer LHS.
  if (auto lhsInt = mlir::dyn_cast<IntegerAttr>(lhs)) {
    auto lhsType = mlir::cast<IntegerType>(lhsInt.getType());
    bool isSigned = lhsType.isSigned();
    llvm::APInt lhsValue = lhsInt.getValue();

    if (shiftAmount >= lhsValue.getBitWidth()) {
      if (diag)
        *diag << "shift amount " << shiftAmount << " exceeds bit width "
              << lhsValue.getBitWidth();
      return failure();
    }

    if (opcode == BinaryOpKind::Shl) {
      bool overflow = false;
      llvm::APInt result = isSigned ? lhsValue.sshl_ov(shiftAmount, overflow)
                                    : lhsValue.ushl_ov(shiftAmount, overflow);
      if (overflow) {
        if (diag)
          *diag << "left shift overflow";
        return failure();
      }
      return (TypedAttr)IntegerAttr::get(lhsType, result);
    }
    llvm::APInt result =
        isSigned ? lhsValue.ashr(shiftAmount) : lhsValue.lshr(shiftAmount);
    return (TypedAttr)IntegerAttr::get(lhsType, result);
  }

  // Comptime integer LHS.
  if (auto lhsCI = mlir::dyn_cast<ComptimeIntAttr>(lhs)) {
    MLIRContext *ctx = lhs.getContext();
    llvm::APInt lhsValue = lhsCI.getValue();

    if (opcode == BinaryOpKind::Shl) {
      // Extend to prevent overflow.
      lhsValue = lhsValue.sext(lhsValue.getBitWidth() + shiftAmount);
      llvm::APInt result = lhsValue.shl(shiftAmount);
      return (TypedAttr)ComptimeIntAttr::get(ctx, result);
    }
    llvm::APInt result = lhsValue.ashr(shiftAmount);
    return (TypedAttr)ComptimeIntAttr::get(ctx, result);
  }

  if (diag)
    *diag << "shift operand must be an integer type";
  return failure();
}

static FailureOr<TypedAttr> evaluateBinaryStrict(BinaryOpKind opcode,
                                                 TypedAttr lhs, TypedAttr rhs,
                                                 Diagnostic *diag) {
  // Shifts allow mismatched LHS/RHS types and are handled separately.
  if (opcode == BinaryOpKind::Shl || opcode == BinaryOpKind::Shr)
    return evaluateShift(opcode, lhs, rhs, diag);

  // Non-shift: operand types must match exactly.
  if (lhs.getType() != rhs.getType()) {
    if (diag)
      *diag << "operand types must match, got " << lhs.getType() << " and "
            << rhs.getType();
    return failure();
  }

  // Dispatch to appropriate evaluator based on attribute type.
  if (auto lhsInt = mlir::dyn_cast<IntegerAttr>(lhs)) {
    auto rhsInt = mlir::dyn_cast<IntegerAttr>(rhs);
    assert(rhsInt && "rhs must be IntegerAttr if lhs is");

    auto intType = mlir::dyn_cast<IntegerType>(lhsInt.getType());
    if (intType && intType.getWidth() == 1)
      return evaluateBoolBinary(opcode, lhsInt, rhsInt, diag);
    return evaluateRuntimeIntBinary(opcode, lhsInt, rhsInt, diag);
  }

  if (auto lhsFP = mlir::dyn_cast<FloatAttr>(lhs)) {
    auto rhsFP = mlir::dyn_cast<FloatAttr>(rhs);
    assert(rhsFP && "rhs must be FloatAttr if lhs is");
    return evaluateRuntimeFloatBinary(opcode, lhsFP, rhsFP, diag);
  }

  if (auto lhsComptimeInt = mlir::dyn_cast<ComptimeIntAttr>(lhs)) {
    auto rhsComptimeInt = mlir::dyn_cast<ComptimeIntAttr>(rhs);
    assert(rhsComptimeInt && "rhs must be ComptimeIntAttr if lhs is");
    return evaluateComptimeIntBinary(opcode, lhsComptimeInt, rhsComptimeInt,
                                     diag);
  }

  if (auto lhsComptimeFloat = mlir::dyn_cast<ComptimeFloatAttr>(lhs)) {
    auto rhsComptimeFloat = mlir::dyn_cast<ComptimeFloatAttr>(rhs);
    assert(rhsComptimeFloat && "rhs must be ComptimeFloatAttr if lhs is");
    return evaluateComptimeFloatBinary(opcode, lhsComptimeFloat,
                                       rhsComptimeFloat, diag);
  }

  if (auto lhsComptimeString = mlir::dyn_cast<ComptimeStringAttr>(lhs)) {
    auto rhsComptimeString = mlir::dyn_cast<ComptimeStringAttr>(rhs);
    assert(rhsComptimeString && "rhs must be ComptimeStringAttr if lhs is");
    return evaluateComptimeStringBinary(opcode, lhsComptimeString,
                                        rhsComptimeString, diag);
  }

  // Direction equality comparison
  if (auto lhsDir = mlir::dyn_cast<DirectionAttr>(lhs)) {
    auto rhsDir = mlir::dyn_cast<DirectionAttr>(rhs);
    assert(rhsDir && "rhs must be DirectionAttr if lhs is");
    MLIRContext *ctx = lhs.getContext();
    auto boolType = IntegerType::get(ctx, 1);
    switch (opcode) {
    case BinaryOpKind::Eq:
      return (TypedAttr)IntegerAttr::get(
          boolType, APInt(1, lhsDir.getValue() == rhsDir.getValue()));
    case BinaryOpKind::Ne:
      return (TypedAttr)IntegerAttr::get(
          boolType, APInt(1, lhsDir.getValue() != rhsDir.getValue()));
    default:
      if (diag)
        *diag << "only == and != are supported for direction values";
      return failure();
    }
  }

  // Enum equality comparison
  if (auto lhsEnum = mlir::dyn_cast<EnumValueAttr>(lhs)) {
    auto rhsEnum = mlir::dyn_cast<EnumValueAttr>(rhs);
    assert(rhsEnum && "rhs must be EnumValueAttr if lhs is");
    MLIRContext *ctx = lhs.getContext();
    auto boolType = IntegerType::get(ctx, 1);
    switch (opcode) {
    case BinaryOpKind::Eq:
      return (TypedAttr)IntegerAttr::get(
          boolType, APInt(1, lhsEnum.getIntValue() == rhsEnum.getIntValue()));
    case BinaryOpKind::Ne:
      return (TypedAttr)IntegerAttr::get(
          boolType, APInt(1, lhsEnum.getIntValue() != rhsEnum.getIntValue()));
    default:
      if (diag)
        *diag << "only == and != are supported for enum values";
      return failure();
    }
  }

  // Type equality comparison
  if (auto lhsTypeAttr = mlir::dyn_cast<csl::TypeAttr>(lhs)) {
    auto rhsTypeAttr = mlir::dyn_cast<csl::TypeAttr>(rhs);
    assert(rhsTypeAttr && "rhs must be TypeAttr if lhs is");
    MLIRContext *ctx = lhs.getContext();
    auto boolType = IntegerType::get(ctx, 1);
    switch (opcode) {
    case BinaryOpKind::Eq:
      return (TypedAttr)IntegerAttr::get(
          boolType, APInt(1, lhsTypeAttr.getValue() == rhsTypeAttr.getValue()));
    case BinaryOpKind::Ne:
      return (TypedAttr)IntegerAttr::get(
          boolType, APInt(1, lhsTypeAttr.getValue() != rhsTypeAttr.getValue()));
    default:
      if (diag)
        *diag << "only == and != are supported for type values";
      return failure();
    }
  }

  if (diag)
    *diag << "unsupported operand types for binary operation";
  return failure();
}

FailureOr<BinaryOpResult> mlir::csl::inferBinaryOp(BinaryOpKind opcode,
                                                   TypeOrAttr lhs,
                                                   TypeOrAttr rhs,
                                                   Diagnostic *diag) {
  Type lhsType = getType(lhs);
  Type rhsType = getType(rhs);

  bool isShift = (opcode == BinaryOpKind::Shl || opcode == BinaryOpKind::Shr);

  Type lhsTarget, rhsTarget;
  Type resultType;

  if (isShift) {
    // Shift operations: both operands must be integer types.
    bool lhsIsInt =
        isAnyRuntimeIntType(lhsType) || mlir::isa<ComptimeIntType>(lhsType);
    bool rhsIsInt =
        isAnyRuntimeIntType(rhsType) || mlir::isa<ComptimeIntType>(rhsType);
    if (!lhsIsInt || !rhsIsInt) {
      if (diag)
        *diag << "shift operands must be integer types, got "
              << prettyPrintType(lhsType) << " and "
              << prettyPrintType(rhsType);
      return failure();
    }

    // No coercion for shifts — LHS and RHS types are independent.
    // When LHS is a runtime int and RHS is comptime_int, cast the shift amount
    // to the unsigned version of the LHS type.
    lhsTarget = lhsType;
    if (mlir::isa<ComptimeIntType>(rhsType) && isAnyRuntimeIntType(lhsType)) {
      auto lhsIntType = mlir::cast<IntegerType>(lhsType);
      rhsTarget = IntegerType::get(lhsType.getContext(), lhsIntType.getWidth(),
                                   IntegerType::Unsigned);
    } else {
      rhsTarget = rhsType;
    }
    resultType = lhsType;

    // Validate shift amount constraints when the RHS is not comptime-known.
    if (!isComptimeKnown(rhs)) {
      // comptime_int can only be shifted by a comptime-known amount.
      if (mlir::isa<ComptimeIntType>(lhsType)) {
        if (diag)
          *diag << "shifting comptime_int requires a comptime-known shift "
                   "amount";
        return failure();
      }
      // A runtime shift amount must be an unsigned integer type.
      if (auto rhsIntType = mlir::dyn_cast<IntegerType>(rhsType)) {
        if (!rhsIntType.isUnsigned()) {
          if (diag)
            *diag << "shift amount must be an unsigned integer type, got "
                  << prettyPrintType(rhsType);
          return failure();
        }
      }
    }
  } else {
    // Non-shift: find common type with implicit coercion.
    auto commonType = getCommonType(lhsType, rhsType);
    if (failed(commonType)) {
      if (diag)
        *diag << "incompatible types for binary operation: "
              << prettyPrintType(lhsType) << " and "
              << prettyPrintType(rhsType);
      return failure();
    }

    lhsTarget = *commonType;
    rhsTarget = *commonType;

    // Comparisons and logical ops return bool (i1).
    switch (opcode) {
    case BinaryOpKind::Eq:
    case BinaryOpKind::Ne:
    case BinaryOpKind::Lt:
    case BinaryOpKind::Le:
    case BinaryOpKind::Gt:
    case BinaryOpKind::Ge:
    case BinaryOpKind::LogicalAnd:
    case BinaryOpKind::LogicalOr:
      resultType = IntegerType::get(lhsType.getContext(), 1);
      break;
    default:
      resultType = *commonType;
      break;
    }
  }

  // If both operands are comptime-known, evaluate the operation.
  if (isComptimeKnown(lhs) && isComptimeKnown(rhs)) {
    TypedAttr lhsAttr = std::get<TypedAttr>(lhs);
    TypedAttr rhsAttr = std::get<TypedAttr>(rhs);

    if (isShift) {
      // Validate non-negative shift amount.
      bool isNegative = false;
      if (auto ci = mlir::dyn_cast<ComptimeIntAttr>(rhsAttr))
        isNegative = ci.getValue().isNegative();
      else if (auto ia = mlir::dyn_cast<IntegerAttr>(rhsAttr)) {
        if (mlir::cast<IntegerType>(ia.getType()).isSigned())
          isNegative = ia.getValue().isNegative();
      }
      if (isNegative) {
        if (diag)
          *diag << "shift amount must be non-negative";
        return failure();
      }
    } else {
      // Non-shift: cast both operands to the common type.
      if (lhsAttr.getType() != lhsTarget) {
        auto castResult = evaluateCast(lhsAttr, lhsTarget, diag);
        if (failed(castResult))
          return failure();
        lhsAttr = *castResult;
      }
      if (rhsAttr.getType() != rhsTarget) {
        auto castResult = evaluateCast(rhsAttr, rhsTarget, diag);
        if (failed(castResult))
          return failure();
        rhsAttr = *castResult;
      }
    }

    auto evalResult = evaluateBinaryStrict(opcode, lhsAttr, rhsAttr, diag);
    if (failed(evalResult))
      return failure();

    return BinaryOpResult{lhsTarget, rhsTarget, TypeOrAttr(*evalResult)};
  }

  return BinaryOpResult{lhsTarget, rhsTarget, TypeOrAttr(resultType)};
}
