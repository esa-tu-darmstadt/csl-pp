//===- CSLCastFolders.cpp - Cast Operation Folder Helpers -------*- C++ -*-===//
//
// Helper functions for constant folding of cast operations on both
// comptime (arbitrary precision) and runtime (fixed-width) types.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include <sstream>

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Implicit Cast Validation
//===----------------------------------------------------------------------===//

bool mlir::csl::isImplicitCastAllowed(Type sourceType, Type targetType,
                                      bool comptimeKnown) {
  // Same type is always valid (no-op)
  if (sourceType == targetType)
    return true;

  // Comptime-known integer to integer of different width (allowed when
  // comptimeKnown) - the actual value check happens in evaluateCast
  if (comptimeKnown && mlir::isa<IntegerType>(sourceType) &&
      mlir::isa<IntegerType>(targetType))
    return true;

  // Comptime int to runtime int (implicit coercion)
  if (mlir::isa<ComptimeIntType>(sourceType) &&
      mlir::isa<IntegerType>(targetType))
    return true;

  // Runtime int to comptime int (implicit coercion - widening)
  if (mlir::isa<IntegerType>(sourceType) &&
      mlir::isa<ComptimeIntType>(targetType))
    return true;

  // Comptime float to runtime float (implicit coercion)
  if (mlir::isa<ComptimeFloatType>(sourceType) &&
      mlir::isa<FloatType>(targetType))
    return true;

  // Runtime float to comptime float (implicit coercion - widening)
  if (mlir::isa<FloatType>(sourceType) &&
      mlir::isa<ComptimeFloatType>(targetType))
    return true;

  // Anonymous struct or tuple to a struct or tuple with compatible fields
  // (implicit coercion). Fields of a tuple match by position, fields of a
  // struct by name; a struct with names does not coerce to a tuple.
  if (auto srcStructType = mlir::dyn_cast<StructType>(sourceType);
      srcStructType && srcStructType.isAnonymous()) {
    if (auto tgtStructType = mlir::dyn_cast<StructType>(targetType)) {
      // Allow implicit cast if all field types can be implicitly casted
      if (srcStructType.getNumFields() != tgtStructType.getNumFields())
        return false;
      if (tgtStructType.isTuple() && !srcStructType.isTuple())
        return false;

      for (size_t i = 0; i < srcStructType.getNumFields(); ++i) {
        size_t sourceIndex = i;
        if (!srcStructType.isTuple()) {
          FailureOr<StringRef> targetName = tgtStructType.getFieldName(i);
          if (failed(targetName))
            return false;
          FailureOr<size_t> matchingIndex =
              srcStructType.getFieldIndex(*targetName);
          if (failed(matchingIndex))
            return false;
          sourceIndex = *matchingIndex;
        }

        if (!isImplicitCastAllowed(srcStructType.getFieldType(sourceIndex),
                                   tgtStructType.getFieldType(i),
                                   comptimeKnown)) {
          return false;
        }
      }
      return true;
    }
  }

  // Coercion from non-const to const pointer is allowed
  if (auto srcPtrType = mlir::dyn_cast<PointerType>(sourceType)) {
    if (auto tgtPtrType = mlir::dyn_cast<PointerType>(targetType)) {
      if (srcPtrType.getPointeeType() == tgtPtrType.getPointeeType() &&
          !srcPtrType.getIsConst() && tgtPtrType.getIsConst())
        return true;
    }
  }

  // Coercion from function reference to const function pointer is allowed
  if (mlir::isa<FunctionType>(sourceType)) {
    if (auto tgtPtrType = mlir::dyn_cast<PointerType>(targetType)) {
      if (auto funcType =
              mlir::dyn_cast<FunctionType>(tgtPtrType.getPointeeType())) {
        if (tgtPtrType.getIsConst() && sourceType == funcType)
          return true;
      }
    }
  }

  // Coercion from pointer to array to many-pointer is allowed
  if (auto srcPtrType = mlir::dyn_cast<PointerType>(sourceType);
      srcPtrType && isa<ArrayType>(srcPtrType.getPointeeType())) {
    if (auto tgtManyPtrType = mlir::dyn_cast<PointerType>(targetType);
        tgtManyPtrType && tgtManyPtrType.getIsManyPtr()) {
      if (cast<ArrayType>(srcPtrType.getPointeeType()).getElementType() ==
              tgtManyPtrType.getPointeeType() &&
          (!srcPtrType.getIsConst() || tgtManyPtrType.getIsConst()))
        return true;
    }
  }

  // All other casts are explicit only
  return false;
}

FailureOr<TypedAttr> mlir::csl::evaluateImplicitCast(TypedAttr srcAttr,
                                                     Type targetType,
                                                     Diagnostic *diag) {
  // Check if this cast is allowed as an implicit cast.
  // Pass comptimeKnown=true since we have a TypedAttr, meaning the value is
  // known at compile time.
  if (!isImplicitCastAllowed(srcAttr.getType(), targetType,
                             /*comptimeKnown=*/true)) {
    if (diag)
      *diag << "implicit cast from '" << prettyPrintType(srcAttr.getType())
            << "' to '" << prettyPrintType(targetType) << "' is not allowed";
    return failure();
  }

  // Perform the actual cast using the general cast evaluator
  return evaluateCast(srcAttr, targetType, diag);
}

//===----------------------------------------------------------------------===//
// Unified Cast Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::evaluateCast(TypedAttr srcAttr, Type targetType,
                                             Diagnostic *diag) {
  // No cast needed - same type
  if (srcAttr.getType() == targetType)
    return srcAttr;

  // Dispatch based on source attribute type and target type (MLIR builtins)

  // Runtime Int (IntegerAttr) -> Runtime Int/Float/ComptimeInt
  if (auto srcInt = mlir::dyn_cast<IntegerAttr>(srcAttr)) {
    if (auto targetInt = mlir::dyn_cast<IntegerType>(targetType))
      return castRuntimeIntToRuntimeInt(srcInt, targetInt, diag);
    if (auto targetFloat = mlir::dyn_cast<FloatType>(targetType))
      return castRuntimeIntToRuntimeFloat(srcInt, targetFloat, diag);
    if (mlir::isa<ComptimeIntType>(targetType))
      return castRuntimeIntToComptimeInt(srcInt, diag);
  }

  // Runtime Float (FloatAttr) -> Runtime Int/Float/ComptimeFloat
  if (auto srcFloat = mlir::dyn_cast<FloatAttr>(srcAttr)) {
    if (auto targetInt = mlir::dyn_cast<IntegerType>(targetType))
      return castRuntimeFloatToRuntimeInt(srcFloat, targetInt, diag);
    if (auto targetFloat = mlir::dyn_cast<FloatType>(targetType))
      return castRuntimeFloatToRuntimeFloat(srcFloat, targetFloat, diag);
    if (mlir::isa<ComptimeFloatType>(targetType))
      return castRuntimeFloatToComptimeFloat(srcFloat, diag);
  }

  // Comptime Int -> Runtime Int/Float
  if (auto srcComptimeInt = mlir::dyn_cast<ComptimeIntAttr>(srcAttr)) {
    if (auto targetInt = mlir::dyn_cast<IntegerType>(targetType))
      return castComptimeIntToRuntimeInt(srcComptimeInt, targetInt, diag);
    if (auto targetFloat = mlir::dyn_cast<FloatType>(targetType))
      return castComptimeIntToRuntimeFloat(srcComptimeInt, targetFloat, diag);
  }

  // Comptime Float -> Runtime Float/Int
  if (auto srcComptimeFloat = mlir::dyn_cast<ComptimeFloatAttr>(srcAttr)) {
    if (auto targetFloat = mlir::dyn_cast<FloatType>(targetType))
      return castComptimeFloatToRuntimeFloat(srcComptimeFloat, targetFloat,
                                             diag);
    if (auto targetInt = mlir::dyn_cast<IntegerType>(targetType))
      return castComptimeFloatToRuntimeInt(srcComptimeFloat, targetInt, diag);
  }

  // Anonymous struct or tuple -> struct or tuple with compatible fields
  if (auto structValueAttr = mlir::dyn_cast<StructValueAttr>(srcAttr);
      structValueAttr && structValueAttr.isAnonymous()) {
    if (auto targetStructType = mlir::dyn_cast<StructType>(targetType);
        targetStructType &&
        targetStructType.getNumFields() == structValueAttr.getNumFields() &&
        (structValueAttr.isTuple() || !targetStructType.isTuple())) {
      // Cast each field to the target field type
      SmallVector<TypedAttr> castedFieldValues;
      for (size_t i = 0; i < targetStructType.getNumFields(); ++i) {
        size_t sourceIndex = i;
        if (!structValueAttr.isTuple()) {
          FailureOr<StringRef> targetName = targetStructType.getFieldName(i);
          if (failed(targetName))
            return failure();
          FailureOr<size_t> matchingIndex =
              structValueAttr.getFieldIndex(*targetName);
          if (failed(matchingIndex))
            return failure();
          sourceIndex = *matchingIndex;
        }

        TypedAttr fieldValue = structValueAttr.getFieldValues()[sourceIndex];
        Type targetFieldType = targetStructType.getFieldType(i);

        FailureOr<TypedAttr> castedFieldValue =
            evaluateCast(fieldValue, targetFieldType, diag);
        if (failed(castedFieldValue))
          return failure();

        castedFieldValues.push_back(*castedFieldValue);
      }

      return (TypedAttr)StructValueAttr::get(
          structValueAttr.getContext(), castedFieldValues,
          /*def_location=*/targetStructType.getDefLocation(),
          targetStructType.getFieldNames());
    }
  }

  // Enum value -> Integer (underlying type or compatible)
  if (auto enumValueAttr = mlir::dyn_cast<EnumValueAttr>(srcAttr)) {
    if (auto targetInt = mlir::dyn_cast<IntegerType>(targetType)) {
      // Get the underlying integer value
      int64_t intValue = enumValueAttr.getIntValue();
      IntegerType underlyingType =
          enumValueAttr.getEnumType().getUnderlyingType();
      APInt underlyingValue(underlyingType.getWidth(), intValue,
                            underlyingType.isSigned());

      // If target matches underlying type, just return IntegerAttr
      if (targetInt == underlyingType) {
        return cast<TypedAttr>(IntegerAttr::get(targetInt, underlyingValue));
      }

      // Otherwise, cast from underlying type to target type
      auto underlyingAttr = IntegerAttr::get(underlyingType, underlyingValue);
      return castRuntimeIntToRuntimeInt(underlyingAttr, targetInt, diag);
    }
  }

  // Integer -> Enum value
  if (auto targetEnumType = mlir::dyn_cast<EnumType>(targetType)) {
    int64_t intValue = 0;

    // Get the integer value from source
    if (auto srcInt = mlir::dyn_cast<IntegerAttr>(srcAttr)) {
      intValue = srcInt.getValue().getSExtValue();
    } else if (auto srcComptimeInt = mlir::dyn_cast<ComptimeIntAttr>(srcAttr)) {
      intValue = srcComptimeInt.getValue().getSExtValue();
    } else {
      if (diag)
        *diag << "cannot cast " << srcAttr.getType() << " to enum type";
      return failure();
    }

    // Look up the element name for this value
    std::optional<StringRef> elementName =
        targetEnumType.getElementName(intValue);
    if (!elementName) {
      if (diag)
        *diag << "integer value " << intValue
              << " does not correspond to a valid enum element";
      return failure();
    }

    return cast<TypedAttr>(EnumValueAttr::get(
        srcAttr.getContext(), targetEnumType,
        StringAttr::get(srcAttr.getContext(), *elementName), intValue));
  }

  // Function reference can coerce to const function pointer
  if (auto funcRefAttr = mlir::dyn_cast<FuncRefAttr>(srcAttr)) {
    if (auto targetPtrType = mlir::dyn_cast<PointerType>(targetType)) {
      if (auto funcType =
              mlir::dyn_cast<FunctionType>(targetPtrType.getPointeeType())) {
        // Check that function signature matches
        if (funcRefAttr.getType() != funcType) {
          if (diag)
            *diag << "referenced function type does not match target pointer "
                     "function type";
          return failure();
        }

        if (!targetPtrType.getIsConst()) {
          if (diag)
            *diag << "function reference can only coerce to const function "
                     "pointer";
          return failure();
        }
        return cast<TypedAttr>(
            FuncAddrAttr::get(funcRefAttr.getContext(), funcRefAttr));
      }
    }
  }

  // Fold pointer coercions on GlobalAddrAttr (constness, *[N]T → [*]T).
  // Only implicit casts are safe to fold — explicit @ptrcast may reinterpret
  // the element type and must remain as a runtime operation.
  if (auto globalPtrAttr = mlir::dyn_cast<GlobalAddrAttr>(srcAttr))
    if (auto targetPtrType = dyn_cast<PointerType>(targetType))
      if (isImplicitCastAllowed(srcAttr.getType(), targetType))
        return cast<TypedAttr>(GlobalAddrAttr::get(
            srcAttr.getContext(), targetPtrType, globalPtrAttr.getGlobalRef()));

  // Unsupported cast combination
  if (diag)
    *diag << "unsupported cast from " << prettyPrintType(srcAttr.getType())
          << " to " << prettyPrintType(targetType);
  return failure();
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::checkIntFitsInType(const llvm::APInt &value,
                                            bool valueIsSigned,
                                            IntegerType targetType,
                                            Diagnostic *diag) {
  unsigned targetWidth = targetType.getWidth();
  bool targetIsSigned = targetType.isSigned();

  if (valueIsSigned) {
    if (targetIsSigned) {
      // signed → signed
      if (!value.isSignedIntN(targetWidth)) {
        if (diag)
          *diag << "value " << value.getSExtValue() << " overflows "
                << targetWidth << "-bit signed integer";
        return failure();
      }
    } else {
      // signed → unsigned: must be non-negative
      if (value.isNegative()) {
        if (diag)
          *diag << "cannot cast negative value to unsigned integer";
        return failure();
      }
      if (!value.isIntN(targetWidth)) {
        if (diag)
          *diag << "value " << value.getZExtValue() << " overflows "
                << targetWidth << "-bit unsigned integer";
        return failure();
      }
    }
  } else {
    if (targetIsSigned) {
      // unsigned → signed: must fit in positive range
      llvm::APInt maxSigned = llvm::APInt::getSignedMaxValue(targetWidth);
      maxSigned = maxSigned.zextOrTrunc(value.getBitWidth());
      if (value.ugt(maxSigned)) {
        if (diag)
          *diag << "value " << value.getZExtValue() << " overflows "
                << targetWidth << "-bit signed integer";
        return failure();
      }
    } else {
      // unsigned → unsigned
      if (!value.isIntN(targetWidth)) {
        if (diag)
          *diag << "value " << value.getZExtValue() << " overflows "
                << targetWidth << "-bit unsigned integer";
        return failure();
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Runtime Int to Runtime Int
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castRuntimeIntToRuntimeInt(
    IntegerAttr srcAttr, IntegerType targetType, Diagnostic *diag) {
  llvm::APInt srcValue = srcAttr.getValue();
  auto srcType = mlir::cast<IntegerType>(srcAttr.getType());

  bool srcIsSigned = srcType.isSigned();
  unsigned targetWidth = targetType.getWidth();

  // Check for overflow
  if (failed(checkIntFitsInType(srcValue, srcIsSigned, targetType, diag)))
    return failure();

  // Resize to target width
  llvm::APInt result = srcIsSigned ? srcValue.sextOrTrunc(targetWidth)
                                   : srcValue.zextOrTrunc(targetWidth);

  return (TypedAttr)IntegerAttr::get(targetType, result);
}

//===----------------------------------------------------------------------===//
// Runtime Int to Runtime Float
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castRuntimeIntToRuntimeFloat(
    IntegerAttr srcAttr, FloatType targetType, Diagnostic *diag) {
  llvm::APInt srcValue = srcAttr.getValue();
  auto srcType = mlir::cast<IntegerType>(srcAttr.getType());
  bool srcIsSigned = srcType.isSigned();

  // Convert int to float using target type's semantics
  const llvm::fltSemantics &semantics = targetType.getFloatSemantics();
  llvm::APFloat result(semantics);

  if (srcIsSigned) {
    result.convertFromAPInt(srcValue, true, llvm::APFloat::rmNearestTiesToEven);
  } else {
    result.convertFromAPInt(srcValue, false,
                            llvm::APFloat::rmNearestTiesToEven);
  }

  return (TypedAttr)FloatAttr::get(targetType, result);
}

//===----------------------------------------------------------------------===//
// Runtime Float to Runtime Int
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castRuntimeFloatToRuntimeInt(
    FloatAttr srcAttr, IntegerType targetType, Diagnostic *diag) {
  llvm::APFloat srcValue = srcAttr.getValue();
  unsigned targetWidth = targetType.getWidth();
  bool targetIsSigned = targetType.isSigned();

  // Convert float to int
  llvm::APSInt result(targetWidth, !targetIsSigned);
  bool isExact = false;
  srcValue.convertToInteger(result, llvm::APFloat::rmTowardZero, &isExact);

  // Convert APSInt to APInt for IntegerAttr
  llvm::APInt resultInt = llvm::APInt(result);
  return (TypedAttr)IntegerAttr::get(targetType, resultInt);
}

//===----------------------------------------------------------------------===//
// Runtime Float to Runtime Float
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castRuntimeFloatToRuntimeFloat(
    FloatAttr srcAttr, FloatType targetType, Diagnostic *diag) {
  llvm::APFloat srcValue = srcAttr.getValue();

  // Get target semantics
  const llvm::fltSemantics &targetSemantics = targetType.getFloatSemantics();

  // Convert float to target precision
  bool losesInfo = false;
  llvm::APFloat result = srcValue;
  result.convert(targetSemantics, llvm::APFloat::rmNearestTiesToEven,
                 &losesInfo);

  return (TypedAttr)FloatAttr::get(targetType, result);
}

//===----------------------------------------------------------------------===//
// Comptime Int to Runtime Int
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castComptimeIntToRuntimeInt(
    ComptimeIntAttr srcAttr, IntegerType targetType, Diagnostic *diag) {
  llvm::APInt value = srcAttr.getValue();

  unsigned width = targetType.getWidth();
  bool isSigned = targetType.isSigned();

  // Check if value fits (comptime_int is always signed)
  if (isSigned) {
    if (!value.isSignedIntN(width)) {
      if (diag) {
        std::string valueStr;
        llvm::raw_string_ostream(valueStr) << value;
        *diag << "comptime_int value " << valueStr << " does not fit in "
              << prettyPrintType(targetType);
      }
      return failure();
    }
    value = value.sextOrTrunc(width);
  } else {
    if (!value.isIntN(width)) {
      if (diag) {
        std::string valueStr;
        llvm::raw_string_ostream(valueStr) << value;
        *diag << "comptime_int value " << valueStr << " does not fit in "
              << prettyPrintType(targetType);
      }
      return failure();
    }
    value = value.zextOrTrunc(width);
  }

  return (TypedAttr)IntegerAttr::get(targetType, value);
}

//===----------------------------------------------------------------------===//
// Comptime Int to Runtime Float
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castComptimeIntToRuntimeFloat(
    ComptimeIntAttr srcAttr, FloatType targetType, Diagnostic *diag) {
  llvm::APInt value = srcAttr.getValue();

  const llvm::fltSemantics &semantics = targetType.getFloatSemantics();

  llvm::APFloat result(semantics);
  result.convertFromAPInt(value, /*isSigned=*/true,
                          llvm::APFloat::rmNearestTiesToEven);
  return (TypedAttr)FloatAttr::get(targetType, result);
}

//===----------------------------------------------------------------------===//
// Comptime Float to Runtime Float
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castComptimeFloatToRuntimeFloat(
    ComptimeFloatAttr srcAttr, FloatType targetType, Diagnostic *diag) {
  llvm::APFloat value = srcAttr.getValue();

  const llvm::fltSemantics &semantics = targetType.getFloatSemantics();

  bool losesInfo;
  llvm::APFloat::opStatus status =
      value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);

  if (status != llvm::APFloat::opOK && status != llvm::APFloat::opInexact) {
    if (diag)
      *diag << "cannot convert comptime_float to runtime float";
    return failure();
  }

  return (TypedAttr)FloatAttr::get(targetType, value);
}

//===----------------------------------------------------------------------===//
// Comptime Float to Runtime Int
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castComptimeFloatToRuntimeInt(
    ComptimeFloatAttr srcAttr, IntegerType targetType, Diagnostic *diag) {
  llvm::APFloat value = srcAttr.getValue();

  bool isExact;
  llvm::APSInt intValue(targetType.getWidth(), !targetType.isSigned());
  llvm::APFloat::opStatus status =
      value.convertToInteger(intValue, llvm::APFloat::rmTowardZero, &isExact);

  if (status != llvm::APFloat::opOK && status != llvm::APFloat::opInexact) {
    if (diag)
      *diag << "cannot convert comptime_float to integer";
    return failure();
  }

  return (TypedAttr)IntegerAttr::get(targetType, llvm::APInt(intValue));
}

//===----------------------------------------------------------------------===//
// Runtime Int to Comptime Int
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> mlir::csl::castRuntimeIntToComptimeInt(IntegerAttr srcAttr,
                                                            Diagnostic *diag) {
  APInt value = srcAttr.getValue();
  auto intType = mlir::cast<IntegerType>(srcAttr.getType());
  bool isSigned = intType.isSigned();

  // comptime_int is always signed, so we need to sign-extend unsigned values
  // to preserve their numeric value when interpreted as signed
  if (!isSigned && value.isNegative()) {
    // The unsigned value has the high bit set, so we need to extend it
    // to preserve the unsigned interpretation
    value = value.zext(value.getBitWidth() + 1);
  }

  return (TypedAttr)ComptimeIntAttr::get(srcAttr.getContext(), value);
}

//===----------------------------------------------------------------------===//
// Runtime Float to Comptime Float
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
mlir::csl::castRuntimeFloatToComptimeFloat(FloatAttr srcAttr,
                                           Diagnostic *diag) {
  // ComptimeFloatAttr stores as double precision, convert from source
  // precision
  APFloat value = srcAttr.getValue();
  bool losesInfo;
  value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                &losesInfo);

  return (TypedAttr)ComptimeFloatAttr::get(srcAttr.getContext(), value);
}

//===----------------------------------------------------------------------===//
// Cast Compatibility Check
//===----------------------------------------------------------------------===//

bool mlir::csl::areCastCompatible(Type sourceType, Type targetType) {
  if (isImplicitCastAllowed(sourceType, targetType, false))
    return true;

  // Same type is always valid (no-op)
  if (sourceType == targetType)
    return true;

  // Runtime integer to runtime integer
  if (mlir::isa<IntegerType>(sourceType) && mlir::isa<IntegerType>(targetType))
    return true;

  // Runtime float to runtime float
  if (mlir::isa<FloatType>(sourceType) && mlir::isa<FloatType>(targetType))
    return true;

  // Runtime integer to runtime float
  if (mlir::isa<IntegerType>(sourceType) && mlir::isa<FloatType>(targetType))
    return true;

  // Runtime float to runtime integer
  if (mlir::isa<FloatType>(sourceType) && mlir::isa<IntegerType>(targetType))
    return true;

  // Comptime integer to runtime integer
  if (mlir::isa<ComptimeIntType>(sourceType) &&
      mlir::isa<IntegerType>(targetType))
    return true;

  // Comptime integer to comptime integer (identity)
  if (mlir::isa<ComptimeIntType>(sourceType) &&
      mlir::isa<ComptimeIntType>(targetType))
    return true;

  // Comptime integer to runtime float
  if (mlir::isa<ComptimeIntType>(sourceType) &&
      mlir::isa<FloatType>(targetType))
    return true;

  // Comptime float to runtime float
  if (mlir::isa<ComptimeFloatType>(sourceType) &&
      mlir::isa<FloatType>(targetType))
    return true;

  // Comptime float to comptime float (identity)
  if (mlir::isa<ComptimeFloatType>(sourceType) &&
      mlir::isa<ComptimeFloatType>(targetType))
    return true;

  // Comptime float to runtime integer
  if (mlir::isa<ComptimeFloatType>(sourceType) &&
      mlir::isa<IntegerType>(targetType))
    return true;

  // Runtime integer to comptime integer
  if (mlir::isa<IntegerType>(sourceType) &&
      mlir::isa<ComptimeIntType>(targetType))
    return true;

  // Runtime float to comptime float
  if (mlir::isa<FloatType>(sourceType) &&
      mlir::isa<ComptimeFloatType>(targetType))
    return true;

  // Enum to underlying integer type
  if (auto enumType = mlir::dyn_cast<EnumType>(sourceType)) {
    if (auto targetInt = mlir::dyn_cast<IntegerType>(targetType)) {
      // Allow cast to the underlying type or compatible integer types
      return true;
    }
  }

  // Integer to enum type
  if (mlir::isa<IntegerType>(sourceType) ||
      mlir::isa<ComptimeIntType>(sourceType)) {
    if (mlir::isa<EnumType>(targetType))
      return true;
  }

  // Pointer to pointer (@ptrcast)
  if (mlir::isa<PointerType>(sourceType) && mlir::isa<PointerType>(targetType))
    return true;

  // FutureType <-> *CoroFrameType (a future IS a pointer to the frame at
  // runtime; this cast is used by the coroutine lowering pipeline).
  if (auto futureTy = mlir::dyn_cast<FutureType>(sourceType)) {
    if (auto ptrTy = mlir::dyn_cast<PointerType>(targetType))
      if (mlir::isa<CoroFrameType>(ptrTy.getPointeeType()))
        return true;
  }
  if (auto ptrTy = mlir::dyn_cast<PointerType>(sourceType)) {
    if (mlir::isa<CoroFrameType>(ptrTy.getPointeeType()))
      if (mlir::isa<FutureType>(targetType))
        return true;
  }

  return false;
}
