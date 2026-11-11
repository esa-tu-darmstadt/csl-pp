//===- ValidateTypedCSLPass.cpp - TypedCSL IR Validation ----------*- C++
//-*-===//
//
// This file implements a validation pass for TypedCSL IR after
// canonicalization.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_VALIDATETYPEDCSLPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

struct ValidateTypedCSLPass
    : public typed_csl::impl::ValidateTypedCSLPassBase<ValidateTypedCSLPass> {
  using ValidateTypedCSLPassBase::ValidateTypedCSLPassBase;

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    bool hasErrors = false;

    unit.walk([&](Operation *op) {
      if (failed(validateComptimeOnlyTypes(op)) ||
          failed(validateArraysOfStructs(op)) ||
          failed(validateArraySignature(op)))
        hasErrors = true;
    });

    if (hasErrors)
      signalPassFailure();
  }

private:
  /// Whether `value` contains an array of structs, possibly nested in a
  /// struct or another array.
  static bool containsArrayOfStructs(Attribute value) {
    if (auto array = mlir::dyn_cast<csl::ArrayValueAttr>(value))
      return mlir::isa<csl::StructType>(array.getElementType()) ||
             llvm::any_of(array.getElements(), containsArrayOfStructs);
    if (auto structValue = mlir::dyn_cast<csl::StructValueAttr>(value))
      return llvm::any_of(structValue.getFieldValues(),
                          containsArrayOfStructs);
    return false;
  }

  /// cslc does not support arrays of structs whose value is known at compile
  /// time: it rejects them or crashes. Such a value reaches the output as a
  /// constant in a function or as the initializer of a global.
  LogicalResult validateArraysOfStructs(Operation *op) {
    if (auto global = mlir::dyn_cast<GlobalOp>(op)) {
      std::optional<TypedAttr> initializer = global.getInitializer();
      if (!initializer || !containsArrayOfStructs(*initializer))
        return success();
      return global.emitError()
             << "global '" << global.getSymName()
             << "' cannot be initialized with an array of structs, because "
                "Cerebras' cslc compiler does not support arrays of structs "
                "that are known at compile time; remove the initializer and "
                "assign the elements at runtime";
    }
    auto constant = mlir::dyn_cast<ConstantOp>(op);
    if (!constant || !constant->getParentOfType<FuncOp>() ||
        !containsArrayOfStructs(constant.getValue()))
      return success();
    return constant.emitError()
           << "Cerebras' cslc compiler does not support arrays of structs "
              "that are known at compile time; assign the elements at runtime "
              "instead";
  }

  /// cslc does not support arrays as parameters or results of runtime
  /// functions. Async functions are exempt: their lowering replaces the
  /// signature.
  LogicalResult validateArraySignature(Operation *op) {
    auto func = mlir::dyn_cast<FuncOp>(op);
    if (!func || func.getIsAsync())
      return success();
    FunctionType funcType = func.getFunctionType();
    for (auto [index, input] : llvm::enumerate(funcType.getInputs())) {
      if (!mlir::isa<csl::ArrayType, csl::PerPEArrayType>(input))
        continue;
      return func.emitError()
             << "parameter " << index + 1 << " of '" << func.getSymName()
             << "' has type '" << csl::prettyPrintType(input)
             << "', but Cerebras' cslc compiler does not support arrays as "
                "parameters of runtime functions; pass a pointer to the array "
                "instead";
    }
    for (Type result : funcType.getResults()) {
      if (!mlir::isa<csl::ArrayType, csl::PerPEArrayType>(result))
        continue;
      return func.emitError()
             << "'" << func.getSymName() << "' returns '"
             << csl::prettyPrintType(result)
             << "', but Cerebras' cslc compiler does not support arrays as "
                "results of runtime functions; write the array through a "
                "pointer parameter instead";
    }
    return success();
  }

  /// Validate that values with comptime-only types are defined by constant-like
  /// operations. Returns failure if validation fails.
  LogicalResult validateComptimeOnlyTypes(Operation *op) {
    // Skip constant-like ops - they're valid producers of comptime values
    if (op->hasTrait<mlir::OpTrait::ConstantLike>())
      return success();

    for (Value result : op->getResults()) {
      Type type = result.getType();

      // Unwrap lvalue if present
      if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(type))
        type = lvalueType.getValueType();

      if (op->hasTrait<mlir::OpTrait::ConstantLike>())
        continue;

      if (!csl::isComptimeOnlyType(type))
        continue;

      if (llvm::all_of(result.getUsers(),
                       [](Operation *user) { return isa<CastOp>(user); })) {
        // Intermediate casts of comptime-only values are allowed as long as
        // they eventually lead to a non-comptime-only type (validated in the
        // CastOp folder)
        continue;
      }

      return op->emitError() << "operation produces the comptime-only type '"
                             << csl::prettyPrintType(result.getType())
                             << "' but failed to evaluate at comptime";
    }

    return success();
  }
};

} // namespace
