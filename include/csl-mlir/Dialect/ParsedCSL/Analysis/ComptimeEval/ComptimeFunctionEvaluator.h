//===- ComptimeFunctionEvaluator.h - Comptime Function Eval ----*- C++ -*-===//
//
// Evaluator for comptime function calls.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_COMPTIME_FUNCTION_EVALUATOR_H
#define PARSED_CSL_COMPTIME_FUNCTION_EVALUATOR_H

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <variant>

namespace mlir {

class Attribute;

namespace parsed_csl {

class ComptimeEvalContext;
class GlobalEvalContext;

/// Represents a function call on the evaluation stack.
/// Used to detect recursive calls with the same arguments.
/// Tracks the original GenericFuncOp (not the temp FuncOp) so that
/// comptime calls via temporary instantiations are properly detected.
struct CallFrame {
  GenericFuncOp generic;
  SmallVector<TypedAttr> comptimeArgValues;
  SmallVector<TypedAttr> runtimeArgValues;

  bool operator==(const CallFrame &other) const {
    return generic == other.generic &&
           comptimeArgValues == other.comptimeArgValues &&
           runtimeArgValues == other.runtimeArgValues;
  }
};

/// Evaluator for comptime function calls.
///
/// This class orchestrates the compile-time evaluation of function calls
/// on instantiated FuncOp. It:
/// 1. Creates a FunctionEvalContext for the function body
/// 2. Casts arguments to parameter types
/// 3. Binds parameters to argument values
/// 4. Evaluates all operations in the function body
/// 5. Extracts and returns the result value
///
/// Recursion is detected by tracking (GenericFuncOp, arguments) on the
/// call stack. Calling the same function with the same arguments while
/// it's already being evaluated is detected as infinite recursion.
///
class ComptimeFunctionEvaluator {
public:
  /// Construct a function evaluator
  ComptimeFunctionEvaluator() = default;

  /// Evaluate an instantiated function call at compile time.
  /// Returns the function's return value, or a ComptimeError if:
  /// - The function cannot be evaluated at compile time
  /// - Any operation in the function fails comptime evaluation
  /// - Recursion with the same arguments is detected
  /// - Argument casting fails
  ///
  /// \param func The instantiated function to evaluate
  /// \param argValues Non-comptime argument values (comptime values are baked
  ///                  into the FuncOp body)
  /// \param parentContext The enclosing scope that the function is defined in.
  ///                      This becomes the parent context for the function's
  ///                      evaluation, allowing the function to access comptime
  ///                      values from its enclosing scope (e.g., global
  ///                      constants for top-level functions, or class members
  ///                      for methods).
  std::variant<ComptimeError, TypedAttr>
  evaluateCall(FuncOp func, ArrayRef<TypedAttr> argValues,
               ComptimeEvalContext *parentContext);

  /// Evaluate a comptime function call via a temporary instantiation.
  /// Creates a temporary FuncOp from the GenericFuncOp, evaluates it, then
  /// erases the temp. This avoids polluting the instantiation cache with
  /// comptime-only calls.
  ///
  /// \param generic The generic function to evaluate
  /// \param funcType The already-computed concrete function type
  /// \param comptimeArgValues Values for comptime parameters
  /// \param anytypeArgTypes Types for anytype parameters
  /// \param runtimeArgValues Values for non-comptime arguments
  /// \param parentContext The enclosing scope context
  std::variant<ComptimeError, TypedAttr> evaluateComptimeCallViaTemp(
      GenericFuncOp generic, FunctionType funcType,
      ArrayRef<TypedAttr> comptimeArgValues, ArrayRef<Type> anytypeArgTypes,
      ArrayRef<TypedAttr> runtimeArgValues, ComptimeEvalContext *parentContext);

  /// Evaluate a GenericFuncOp's type region to get the concrete function type.
  /// This binds comptime params to their values and anytype params to their
  /// types, then evaluates the type region to get the concrete FunctionType.
  ///
  /// \param generic The generic function whose type region to evaluate
  /// \param comptimeArgValues Values for comptime parameters (in comptime param
  /// order)
  /// \param anytypeArgTypes Types for anytype parameters (in anytype param
  /// order)
  /// \param context The context for evaluation (used for binding and
  /// evaluation)
  /// \return The concrete FunctionType or a ComptimeError
  std::variant<ComptimeError, FunctionType> evaluateTypeRegion(
      GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgValues,
      ArrayRef<Type> anytypeArgTypes, ComptimeEvalContext &context);

private:
  /// Call stack for recursion detection
  /// Each entry is (GenericFuncOp, comptime args, runtime args) currently being
  /// evaluated
  SmallVector<CallFrame> callStack_;

  /// Maximum recursion depth (to prevent pathological cases)
  static constexpr unsigned maxRecursionDepth = 1000;

  /// Check if (generic, comptimeArgs, runtimeArgs) is currently on the call
  /// stack
  bool isRecursive(GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgs,
                   ArrayRef<TypedAttr> runtimeArgs) const;

  /// Counter for unique temporary instantiation names
  static std::atomic<int64_t> tempCounter_;

  /// Create a temporary FuncOp from a GenericFuncOp with the given
  /// comptime/anytype arguments. The resulting FuncOp has comptime parameters
  /// inlined as constants.
  std::variant<ComptimeError, FuncOp>
  createTempInstantiation(GenericFuncOp generic, FunctionType funcType,
                          ArrayRef<TypedAttr> comptimeArgValues,
                          ArrayRef<Type> anytypeArgTypes,
                          ComptimeEvalContext &context);
};

} // namespace parsed_csl
} // namespace mlir

#endif // PARSED_CSL_COMPTIME_FUNCTION_EVALUATOR_H
