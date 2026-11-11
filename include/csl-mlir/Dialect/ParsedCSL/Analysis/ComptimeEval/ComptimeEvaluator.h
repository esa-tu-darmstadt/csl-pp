//===- ComptimeEvaluator.h - Compile-Time Evaluation -----------*- C++ -*-===//
//
// Top-level compile-time evaluation for compilation units.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_COMPTIME_EVALUATOR_H
#define PARSED_CSL_COMPTIME_EVALUATOR_H

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeFunctionEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir {

class Attribute;
class Type;

namespace csl {
class CompilationUnitOp;
class CslModuleResolver;
} // namespace csl

namespace parsed_csl {

class CallOp;
class FuncOp;
class FunctionEvalContext;

/// Top-level compile-time evaluator for a compilation unit.
///
/// The evaluator owns all state for one evaluation session: the low-level
/// function evaluator, global and per-function contexts, the instantiation
/// cache, and inferred compile-time values and types. Compiler clients may
/// explicitly instantiate selected runtime functions before the evaluation
/// results are materialized into ParsedCSL IR.
///
/// Query routing:
/// 1. First tries the global context (for top-level values)
/// 2. Then finds the function containing the value and queries that context
///
/// An evaluator mutates its compilation unit when it instantiates a function
/// and is not thread-safe. Independent compilation units may be evaluated in
/// parallel with one evaluator per unit.
///
class ComptimeEvaluator {
public:
  /// Construct an evaluation session and evaluate its global scope.
  /// \param unit The compilation unit to analyze
  /// \param moduleResolver Module resolver for importing modules
  /// \param options Comptime evaluation options (arch, etc.)
  explicit ComptimeEvaluator(
      csl::CompilationUnitOp unit,
      std::shared_ptr<csl::CslModuleResolver> moduleResolver = nullptr,
      ComptimeEvalOptions options = {});

  /// Evaluate a generic function's concrete type without instantiating it.
  ///
  /// `comptimeArguments` contains values for the function's comptime
  /// parameters in declaration order. `anytypeArgumentTypes` contains the
  /// inferred types of its anytype parameters in declaration order.
  std::variant<ComptimeError, FunctionType>
  evaluateFunctionType(GenericFuncOp function,
                       ArrayRef<TypedAttr> comptimeArguments,
                       ArrayRef<Type> anytypeArgumentTypes = {});

  /// Return the cached concrete runtime function or instantiate it.
  ///
  /// Instantiation evaluates the function's type and body, removes comptime
  /// parameters from its runtime signature, and creates a persistent FuncOp
  /// in the compilation unit. Repeating the same query returns the cached op.
  std::variant<ComptimeError, FuncOp>
  instantiateFunction(GenericFuncOp function,
                      ArrayRef<TypedAttr> comptimeArguments,
                      ArrayRef<Type> anytypeArgumentTypes = {});

  //===--------------------------------------------------------------------===//
  // Query Interface
  //===--------------------------------------------------------------------===//

  /// Get the comptime value for an SSA rvalue.
  /// Returns failure if the value is not comptime-known.
  FailureOr<TypedAttr> getComptimeRValue(Value v) const;

  /// Get the descriptor for an SSA lvalue.
  /// Returns failure if the descriptor is not comptime-known.
  FailureOr<LValueDescriptor> getComptimeLValue(Value v) const;

  /// Get the comptime rvalue for an SSA value, loading from lvalue if needed.
  /// Returns failure if the value is not comptime-known.
  FailureOr<TypedAttr> getOrLoadComptimeRValue(Value v) const;

  /// Get the inferred type for an SSA value.
  /// Returns failure if no type could be inferred.
  FailureOr<Type> getInferredType(Value v) const;

  /// Check if a value is comptime-known.
  /// Works for both rvalues (checks if value is comptime-known) and lvalues
  /// (checks if the base variable is comptime-known).
  bool isComptimeKnown(Value v, bool voidAlwaysComptimeKnown = false) const;

  /// Get the appropriate evaluation context for a value.
  /// Returns the function context if the value is in a function,
  /// or the global context if the value is at global scope.
  const ComptimeEvalContext &getContextForValue(Value v) const;

  /// Get the appropriate evaluation context for an operation.
  /// Returns the function context if the operation is in a function,
  /// or the global context if the operation is at global scope.
  /// Returns nullptr if the operation is in a function without a context
  /// (e.g., comptime-only functions that will be erased).
  const ComptimeEvalContext *getContextForOp(Operation *op) const;

  /// Check if an operation was evaluated (is alive/reachable).
  /// Dead code after early exits will not be marked as alive.
  bool isOpAlive(Operation *op) const;

  /// Mark an operation as alive.
  void markOpAlive(Operation *op);

  /// Get the already-instantiated callee for a CallOp.
  /// Routes to the appropriate context based on where the CallOp is located.
  /// Returns nullopt if the callee was not previously instantiated.
  std::optional<ComptimeEvalContext::InstantiationResult>
  getInstantiatedCallee(CallOp callOp) const;

  /// Get the global evaluation context (for global scope queries)
  const GlobalEvalContext &getGlobalContext() const { return globalContext; }

  /// Get the global evaluation context (mutable)
  GlobalEvalContext &getGlobalContext() { return globalContext; }

  /// Get the function evaluation context for a specific function.
  /// Returns nullptr if the function was not analyzed.
  const FunctionEvalContext *getFunctionContext(FuncOp func) const;

  /// Check if a function has comptime-only argument or return types
  FailureOr<bool> isComptimeOnlyFunction(FuncOp func) const;

  //===--------------------------------------------------------------------===//
  // Materialization Interface
  //===--------------------------------------------------------------------===//

  /// Get the constant that materialization substitutes for `operand`.
  /// Returns failure if the operand remains a runtime value.
  FailureOr<TypedAttr> getMaterializedValue(OpOperand &operand) const;

  /// Check if materialization erases `op` because it was evaluated entirely
  /// at comptime: comptime calls, stores to comptime variables, and stores of
  /// void-like values, which change nothing.
  bool isErasedAsComptimeStatement(Operation *op) const;

  /// Instantiate every non-generic function referenced from code that remains
  /// in the runtime program after materialization: materialized operands,
  /// initializers of runtime globals, and arguments of recorded builtin calls.
  /// Instantiating a function analyzes its body, which may reference further
  /// functions, so this repeats until no new function is instantiated.
  ///
  /// References keep the name of the GenericFuncOp. A non-generic function
  /// has exactly one instance, which FinalizeInstantiationPass renames back
  /// to the original name.
  LogicalResult instantiateRuntimeReferences(csl::CompilationUnitOp unit);

  /// Force evaluation of a parsed function with concrete argument values.
  ///
  /// The function may have ordinary, comptime, and anytype parameters. This
  /// method partitions the supplied values according to the GenericFuncOp's
  /// parameter metadata, evaluates its concrete function type, and uses a
  /// temporary instantiation so a compiler client can call a comptime-only
  /// helper without making it part of the emitted runtime program.
  ///
  /// Every argument must be known as a TypedAttr. In particular, an anytype
  /// parameter is specialized to the concrete type of its corresponding
  /// argument while the value remains available to the evaluator.
  std::variant<ComptimeError, TypedAttr>
  evaluateFunction(GenericFuncOp func, ArrayRef<TypedAttr> arguments);

  /// Check if initialization and evaluation succeeded.
  operator LogicalResult() const { return successful ? success() : failure(); }

  //===--------------------------------------------------------------------===//
  // Mutation Interface (for use during pattern rewrites)
  //===--------------------------------------------------------------------===//

  /// Manually register the inferred type for an SSA value.
  /// Can be used by pattern rewrites that create new operations where automatic
  /// type inference hasn't run yet.
  ///
  /// WARNING: When used from within a pattern rewrite, make sure pattern
  /// rollback is disabled, as the analysis and the IR would get out of sync
  /// otherwise.
  void setInferredType(Value v, Type type);

private:
  /// Function evaluator for comptime function calls
  /// NOTE: Must be declared before globalContext since globalContext
  /// constructor takes a pointer to this
  ComptimeFunctionEvaluator funcEvaluator;

  /// Global evaluation context (evaluates globals and functions)
  GlobalEvalContext globalContext;

  /// Whether initialization and evaluation succeeded without errors.
  bool successful = true;
};

} // namespace parsed_csl
} // namespace mlir

#endif // PARSED_CSL_COMPTIME_EVALUATOR_H
