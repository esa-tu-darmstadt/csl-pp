//===- FunctionEvalContext.h - Function Evaluation Context -----*- C++ -*-===//
//
// Evaluation context for function-local comptime evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_FUNCTION_EVAL_CONTEXT_H
#define PARSED_CSL_FUNCTION_EVAL_CONTEXT_H

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include <memory>

namespace mlir {
namespace parsed_csl {

/// Evaluation context for function-local comptime evaluation.
///
/// Supports nested scopes with parent-chain tracking. Key features:
///
/// 1. **Runtime Conditional Scopes (Write Barriers):**
///    A scope that may not execute (e.g., if statement with runtime condition)
///    acts as a write barrier: you cannot modify comptime variables declared
///    in outer scopes across this boundary, but you can declare and modify
///    local comptime variables within the scope.
///
/// 2. **Comptime-Required Contexts:**
///    A scope where ALL operations must be comptime evaluable (e.g., inside
///    comptime blocks, or when evaluating comptime function calls). This
///    property is inherited: once a parent requires comptime, all descendants
///    must too.
///
/// 3. **Variable Shadowing:**
///    Inner scopes can declare variables that shadow outer scope variables.
///
class FunctionEvalContext : public ComptimeEvalContext {
public:
  struct Scope {
    bool isRuntimeConditional;
    bool requiresComptime;
    /// Variables (by Operation*) declared in this scope
    DenseSet<Operation *> declaredInThisScope;

    Scope(bool isRuntimeConditional, bool requiresComptime)
        : isRuntimeConditional(isRuntimeConditional),
          requiresComptime(requiresComptime) {
      assert(!(isRuntimeConditional && requiresComptime) &&
             "A runtime conditional scope cannot require comptime "
             "evaluation, as it may not execute.");
    }

    struct Guard {
      FunctionEvalContext &context;
      Guard(FunctionEvalContext &context, bool isRuntimeConditional,
            bool requiresComptime)
          : context(context) {
        context.scopeStack_.emplace_back(isRuntimeConditional,
                                         requiresComptime);
        assert(!(context.functionIsComptimeEvaluated && isRuntimeConditional) &&
               "A runtime conditional scope cannot be created inside a "
               "comptime-evaluated function.");
      }
      ~Guard() { context.scopeStack_.pop_back(); }
    };
  };

  /// Create context for function evaluation.
  /// \param parentContext The parent context (typically GlobalEvalContext)
  /// \param funcEvaluator The function evaluator for nested comptime calls
  /// \param func The function being evaluated
  /// \param isComptimeEvaluated Whether this function is being evaluated
  ///        as part of a comptime function call, requiring all operations to be
  ///        evaluated at comptime. If false, only operations marked as
  ///        comptime-required must be evaluated at comptime.
  FunctionEvalContext(ComptimeEvalContext *parentContext,
                      ComptimeFunctionEvaluator *funcEvaluator, FuncOp func,
                      bool isComptimeEvaluated);

  //===--------------------------------------------------------------------===//
  // Comptime-Required Context Query
  //===--------------------------------------------------------------------===//

  bool isComptimeContextImpl() const override;

  bool evaluatingComptimeFunctionCall() const override {
    return functionIsComptimeEvaluated;
  }

  //===--------------------------------------------------------------------===//
  // Variable Operations with Write Barrier Checking
  //===--------------------------------------------------------------------===//

  /// Declare a comptime variable in the current scope inside this function
  /// context.
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  void declareComptimeVariable(Operation *varOp) override;

  /// Check if a store to the given variable would cross a write barrier.
  /// Returns failure if the store would cross a runtime conditional scope
  /// boundary.
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  LogicalResult checkWriteBarrier(Operation *varOp) const override;

  //===--------------------------------------------------------------------===//
  // Comptime Function Call Evaluation
  //===--------------------------------------------------------------------===//

  /// Bind a function parameter to its argument value.
  /// Creates an lvalue descriptor for the parameter and stores the value.
  void bindParameter(Value param, TypedAttr value) {
    auto blockArg = mlir::cast<BlockArgument>(param);
    Type paramType = value.getType();
    auto lvalueType = csl::LValueType::get(paramType, /*isConst=*/true);

    // Create lvalue descriptor for the parameter
    LValueDescriptor paramDesc(blockArg, {}, paramType);

    // Store as lvalue in comptimeValues (so getComptimeLValue finds it)
    comptimeValues.insert_or_assign(param, ComptimeEvalResult(paramDesc));
    inferredTypes[param] = lvalueType;

    // Store the parameter value (so loadLValue can read it)
    (void)storeLValue(paramDesc, value);
  }

  //===--------------------------------------------------------------------===//
  // Comptime Evaluation
  //===--------------------------------------------------------------------===//
  ControlFlowOutcomes evaluateRegion(Region &region, bool isRuntimeConditional,
                                     bool requiresComptime) override;

protected:
  std::string getContextDescription() const override {
    return "FunctionEvalContext for function '" +
           const_cast<FuncOp &>(func_).getSymName().str() + "'";
  }

private:
  /// This function is being evaluated as part of a comptime function call,
  /// requiring all operations to be evaluated at comptime.
  bool functionIsComptimeEvaluated = false;

  /// The function being evaluated
  FuncOp func_;

  /// Stack of nested scopes
  llvm::SmallVector<Scope> scopeStack_;
};

} // namespace parsed_csl
} // namespace mlir

#endif // PARSED_CSL_FUNCTION_EVAL_CONTEXT_H
