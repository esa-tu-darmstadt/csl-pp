//===- FunctionEvalContext.cpp - Function Evaluation Context -------------===//
//
// Implementation of function-local comptime evaluation context.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/FunctionEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeFunctionEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::parsed_csl;

FunctionEvalContext::FunctionEvalContext(
    ComptimeEvalContext *parentContext,
    ComptimeFunctionEvaluator *funcEvaluator, FuncOp func,
    bool isComptimeEvaluated)
    : ComptimeEvalContext(parentContext, funcEvaluator),
      functionIsComptimeEvaluated(isComptimeEvaluated), func_(func) {
  // Infer parameter types from function type
  Region &body = func.getBody();
  FunctionType funcType = func.getFunctionType();
  if (!body.empty()) {
    Block &entryBlock = body.front();
    auto blockArgs = entryBlock.getArguments();
    auto paramTypes = funcType.getInputs();

    // Map each parameter (block argument) to its const lvalue type
    for (auto [blockArg, paramType] : llvm::zip(blockArgs, paramTypes)) {
      inferredTypes[blockArg] =
          csl::LValueType::get(paramType, /*isConst=*/true);
    }
  }
}

LogicalResult FunctionEvalContext::checkWriteBarrier(Operation *varOp) const {
  // Global variables always cross write barriers from inside functions:
  // A comptime function must not modify global comptime variables.
  if (!isComptimeVariableDeclaredHere(varOp)) {
    return emitError(varOp->getLoc())
           << "cannot modify comptime variable not declared in this function";
  }

  // Declared in this context - check the scope stack
  // Walk from innermost to outermost, looking for runtime conditional scopes
  for (auto it = scopeStack_.rbegin(); it != scopeStack_.rend(); ++it) {
    const Scope &scope = *it;

    // If we reach the scope where the variable was declared, no barrier exists
    if (scope.declaredInThisScope.contains(varOp)) {
      return success();
    }

    // If we cross a runtime conditional scope before finding the declaration,
    // that's a write barrier
    if (scope.isRuntimeConditional) {
      return emitError(varOp->getLoc())
             << "cannot modify comptime variable across runtime conditional "
                "scope boundary";
    }
  }

  llvm::report_fatal_error(
      "Variable declared in this context but not found in any scope");
}

ControlFlowOutcomes
FunctionEvalContext::evaluateRegion(Region &region, bool isRuntimeConditional,
                                    bool requiresComptime) {
  // Create a new scope for the region
  Scope::Guard scopeGuard(*this, isRuntimeConditional, requiresComptime);

  return ComptimeEvalContext::evaluateRegion(region);
}

void FunctionEvalContext::declareComptimeVariable(Operation *varOp) {
  // Declare in the current scope
  assert(!scopeStack_.empty() &&
         "No scope available to declare comptime variable");
  scopeStack_.back().declaredInThisScope.insert(varOp);

  // Declare in this context too (using base class method)
  ComptimeEvalContext::declareComptimeVariable(varOp);
}

bool FunctionEvalContext::isComptimeContextImpl() const {
  // The function is a comptime context if it is being comptime-evaluated
  // or if we are inside a scope that requires comptime evaluation
  if (functionIsComptimeEvaluated)
    return true;

  return llvm::any_of(
      scopeStack_, [](const Scope &scope) { return scope.requiresComptime; });
}