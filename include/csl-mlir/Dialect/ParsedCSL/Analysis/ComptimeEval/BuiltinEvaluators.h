//===- BuiltinEvaluators.h - Builtin Comptime Evaluators --------*- C++ -*-===//
//
// Registry for per-builtin comptime evaluators.
// Stored in GlobalEvalContext, accessed via ComptimeEvalContext.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_BUILTIN_EVALUATORS_H
#define PARSED_CSL_BUILTIN_EVALUATORS_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir::parsed_csl {

/// Registry holding per-builtin comptime evaluators.
/// Owned by GlobalEvalContext, accessed via
/// ComptimeEvalContext::getBuiltinRegistry().
class BuiltinEvaluatorRegistry {
public:
  using Concept = ComptimeEvaluableOpInterface::Concept;

  BuiltinEvaluatorRegistry();
  ~BuiltinEvaluatorRegistry();

  BuiltinEvaluatorRegistry(const BuiltinEvaluatorRegistry &) = delete;
  BuiltinEvaluatorRegistry &
  operator=(const BuiltinEvaluatorRegistry &) = delete;

  /// Get the evaluator Concept for a builtin kind.
  const Concept &get(csl::BuiltinKind kind) const;

private:
  llvm::DenseMap<unsigned, std::unique_ptr<Concept>> evaluators;
  std::unique_ptr<Concept> defaultEvaluator;
};

} // namespace mlir::parsed_csl

#endif // PARSED_CSL_BUILTIN_EVALUATORS_H
