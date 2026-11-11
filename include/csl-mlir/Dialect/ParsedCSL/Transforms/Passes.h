//===- Passes.h - ParsedCSL Transformation Passes --------------*- C++ -*-===//
//
// Pass registration and declarations for ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_PARSEDCSL_TRANSFORMS_PASSES_H
#define CSL_MLIR_DIALECT_PARSEDCSL_TRANSFORMS_PASSES_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

namespace csl {
class CompilationUnitOp;
class CslFlags;
class CslModuleResolver;
} // namespace csl

namespace parsed_csl {

class ComptimeEvaluator;

#define GEN_PASS_DECL
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

/// Create a pass that evaluates and folds compile-time operations.
/// \param resolver Module resolver for importing modules
/// \param options Comptime evaluation options (cslFlags, etc.)
std::unique_ptr<Pass> createComptimeEvalPass(
    std::shared_ptr<csl::CslModuleResolver> resolver = nullptr,
    const ComptimeEvalPassOptions &options = {});

/// Materialize the results of an existing evaluation session into ParsedCSL.
///
/// This performs the rewriting normally driven by ComptimeEvalPass without
/// constructing a new evaluator. Clients may therefore instantiate selected
/// functions through `evaluator` before committing its inferred types,
/// constants, reachability, and recorded builtin calls to the IR. Functions
/// referenced from the runtime program are instantiated first.
LogicalResult materializeComptimeEvaluation(csl::CompilationUnitOp unit,
                                            ComptimeEvaluator &evaluator);

/// Create a pass that erases GenericFuncOps and renames instantiated FuncOps
std::unique_ptr<Pass> createFinalizeInstantiationPass();

/// Create a pass that validates and normalizes cast operations
std::unique_ptr<Pass> createNormalizeCastsPass();

/// Create a pass that turns the reads of locations into loads
std::unique_ptr<Pass> createResolveReadsPass();

/// Create a pass that lowers runtime TensorAccessOps to ConstructStructOps
std::unique_ptr<Pass> createLowerTensorAccessPass();

/// Create a pass that validates ParsedCSL IR before lowering
std::unique_ptr<Pass> createValidateParsedCSLPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

} // namespace parsed_csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_PARSEDCSL_TRANSFORMS_PASSES_H
