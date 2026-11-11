//===- GlobalEvalContext.h - Global Evaluation Context ---------*- C++ -*-===//
//
// Evaluation context for global scope (compilation units and modules).
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_GLOBAL_EVAL_CONTEXT_H
#define PARSED_CSL_GLOBAL_EVAL_CONTEXT_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/ResourceBitmaps.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/BuiltinEvaluators.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/FunctionEvalContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include <functional>
#include <memory>
#include <vector>

namespace mlir {

class Attribute;
class Type;

namespace csl {
class CompilationUnitOp;
class CslFlags;
class CslModuleResolver;
} // namespace csl

namespace parsed_csl {

class ComptimeFunctionEvaluator;

/// Options for comptime evaluation.
/// Encapsulates configuration that affects how comptime evaluation behaves.
struct ComptimeEvalOptions {
  ComptimeEvalOptions() = default;

  /// Construct from CslFlags.
  explicit ComptimeEvalOptions(const csl::CslFlags &flags) : cslFlags(flags) {}

  /// All compiler flags (arch, rectangle, fabric dims, fp16 format, diagnostic
  /// flags, etc.).
  csl::CslFlags cslFlags;

  /// Whether the parser is running in LSP mode (emits LocOps for local
  /// variable references to enable go-to-definition).
  bool lspMode = false;

  /// Predicate: should functions/globals in this file be eagerly evaluated?
  /// When set, non-generic functions and globals in matching files are
  /// evaluated even if not exported or referenced. Used in LSP mode to
  /// provide diagnostics and hover info for open documents.
  std::function<bool(llvm::StringRef filePath)> shouldEagerlyEvaluate;
};

/// Record of a DontEvaluate and MustEvaluateAtFinalComptime builtin call with
/// resolved arguments. Used to track builtin calls that occur during comptime
/// loop unrolling so they can be emitted in the output with their per-iteration
/// argument values.
struct RecordedBuiltinCall {
  csl::BuiltinKind builtin;         ///< Which builtin was called
  SmallVector<TypedAttr> argValues; ///< Resolved argument values as TypedAttr
  Type resultType;                  ///< Inferred result type
  Location location;                ///< Source location
  bool fromLayoutBlock =
      false; ///< True if recorded from within a LayoutBlockOp

  /// Rematerialize the call operation for this recorded builtin call.
  FailureOr<Value> rematerializeCall(OpBuilder &builder) const;
};

/// Evaluation context for global scope (compilation units and modules).
///
/// This context evaluates all operations at global scope in order,
/// including global declarations and function definitions.
/// For imported modules, their contents are evaluated into this same
/// flat context.
class GlobalEvalContext : public ComptimeEvalContext {
public:
  /// Construct and run the analysis on a compilation unit
  explicit GlobalEvalContext(
      ComptimeFunctionEvaluator *funcEvaluator, csl::CompilationUnitOp unit,
      std::shared_ptr<csl::CslModuleResolver> moduleResolver = nullptr,
      ComptimeEvalOptions options = {});

  operator LogicalResult() const { return successful ? success() : failure(); }

  ControlFlowOutcomes evaluateRegion(Region &region, bool isRuntimeConditional,
                                     bool requiresComptime) override;

  /// Override to evaluate an imported module's body into this context.
  ControlFlowOutcomes evaluateModule(ModuleOp module) override;

  /// Override to return this (the global evaluation context itself)
  GlobalEvalContext *getGlobalEvalContext() override { return this; }

  /// Override to return this (const version)
  const GlobalEvalContext *getGlobalEvalContext() const override {
    return this;
  }

  /// Retrieve the source manager (if any) for importing modules.
  std::shared_ptr<llvm::SourceMgr> getSourceMgr() const override;

  /// Retrieve the module resolver (if any) for importing modules.
  csl::CslModuleResolver *getModuleResolver() const override {
    return moduleResolver_ ? moduleResolver_.get() : nullptr;
  }

  bool isComptimeContextImpl() const override;

  /// Evaluate a GenericFuncOp's type region to get the concrete function type,
  /// WITHOUT creating an instantiated FuncOp. This is used to determine the
  /// type of a function call without triggering instantiation.
  std::variant<ComptimeError, FunctionType>
  evaluateFuncType(GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgValues,
                   ArrayRef<Type> anytypeArgTypes) override;

  /// Override to implement function instantiation with caching.
  /// Returns a ComptimeError on failure or the instantiated FuncOp on success.
  /// ALL instantiated FuncOps get unique suffixes (e.g., `_inst0`).
  /// GenericFuncOps are NEVER erased during instantiation.
  std::variant<ComptimeError, FuncOp>
  getOrInstantiateFunc(GenericFuncOp generic,
                       ArrayRef<TypedAttr> comptimeArgValues,
                       ArrayRef<Type> anytypeArgTypes) override;

  /// Look up an already-instantiated function from the cache.
  /// Returns nullptr if not found.
  FuncOp lookupInstantiatedFunc(GenericFuncOp generic,
                                ArrayRef<TypedAttr> comptimeArgValues,
                                ArrayRef<Type> anytypeArgTypes) const;

  /// Get the function evaluation context for a specific function.
  /// Returns nullptr if the function was not analyzed.
  FunctionEvalContext *getFunctionContext(FuncOp func);
  const FunctionEvalContext *getFunctionContext(FuncOp func) const;

  /// Get all function contexts for iteration
  const DenseMap<FuncOp, std::unique_ptr<FunctionEvalContext>> &
  getFunctionContexts() const {
    return functionContexts;
  }

  /// Override to return the registry owned by this context
  const BuiltinEvaluatorRegistry &getBuiltinRegistry() const override {
    return builtinRegistry_;
  }

  /// Record a DontEvaluate/MustEvaluateAtFinalComptime builtin call with its
  /// resolved arguments. Called during comptime evaluation of loops.
  /// The operation is used to determine if the call is from within a layout
  /// block.
  void recordBuiltinCall(csl::BuiltinKind builtin, ArrayRef<TypedAttr> args,
                         Type resultType, Operation *op);

  /// Get all recorded builtin calls
  ArrayRef<RecordedBuiltinCall> getRecordedBuiltinCalls() const {
    return recordedBuiltinCalls;
  }

  /// Evaluate a GlobalOp on demand (for forward references).
  /// Evaluates the global's SSA dependencies first, then the global itself.
  /// Detects circular dependencies via currentlyEvaluatingGlobals_.
  ControlFlowOutcomes evaluateGlobalOnDemand(GlobalOp target);

  /// Get the comptime evaluation options
  const ComptimeEvalOptions &getOptions() const { return options_; }

  /// Update the rectangle dimensions (called when @set_rectangle is evaluated)
  void setRectangle(uint16_t width, uint16_t height) {
    options_.cslFlags.rectangle = {width, height};
  }

  /// Get the current error count (for detecting new errors across a scope)
  unsigned getErrorCount() const { return errorCount_; }

  /// Increment the error count (called from reportComptimeEvalFailure)
  void incrementErrorCount() const { errorCount_++; }

  /// Set whether we're currently evaluating a layout block. Set by the eval
  /// model of LayoutBlockOp and used to determine whether recorded builtin
  /// calls are from within a layout block.
  void setEvaluatingLayoutBlock(bool value) { evaluatingLayoutBlock = value; }

  /// Get whether we're currently evaluating a layout block.
  bool isEvaluatingLayoutBlock() const { return evaluatingLayoutBlock; }

  /// Get the resource bitmaps for comptime resource allocation.
  csl::ResourceBitmaps &getResourceBitmaps() { return resourceBitmaps_; }
  const csl::ResourceBitmaps &getResourceBitmaps() const {
    return resourceBitmaps_;
  }

protected:
  std::string getContextDescription() const override {
    return "GlobalEvalContext";
  }

  llvm::ScopedPrinter &logger() const override {
    return const_cast<llvm::ScopedPrinter &>(logger_);
  }

private:
  /// Comptime evaluation options (arch, etc.)
  ComptimeEvalOptions options_;

  /// Bitmap tracking for comptime resource allocation.
  csl::ResourceBitmaps resourceBitmaps_;

  /// Registry for per-builtin comptime evaluators.
  BuiltinEvaluatorRegistry builtinRegistry_;

  llvm::impl::raw_ldbg_ostream os_{std::string("[comptime-eval :1]"),
                                   llvm::dbgs()};

  /// A logger used to emit diagnostics during the conversion process.
  llvm::ScopedPrinter logger_{os_};

  /// Evaluate a scope body (compilation unit or module)
  ControlFlowOutcomes evaluateScopeBody(Region &body);

  /// Eagerly evaluate non-generic functions and globals in files matched by
  /// the shouldEagerlyEvaluate predicate. Called from the constructor after
  /// export instantiation.
  void eagerlyEvaluateOpenFiles(csl::CompilationUnitOp unit);

  /// Analyze a single function body
  ControlFlowOutcomes analyzeFunctionBody(FuncOp func);

  /// Function-local evaluation contexts (one per function)
  DenseMap<FuncOp, std::unique_ptr<FunctionEvalContext>> functionContexts;

  bool evaluatingLayoutBlock = false;

  /// Cache key for function instantiation
  struct InstantiationKey {
    GenericFuncOp generic;
    SmallVector<TypedAttr> comptimeArgValues;
    SmallVector<Type> anytypeArgTypes;

    bool operator==(const InstantiationKey &other) const {
      return generic == other.generic &&
             comptimeArgValues == other.comptimeArgValues &&
             anytypeArgTypes == other.anytypeArgTypes;
    }
  };

  /// Hash function for InstantiationKey
  struct InstantiationKeyInfo : public DenseMapInfo<InstantiationKey> {
    static unsigned getHashValue(const InstantiationKey &key) {
      unsigned hash = DenseMapInfo<GenericFuncOp>::getHashValue(key.generic);
      for (auto attr : key.comptimeArgValues)
        hash = llvm::hash_combine(hash, mlir::hash_value(attr));
      for (auto type : key.anytypeArgTypes)
        hash = llvm::hash_combine(hash, mlir::hash_value(type));
      return hash;
    }
    static bool isEqual(const InstantiationKey &lhs,
                        const InstantiationKey &rhs) {
      return lhs == rhs;
    }
  };

  /// Cache of instantiated functions
  DenseMap<InstantiationKey, FuncOp, InstantiationKeyInfo> instantiationCache;

  /// Deterministic suffix source for persistent functions in this unit.
  uint64_t nextInstantiationId = 0;

  bool successful;
  bool inComptimeBlock = false;

  /// Counter for errors reported via reportComptimeEvalFailure.
  /// Used by evaluateModule to detect errors that occurred during module
  /// evaluation (even if swallowed by LSP-mode error tolerance).
  mutable unsigned errorCount_ = 0;
  std::shared_ptr<csl::CslModuleResolver> moduleResolver_;

  /// GlobalOps currently being demand-evaluated (for cycle detection).
  DenseSet<Operation *> currentlyEvaluatingGlobals_;

  /// Recorded DontEvaluate/MustEvaluateAtFinalComptime builtin calls from
  /// comptime-only contexts (e.g., during loop unrolling)
  SmallVector<RecordedBuiltinCall> recordedBuiltinCalls;
};

} // namespace parsed_csl
} // namespace mlir

#endif // PARSED_CSL_GLOBAL_EVAL_CONTEXT_H
