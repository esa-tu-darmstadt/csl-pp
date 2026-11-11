//===- ComptimeEvalContext.h - Comptime Evaluation Context -----*- C++ -*-===//
//
// Abstract context interface for compile-time evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_COMPTIME_EVAL_CONTEXT_H
#define PARSED_CSL_COMPTIME_EVAL_CONTEXT_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ScopedPrinter.h"
#include <memory>
#include <variant>

namespace mlir::csl {
class CslModuleResolver;
} // namespace mlir::csl

namespace mlir {
namespace parsed_csl {

class BuiltinEvaluatorRegistry;
class ComptimeFunctionEvaluator;
class GlobalEvalContext;
class ModuleOp;

/// Context for compile-time evaluation.
///
/// Operations implementing ComptimeEvaluableOpInterface use this interface
/// to query comptime known values, inferred types, and current variable values.
/// Contexts can be chained: The function-level context refers to the
/// compilation-unit-level context as its parent. If a value is not found
/// in the current context, the parent context is queried.
class ComptimeEvalContext {
public:
  virtual ~ComptimeEvalContext() = default;
  ComptimeEvalContext(ComptimeEvalContext *parent = nullptr,
                      ComptimeFunctionEvaluator *funcEvaluator = nullptr)
      : parentContext(parent), funcEvaluator_(funcEvaluator) {}

  //===--------------------------------------------------------------------===//
  // Value Queries
  //===--------------------------------------------------------------------===//

  /// Check if a value is comptime-known
  bool isComptimeKnown(Value v, bool voidAlwaysComptimeKnown = true) const;

  /// Get the comptime value for an SSA value.
  /// If `voidAlwaysComptimeKnown` is true, then values of void-like types
  /// (void and structs of only void, see csl::isVoidLikeType) are always
  /// comptime-known with their single value. Returns failure if not
  /// comptime-known.
  FailureOr<TypedAttr>
  getComptimeRValue(Value v, bool voidAlwaysComptimeKnown = true) const;

  /// Get the comptime lvalue descriptor for an SSA value of LValueType
  /// Returns nullopt if not comptime-known as an lvalue
  FailureOr<LValueDescriptor> getComptimeLValue(Value v) const;

  /// Get any failure information for a value that failed comptime evaluation
  FailureOr<ComptimeError> getComptimeEvalFailure(Value v) const;

  /// Check if a value is a comptime-known rvalue (loading lvalues if needed).
  /// This is useful for checking if implicit casts between integer types are
  /// allowed, since comptime-known values can be checked for overflow.
  bool isRValueComptimeKnown(Value v) const {
    return succeeded(getOrLoadComptimeRValue(v));
  }

  //===--------------------------------------------------------------------===//
  // Type Queries
  //===--------------------------------------------------------------------===//

  /// Get the inferred type for an SSA value
  /// Returns the inferred type, or failure if no type is known
  FailureOr<Type> getInferredType(Value v) const;

  /// Get the value type of a GlobalOp during comptime evaluation.
  /// This computes the type from the type operand or initializer,
  /// since the inferred_type attribute is not yet set during evaluation.
  FailureOr<Type> getGlobalValueType(GlobalOp globalOp) const;

  /// Get the inferred concrete type for an SSA value
  /// Returns the inferred type of type, or failure if no type is known or if
  /// the type is not a `ConcreteType`
  template <typename ConcreteType>
  FailureOr<ConcreteType> getInferredType(Value v) {
    FailureOr<Type> type = getInferredType(v);
    if (failed(type))
      return failure();
    if (!mlir::isa<ConcreteType>(*type))
      return failure();
    return mlir::cast<ConcreteType>(*type);
  }

  /// Get the inferred type for a value, automatically unwrapping lvalue types.
  /// Use this when you need the rvalue/element type rather than the lvalue
  /// type.
  FailureOr<Type> getInferredRValueType(Value v) const {
    FailureOr<Type> type = getInferredType(v);
    if (failed(type))
      return failure();
    return unwrapLValueType(*type);
  }

  /// Get the inferred concrete rvalue type for a value.
  /// Unwraps lvalue types and casts to the specified type.
  template <typename ConcreteType>
  FailureOr<ConcreteType> getInferredRValueType(Value v) {
    FailureOr<Type> type = getInferredRValueType(v);
    if (failed(type))
      return failure();
    if (!mlir::isa<ConcreteType>(*type))
      return failure();
    return mlir::cast<ConcreteType>(*type);
  }

  //===--------------------------------------------------------------------===//
  // Variable Lookup (unified interface)
  //===--------------------------------------------------------------------===//

  /// Load the value described by an lvalue descriptor.
  /// Resolves the base variable and applies the access path.
  /// Returns failure if not comptime-known.
  FailureOr<TypedAttr> loadLValue(const LValueDescriptor &descriptor) const;

  /// Store a value to an lvalue descriptor.
  /// Applies updates along the access path using copy-on-write semantics.
  /// Returns failure if not supported or crosses runtime boundary.
  LogicalResult storeLValue(const LValueDescriptor &descriptor,
                            TypedAttr value);

  /// Returns the comptime rvalue for a value, loading it if necessary.
  /// Returns failure if not comptime-known or loading fails.
  FailureOr<TypedAttr> getOrLoadComptimeRValue(Value v) const;

  //===--------------------------------------------------------------------===//
  // Context Queries
  //===--------------------------------------------------------------------===//

  /// Declare a comptime variable (VariableOp or GlobalOp) in this context.
  /// Derived classes may override to track the precise scope inside functions
  /// in which a variable is declared.
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  virtual void declareComptimeVariable(Operation *varOp);

  /// Check if a comptime variable is declared in this specific context
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  bool isComptimeVariableDeclaredHere(Operation *varOp) const;

  /// Check if an operation was evaluated (is alive/reachable).
  /// Dead code after early exits will not be marked as alive.
  bool isOpAlive(Operation *op) const;

  /// Mark an operation as alive. Used for newly created ops during the pass.
  void markOpAlive(Operation *op) { aliveOps_.insert(op); }

  /// Check if an lvalue descriptor's base variable is comptime-known
  bool isLValueComptime(const LValueDescriptor &descriptor) const;

  /// Check if an operation must be evaluated at compile-time if it were
  /// executed now.
  bool opMustEvaluateAtComptime(Operation *op) const;

  /// Returns true if all ops in this context must be evaluated at compile-time,
  /// i.e. because we are in a comptime block or function.
  /// When suppressComptimeBlock is set (via evaluateRegionSuppressed), always
  /// returns false regardless of the subclass's own comptime state.
  virtual bool isComptimeContext() const final {
    if (suppressComptimeBlock)
      return false;
    return isComptimeContextImpl();
  }

  /// Subclasses implement this to report their own comptime-context state.
  /// Called by isComptimeContext() after checking the suppress flag.
  virtual bool isComptimeContextImpl() const = 0;

  /// Get the parent context (for walking the context chain)
  ComptimeEvalContext *getParentContext() { return parentContext; }

  /// Get the global evaluation context.
  /// Walks up the parent chain to find the GlobalEvalContext.
  virtual GlobalEvalContext *getGlobalEvalContext() {
    if (parentContext)
      return const_cast<ComptimeEvalContext *>(parentContext)
          ->getGlobalEvalContext();
    return nullptr;
  }

  /// Get the global evaluation context (const version).
  virtual const GlobalEvalContext *getGlobalEvalContext() const {
    if (parentContext)
      return parentContext->getGlobalEvalContext();
    return nullptr;
  }

  virtual bool evaluatingComptimeFunctionCall() const { return false; }

  /// Check if a function has comptime-only argument or return types
  FailureOr<bool> isComptimeOnlyFunction(FuncOp func) const;

  //===--------------------------------------------------------------------===//
  // Function Instantiation
  //===--------------------------------------------------------------------===//

  /// Result of instantiating a callee function. Contains the instantiated
  /// FuncOp and indices of comptime parameters that were inlined.
  struct InstantiationResult {
    /// The instantiated function
    FuncOp callee;
    /// Indices of comptime params that were inlined into the function body.
    SmallVector<int32_t> comptimeParamIndices;
  };

  /// Get or instantiate the callee FuncOp for a CallOp.
  /// Handles both generic and non-generic functions, including cross-module
  /// calls. Returns the instantiated FuncOp and indices of comptime params
  /// that were inlined into the function body.
  std::variant<ComptimeError, InstantiationResult>
  getOrInstantiateCallee(CallOp callOp);

  /// Get the already-instantiated callee for a CallOp.
  /// Unlike getOrInstantiateCallee, this does not create new instantiations.
  /// Returns nullopt if the callee was not previously instantiated (e.g., dead
  /// code that will be removed).
  ///
  /// IMPORTANT: This method must be called on the context that owns the CallOp.
  FailureOr<InstantiationResult> getInstantiatedCallee(CallOp callOp) const;

  // Get the already-instantiated callee for a comptime-known non-generic
  // function, referenced by the given FuncRefAttr or FuncAddrAttr. The function
  // must be non-generic, as comptime args and anytype args cannot be specified
  // here.
  FailureOr<InstantiationResult>
  getInstantiatedCallee(TypedAttr funcRef, csl::CompilationUnitOp cu) const;
  /// Evaluate a GenericFuncOp's type region to get the concrete function type,
  /// WITHOUT creating an instantiated FuncOp. Delegates to the global context.
  virtual std::variant<ComptimeError, FunctionType>
  evaluateFuncType(GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgValues,
                   ArrayRef<Type> anytypeArgTypes) {
    return parentContext->evaluateFuncType(generic, comptimeArgValues,
                                           anytypeArgTypes);
  }

  /// Get or create an instantiated FuncOp from a GenericFuncOp. Comptime
  /// evaluates the instantiated function body. ALL instantiated FuncOps get
  /// unique suffixes. GenericFuncOps are NEVER erased during instantiation.
  /// \param generic The generic function to instantiate
  /// \param comptimeArgValues Values for comptime parameters
  /// \param anytypeArgTypes Inferred types of anytype parameters
  /// \returns The instantiated FuncOp, or a ComptimeError if instantiation
  /// fails
  virtual std::variant<ComptimeError, FuncOp>
  getOrInstantiateFunc(GenericFuncOp generic,
                       ArrayRef<TypedAttr> comptimeArgValues,
                       ArrayRef<Type> anytypeArgTypes) {
    return parentContext->getOrInstantiateFunc(generic, comptimeArgValues,
                                               anytypeArgTypes);
  }

  /// Retrieve the source manager (if any) for importing modules. Only provided
  /// by the top-level context because module imports are only allowed there.
  virtual std::shared_ptr<llvm::SourceMgr> getSourceMgr() const {
    return nullptr;
  }

  /// Retrieve the module resolver (if any) for importing modules.
  /// Only provided by the top-level context.
  virtual csl::CslModuleResolver *getModuleResolver() const { return nullptr; }

  //===--------------------------------------------------------------------===//
  // Comptime Evaluation
  //===--------------------------------------------------------------------===//

  /// Evaluate a comptime function call on an instantiated FuncOp.
  /// Returns the result value or a ComptimeError if evaluation fails.
  ///
  /// \param func The instantiated function to evaluate
  /// \param argValues Non-comptime argument values (comptime values are baked
  ///                  into the FuncOp body)
  std::variant<ComptimeError, TypedAttr>
  evaluateComptimeCall(FuncOp func, ArrayRef<TypedAttr> argValues) const;

  /// Evaluate a comptime function call via a temporary instantiation.
  /// Creates a temp FuncOp, evaluates it, then erases the temp.
  /// Used for comptime calls to avoid polluting the instantiation cache.
  std::variant<ComptimeError, TypedAttr>
  evaluateComptimeCallViaTemp(GenericFuncOp generic, FunctionType funcType,
                              ArrayRef<TypedAttr> comptimeArgValues,
                              ArrayRef<Type> anytypeArgTypes,
                              ArrayRef<TypedAttr> runtimeArgValues) const;

  /// Evaluate an operation at compile-time.
  ///
  /// Returns a ComptimeEvalResult containing:
  /// - The comptime value (if evaluable) or inferred type
  /// - Any error (fatal) or comptime failure (optional)
  /// - Control flow outcomes
  ///
  /// This method handles requirement policy internally: if the operation
  /// must evaluate at comptime but fails, the comptime failure is fatalized
  /// into a fatal error before returning.
  ComptimeEvalResult evaluateOperation(Operation *op);

  /// Evaluate a region, which can be runtime conditional and/or require
  /// comptime evaluation. Returns ControlFlowOutcomes tracking all possible
  /// control flow exits from the region.
  ///
  /// Derived classes implement this to handle scope management (write barriers,
  /// comptime requirement flags). They delegate to the protected
  /// evaluateRegion(Region&) for the actual per-operation evaluation loop.
  virtual ControlFlowOutcomes evaluateRegion(Region &region,
                                             bool isRuntimeConditional,
                                             bool requiresComptime) = 0;

  /// Evaluate a region with comptime context temporarily suppressed.
  /// isComptimeContext() returns false for the duration of region evaluation,
  /// even if the surrounding context requires comptime. Used for tensor access
  /// expression bodies where induction variables are symbolic.
  ControlFlowOutcomes evaluateRegionSuppressed(Region &region) {
    bool prev = suppressComptimeBlock;
    suppressComptimeBlock = true;
    auto restore = llvm::scope_exit([&] { suppressComptimeBlock = prev; });
    return evaluateRegion(region, /*isRuntimeConditional=*/false,
                          /*requiresComptime=*/false);
  }

  /// Evaluate an imported module's body.
  /// Default implementation returns a fatal error - modules can only be
  /// imported at global scope.
  virtual ControlFlowOutcomes evaluateModule(ModuleOp module);

  /// Report a comptime evaluation failure with dependency chain
  void reportComptimeEvalFailure(MLIRContext *ctx,
                                 const ComptimeError &failure) const;

  /// Manually set the inferred type for a value (for pattern rewrites)
  void setInferredTypeDirectly(Value v, Type type) { inferredTypes[v] = type; }

  /// Set the comptime value for an SSA value (used for parameter binding)
  void setComptimeValue(Value v, TypedAttr attr) {
    inferredTypes[v] = attr.getType();
    comptimeValues.insert_or_assign(v, ComptimeEvalResult(attr));
  }

  /// Clear all cached entries (comptimeValues, inferredTypes, aliveOps) for
  /// Values and Operations defined within the given region. This is used to
  /// prevent dangling pointer issues when the region's parent operation is
  /// erased after evaluation.
  void clearEntriesForRegion(Region &region) {
    for (Block &block : region) {
      // Clear block argument entries
      for (BlockArgument arg : block.getArguments()) {
        comptimeValues.erase(arg);
        inferredTypes.erase(arg);
        comptimeParamValues.erase(arg);
      }
      // Clear operation result entries
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          comptimeValues.erase(result);
          inferredTypes.erase(result);
        }
        aliveOps_.erase(&op);
        // Recursively clear nested regions
        for (Region &nested : op.getRegions()) {
          clearEntriesForRegion(nested);
        }
      }
    }
  }

  /// Get the builtin evaluator registry.
  /// Used by BuiltinCallOpComptimeModel to delegate to per-builtin evaluators.
  virtual const BuiltinEvaluatorRegistry &getBuiltinRegistry() const {
    assert(parentContext && "No parent context to get builtin registry from");
    return parentContext->getBuiltinRegistry();
  }

protected:
  virtual llvm::ScopedPrinter &logger() const {
    assert(parentContext && "No parent context to get logger from");
    return parentContext->logger();
  }

  virtual std::string getContextDescription() const = 0;

  /// Evaluate a region without handling runtime conditional or comptime
  /// requirement flags. Used by derived classes to implement their own
  /// evaluateRegion() methods. Composes per-operation outcomes using
  /// followedBy() sequential composition.
  ControlFlowOutcomes evaluateRegion(Region &region);

  /// Helper function to recursively apply copy-on-write updates along an
  /// access path. Returns nullptr on failure.
  TypedAttr applyAccessPathUpdate(TypedAttr currentValue,
                                  ArrayRef<AccessPathElement> accessPath,
                                  TypedAttr newValue, size_t pathIndex) const;

  /// Load the current value of a comptime variable by its defining operation.
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  /// Returns failure if not comptime-known.
  FailureOr<TypedAttr> loadVariable(Operation *varOp) const;

  /// Store a value to a comptime variable by its defining operation.
  /// This is an internal helper; use storeLValue for the public interface.
  /// Calls checkWriteBarrier before storing.
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  /// \param value The value to store
  LogicalResult storeVariable(Operation *varOp, TypedAttr value);

  /// Check if a store to the given variable would cross a write barrier.
  /// Returns failure if the store is not allowed.
  /// Derived classes may override to implement write barrier semantics.
  /// \param varOp The operation defining the variable (VariableOp or GlobalOp)
  virtual LogicalResult checkWriteBarrier(Operation *varOp) const {
    return success(); // Base implementation: no write barriers
  }

  /// Maps SSA values to their comptime-known values
  DenseMap<Value, ComptimeEvalResult> comptimeValues;

  /// Maps SSA values to their inferred types
  DenseMap<Value, Type> inferredTypes;

  /// Maps variable operations (VariableOp or GlobalOp) to their current values.
  /// Key is Operation* to support both VariableOp (local) and GlobalOp
  /// (global). For VariableOp, use varOp.getOperation() as key. For GlobalOp,
  /// use globalOp.getOperation() as key.
  DenseMap<Operation *, TypedAttr> comptimeVarValues;

  /// Maps function parameter block arguments to their comptime-known values.
  /// Used for comptime function calls where parameter values are known.
  DenseMap<Value, TypedAttr> comptimeParamValues;

  /// Variables declared in this specific context (by Operation*)
  DenseSet<Operation *> declaredInThisContext_;

  /// Set of operations that were actually evaluated (alive/reachable)
  DenseSet<Operation *> aliveOps_;

  ComptimeEvalContext *parentContext = nullptr;

  /// Function evaluator for nested comptime calls
  ComptimeFunctionEvaluator *funcEvaluator_ = nullptr;

  /// When true, isComptimeContext() returns false regardless of subclass state.
  /// Set temporarily by evaluateRegionSuppressed() via llvm::scope_exit.
  bool suppressComptimeBlock = false;
};

} // namespace parsed_csl
} // namespace mlir

#endif // PARSED_CSL_COMPTIME_EVAL_CONTEXT_H
