//===- ComptimeInterfaces.h - Comptime Evaluation Interfaces ---*- C++ -*-===//
//
// Interfaces for compile-time evaluation in ParsedCSL.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_COMPTIME_INTERFACES_H
#define PARSED_CSL_COMPTIME_INTERFACES_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <variant>
#include <vector>

namespace mlir {
namespace parsed_csl {

class ComptimeEvalContext;

//===----------------------------------------------------------------------===//
// LValue Descriptor (Replaces MutableLValueAttr)
//===----------------------------------------------------------------------===//

/// Represents a field access in an lvalue access path.
struct FieldAccess {
  StringRef fieldName;

  FieldAccess(StringRef name) : fieldName(name) {}

  bool operator==(const FieldAccess &other) const {
    return fieldName == other.fieldName;
  }
};

/// Represents an index access in an lvalue access path.
struct IndexAccess {
  int64_t index;

  IndexAccess(int64_t idx) : index(idx) {}

  bool operator==(const IndexAccess &other) const {
    return index == other.index;
  }
};

/// A single element in an lvalue access path.
/// Either a field access (struct.field) or index access (array[index]).
using AccessPathElement = std::variant<FieldAccess, IndexAccess>;

/// Descriptor for a comptime-known lvalue.
///
/// Replaces MutableLValueAttr. Instead of using attributes and symbol
/// references, this directly uses operations to identify the base.
///
/// The base can be either:
/// - A VariableOp: Direct reference to a local variable
/// - A GlobalOp: Direct reference to a global variable
/// - A ModuleOp: Reference to a module instance
/// - A BlockArgument: Reference to a function parameter
///
/// Examples:
/// - Direct local variable: LValueDescriptor(varOp, {}, type)
/// - Direct global variable: LValueDescriptor(globalOp, {}, type)
/// - Field access: LValueDescriptor(varOp, {FieldAccess("x")}, type)
/// - Module reference: LValueDescriptor(moduleOp)
/// - Module member: LValueDescriptor(memberVarOp, {}, type)
class LValueDescriptor {
public:
  using BaseVariant =
      std::variant<VariableOp, GlobalOp, ModuleOp, BlockArgument>;

  LValueDescriptor(VariableOp varOp, SmallVector<AccessPathElement> path,
                   Type type)
      : base_(varOp), accessPath_(std::move(path)), valueType_(type) {}

  LValueDescriptor(GlobalOp globalOp, SmallVector<AccessPathElement> path,
                   Type type)
      : base_(globalOp), accessPath_(std::move(path)), valueType_(type) {}

  LValueDescriptor(ModuleOp moduleOp)
      : base_(moduleOp), accessPath_(),
        valueType_(csl::ImportedModuleType::get(moduleOp.getContext())) {}

  LValueDescriptor(BlockArgument arg, SmallVector<AccessPathElement> path,
                   Type type)
      : base_(arg), accessPath_(std::move(path)), valueType_(type) {}

  /// Get the base variant
  const BaseVariant &getBase() const { return base_; }

  /// Check if this lvalue is based on a local variable
  bool isVariableBased() const {
    return std::holds_alternative<VariableOp>(base_);
  }

  /// Check if this lvalue is based on a global variable
  bool isGlobalBased() const { return std::holds_alternative<GlobalOp>(base_); }

  /// Check if this lvalue is based on a module
  bool isModuleBased() const { return std::holds_alternative<ModuleOp>(base_); }

  /// Check if this lvalue is based on a function parameter (block argument)
  bool isParamBased() const {
    return std::holds_alternative<BlockArgument>(base_);
  }

  /// Get the base local variable (asserts if not variable-based)
  VariableOp getBaseVariable() const {
    assert(isVariableBased() && "LValueDescriptor is not variable-based");
    return std::get<VariableOp>(base_);
  }

  /// Get the base global variable (asserts if not global-based)
  GlobalOp getBaseGlobal() const {
    assert(isGlobalBased() && "LValueDescriptor is not global-based");
    return std::get<GlobalOp>(base_);
  }

  /// Get the base module (asserts if not module-based)
  ModuleOp getBaseModule() const {
    assert(isModuleBased() && "LValueDescriptor is not module-based");
    return std::get<ModuleOp>(base_);
  }

  /// Get the base block argument (asserts if not param-based)
  BlockArgument getBlockArgument() const {
    assert(isParamBased() && "LValueDescriptor is not param-based");
    return std::get<BlockArgument>(base_);
  }

  /// Get the base operation regardless of which variant type it is.
  /// This is useful for using Operation* as a storage key.
  /// For param-based descriptors, returns the parent FuncOp.
  Operation *getBaseOperation() const {
    return std::visit(
        [](auto &v) -> Operation * {
          if constexpr (std::is_same_v<std::decay_t<decltype(v)>,
                                       BlockArgument>)
            return v.getOwner()->getParentOp();
          else
            return v.getOperation();
        },
        const_cast<BaseVariant &>(base_));
  }

  /// Get the access path
  ArrayRef<AccessPathElement> getAccessPath() const { return accessPath_; }

  /// Get the value type at this point in the access path
  Type getValueType() const { return valueType_; }

  /// Get whether this lvalue is const
  bool getIsConst() const {
    if (isModuleBased())
      return true; // Modules are always const
    if (isParamBased())
      return true; // Function parameters are always const
    if (isVariableBased())
      return getBaseVariable().getIsConst();
    if (isGlobalBased())
      return getBaseGlobal().getIsConst();
    llvm_unreachable("Unknown base type");
  }

  /// Get the full LValueType for this descriptor
  csl::LValueType getLValueType() const {
    return csl::LValueType::get(valueType_, getIsConst());
  }

  /// Check if this is a direct reference (no access path)
  bool isDirectReference() const { return accessPath_.empty(); }

  /// Extend with a field access
  /// For module-based lvalues, this resolves to the module member directly
  LValueDescriptor extendWithField(StringRef fieldName, Type newType,
                                   ComptimeEvalContext &context) const;

  /// Extend with an index access
  LValueDescriptor extendWithIndex(int64_t index, Type newType) const {
    if (isModuleBased()) {
      llvm::report_fatal_error(
          "Cannot extend module-based lvalue with index access");
    }
    SmallVector<AccessPathElement> newPath(accessPath_);
    newPath.push_back(IndexAccess(index));
    return LValueDescriptor(base_, std::move(newPath), newType);
  }

  /// If this is a GlobalOp-based lvalue storing a module reference,
  /// resolve it to a ModuleOp-based lvalue. Otherwise returns *this unchanged.
  /// This centralizes the module "unpacking" logic.
  LValueDescriptor resolveModuleIfNeeded(ComptimeEvalContext &context) const;

  /// Convert this LValueDescriptor to a GlobalRefAttr if possible.
  /// Returns nullopt if this is not a direct global reference (i.e., if it's
  /// variable-based, module-based, or has a non-empty access path).
  std::optional<csl::GlobalRefAttr> toGlobalRefAttr() const;

  /// Convert this LValueDescriptor to a ModuleRefAttr if possible.
  /// Returns nullopt if this is not a direct module reference (i.e., if it's
  /// variable-based, global-based, or has a non-empty access path).
  std::optional<ModuleRefAttr> toModuleRefAttr() const;

  /// Convert this LValueDescriptor to an attribute representation if possible.
  /// Returns nullopt if conversion is not possible. Only direct global
  /// references and module references can be converted.
  std::optional<TypedAttr> toAttr() const;

  bool operator==(const LValueDescriptor &other) const {
    return base_ == other.base_ && accessPath_ == other.accessPath_ &&
           valueType_ == other.valueType_;
  }

private:
  // Private constructor for internal use (e.g., extending access paths)
  LValueDescriptor(BaseVariant base, SmallVector<AccessPathElement> path,
                   Type type)
      : base_(std::move(base)), accessPath_(std::move(path)), valueType_(type) {
  }

  BaseVariant base_;
  SmallVector<AccessPathElement> accessPath_;
  Type valueType_;
};

/// Holds either a comptime-known value (TypedAttr, which implicitly captures
/// its type) or just a Type (when the value is runtime-only but type is known).
struct ComptimeValueOrType {
  std::variant<Type, TypedAttr> data;

  Type getType() const {
    if (auto *attr = std::get_if<TypedAttr>(&data))
      return attr->getType();
    return std::get<Type>(data);
  }

  std::optional<TypedAttr> getValue() const {
    if (auto *attr = std::get_if<TypedAttr>(&data))
      return *attr;
    return std::nullopt;
  }

  bool hasValue() const { return std::holds_alternative<TypedAttr>(data); }

  bool operator==(const ComptimeValueOrType &other) const {
    return data == other.data;
  }
  bool operator!=(const ComptimeValueOrType &other) const {
    return data != other.data;
  }

  static ComptimeValueOrType fromType(Type t) { return {{t}}; }
  static ComptimeValueOrType fromValue(TypedAttr v) { return {{v}}; }
};

} // namespace parsed_csl
} // namespace mlir

// DenseMapInfo specialization for ComptimeValueOrType, required by SetVector.
namespace llvm {
template <>
struct DenseMapInfo<mlir::parsed_csl::ComptimeValueOrType> {
  using VT = mlir::parsed_csl::ComptimeValueOrType;

  static unsigned getHashValue(const VT &vt) {
    if (auto *attr = std::get_if<mlir::TypedAttr>(&vt.data))
      return hash_combine(1,
                          DenseMapInfo<mlir::Attribute>::getHashValue(*attr));
    return hash_combine(0, DenseMapInfo<mlir::Type>::getHashValue(
                               std::get<mlir::Type>(vt.data)));
  }
  static bool isEqual(const VT &lhs, const VT &rhs) {
    return lhs.data == rhs.data;
  }
};
} // namespace llvm

namespace mlir {
namespace parsed_csl {

//===----------------------------------------------------------------------===//
// Evaluation Error Hierarchy
//===----------------------------------------------------------------------===//

/// Error kind for comptime evaluation failures.
enum class ErrorKind {
  Semantic,  ///< Type system violation, undefined variable (ALWAYS FATAL)
  Execution, ///< Overflow, div-by-zero, would crash at runtime (ALWAYS FATAL)
  ComptimeFailure, ///< Runtime dependency in optional context
  RequiredFailure, ///< Runtime dependency in required context (FATALIZED)
};

/// Unified error type for comptime evaluation.
/// Replaces SemanticError, ExecutionError, and ComptimeEvalFailure.
class ComptimeError {
public:
  /// A single entry in the call stack trace, representing one level of
  /// function call/instantiation nesting.
  struct CallStackEntry {
    Location callSite;
    std::string description; // e.g. "in call to 'foo'"
  };

private:
  ErrorKind kind_;
  std::shared_ptr<Diagnostic> reason_;
  Operation *failedOp_;
  std::vector<Value>
      dependencies_;  ///< Runtime dependencies (for failure kinds)
  Type inferredType_; ///< Inferred result type (for non-semantic errors)
  std::vector<CallStackEntry> callStack_;

public:
  /// Construct a new error for an operation
  ComptimeError(ErrorKind kind, Operation *op, ComptimeEvalContext &context);

  /// Static factory methods (enable fluent syntax)
  static ComptimeError semanticError(Operation *op, ComptimeEvalContext &ctx) {
    return ComptimeError(ErrorKind::Semantic, op, ctx);
  }

  static ComptimeError executionError(Operation *op, ComptimeEvalContext &ctx) {
    return ComptimeError(ErrorKind::Execution, op, ctx);
  }

  static ComptimeError comptimeFailure(Operation *op,
                                       ComptimeEvalContext &ctx) {
    assert(op->getNumResults() == 0 &&
           "Comptime failure for op with result must include result type");
    return ComptimeError(ErrorKind::ComptimeFailure, op, ctx);
  }

  /// Create a comptime failure with result type attached
  static ComptimeError comptimeFailure(Operation *op, ComptimeEvalContext &ctx,
                                       Type resultType) {
    return ComptimeError(ErrorKind::ComptimeFailure, op, ctx)
        .withType(resultType);
  }

  static ComptimeError requiredFailure(Operation *op,
                                       ComptimeEvalContext &ctx) {
    return ComptimeError(ErrorKind::RequiredFailure, op, ctx);
  }

  std::string reasonString() const { return reason_->str(); }

  /// Get the underlying Diagnostic for passing to CSLFolders functions.
  Diagnostic *getDiag() { return reason_.get(); }

  /// Query methods
  ErrorKind getKind() const { return kind_; }
  bool isSemanticError() const { return kind_ == ErrorKind::Semantic; }
  bool isExecutionError() const { return kind_ == ErrorKind::Execution; }
  bool isComptimeFailure() const { return kind_ == ErrorKind::ComptimeFailure; }
  bool isRequiredFailure() const { return kind_ == ErrorKind::RequiredFailure; }

  bool isAlwaysFatal() const { return kind_ != ErrorKind::ComptimeFailure; }

  Operation *getFailedOp() const { return failedOp_; }
  const std::vector<Value> &getDependencies() const { return dependencies_; }

  /// Conversion: upgrade ComptimeFailure to RequiredFailure
  ComptimeError fatalize() const {
    if (kind_ == ErrorKind::ComptimeFailure) {
      ComptimeError fatal = *this;
      fatal.kind_ = ErrorKind::RequiredFailure;
      return fatal;
    }
    return *this; // Already fatal or different kind
  }

  /// Add runtime dependency (for failure kinds)
  ComptimeError &withDependency(Value dep) {
    dependencies_.push_back(dep);
    return *this;
  }

  ComptimeError &withDependencies(ArrayRef<Value> deps) {
    dependencies_.insert(dependencies_.end(), deps.begin(), deps.end());
    return *this;
  }

  /// Add a call stack entry for error tracing.
  /// The description should be a full message like "in call to 'foo'".
  ComptimeError &addCallStackEntry(Location loc, const Twine &description) {
    callStack_.push_back({loc, description.str()});
    return *this;
  }

  /// Attach inferred result type (for non-semantic errors where type inference
  /// succeeded but evaluation failed). Returns a copy with the type attached.
  ComptimeError withType(Type t) const {
    ComptimeError copy = *this;
    copy.inferredType_ = t;
    return copy;
  }

  /// Get the inferred type (may be null for semantic errors)
  Type getInferredType() const { return inferredType_; }

  /// Stream operator for building diagnostics (fluent syntax)
  template <typename T>
  ComptimeError &operator<<(const T &value) {
    (*reason_) << value;
    return *this;
  }

  ComptimeError &operator<<(InFlightDiagnostic &&diag) {
    Location noteLoc = diag.getUnderlyingDiagnostic()->getLocation();
    (*reason_).attachNote(noteLoc) = std::move(*diag.getUnderlyingDiagnostic());
    diag.abandon();
    return *this;
  }

  /// Emit diagnostic, attaching any call stack entries as notes.
  void emit(MLIRContext *ctx) const {
    for (const auto &entry : callStack_)
      reason_->attachNote(entry.callSite) << entry.description;
    DiagnosticEngine &diagEngine = ctx->getDiagEngine();
    diagEngine.emit(std::move(*reason_));
  }
};

/// Tracks all possible control flow outcomes at a program point.
///
/// In comptime-only contexts (comptime blocks, comptime function calls,
/// unrolled loops), all conditions are comptime-known, so exactly one path
/// is taken and at most one field is "active" at a time. In runtime contexts
/// (runtime conditionals), multiple paths may be taken and the lattice
/// accumulates all possibilities via join().
///
/// Two combine operations:
///   join(a, b)       - Union of possibilities from parallel branches (IfOp)
///   a.followedBy(b)  - Sequential composition; only fallthrough paths continue
struct ControlFlowOutcomes {
  /// If set, a fatal error occurred. All other fields are irrelevant.
  std::optional<ComptimeError> fatalError;

  /// Whether some path may fall through (reach next statement).
  bool mayFallthrough = false;

  /// For each target op: values/types that might be broken with.
  /// Keys are Operation* pointers to ForOp, WhileOp, or LabeledBlockOp.
  /// Uses SetVector to deduplicate identical values/types from different
  /// branches, enabling comptime detection when all branches yield the same
  /// value.
  DenseMap<Operation *, SetVector<ComptimeValueOrType>> mayBreakTo;

  /// Target ops that might be continued to (continues carry no value).
  DenseSet<Operation *> mayContinueTo;

  /// Values/types that might be returned from function.
  /// Uses SetVector to deduplicate — e.g., if both branches of an if return
  /// the same comptime value, the set collapses to size 1.
  SetVector<ComptimeValueOrType> mayReturn;

  /// Values/types that might be yielded (for expression regions).
  SetVector<ComptimeValueOrType> mayYield;

  //===--- Query methods ---===//

  bool hasFatalError() const { return fatalError.has_value(); }

  const ComptimeError &getFatalError() const {
    assert(hasFatalError());
    return *fatalError;
  }

  bool hasBreaksTo(Operation *target) const {
    return mayBreakTo.count(target) > 0;
  }

  bool hasContinuesTo(Operation *target) const {
    return mayContinueTo.contains(target);
  }

  bool hasReturns() const { return !mayReturn.empty(); }

  bool hasYields() const { return !mayYield.empty(); }

  /// True if only fallthrough is possible (no exits of any kind).
  bool isFallthroughOnly() const {
    return !hasFatalError() && mayFallthrough && mayBreakTo.empty() &&
           mayContinueTo.empty() && mayReturn.empty() && mayYield.empty();
  }

  //===--- Factory methods ---===//

  static ControlFlowOutcomes fatal(ComptimeError error) {
    ControlFlowOutcomes result;
    result.fatalError = std::move(error);
    return result;
  }

  static ControlFlowOutcomes fallthrough() {
    ControlFlowOutcomes result;
    result.mayFallthrough = true;
    return result;
  }

  static ControlFlowOutcomes breakTo(Operation *target,
                                     ComptimeValueOrType vt) {
    ControlFlowOutcomes result;
    result.mayBreakTo[target].insert(vt);
    return result;
  }

  static ControlFlowOutcomes continueTo(Operation *target) {
    ControlFlowOutcomes result;
    result.mayContinueTo.insert(target);
    return result;
  }

  static ControlFlowOutcomes returnWith(ComptimeValueOrType vt) {
    ControlFlowOutcomes result;
    result.mayReturn.insert(vt);
    return result;
  }

  static ControlFlowOutcomes yieldWith(ComptimeValueOrType vt) {
    ControlFlowOutcomes result;
    result.mayYield.insert(vt);
    return result;
  }

  //===--- Combine operations ---===//

  /// JOIN: Union of possibilities from parallel branches (IfOp, SwitchOp).
  /// Use when multiple paths could be taken (runtime conditional).
  static ControlFlowOutcomes join(const ControlFlowOutcomes &a,
                                  const ControlFlowOutcomes &b) {
    // Fatal errors propagate
    if (a.hasFatalError())
      return a;
    if (b.hasFatalError())
      return b;

    ControlFlowOutcomes result;
    result.mayFallthrough = a.mayFallthrough || b.mayFallthrough;

    // Merge break targets (SetVector deduplicates identical values)
    for (auto &[target, vts] : a.mayBreakTo)
      result.mayBreakTo[target].insert(vts.begin(), vts.end());
    for (auto &[target, vts] : b.mayBreakTo)
      result.mayBreakTo[target].insert(vts.begin(), vts.end());

    // Merge continue targets
    result.mayContinueTo.insert(a.mayContinueTo.begin(), a.mayContinueTo.end());
    result.mayContinueTo.insert(b.mayContinueTo.begin(), b.mayContinueTo.end());

    // Merge returns and yields (SetVector deduplicates identical values)
    result.mayReturn.insert(a.mayReturn.begin(), a.mayReturn.end());
    result.mayReturn.insert(b.mayReturn.begin(), b.mayReturn.end());
    result.mayYield.insert(a.mayYield.begin(), a.mayYield.end());
    result.mayYield.insert(b.mayYield.begin(), b.mayYield.end());

    return result;
  }

  /// FOLLOWED BY: Sequential composition.
  /// Paths that already exited stay; only fallthrough paths continue.
  ControlFlowOutcomes followedBy(const ControlFlowOutcomes &next) const {
    // Fatal errors propagate immediately
    if (hasFatalError())
      return *this;
    if (next.hasFatalError() && mayFallthrough)
      return next;

    ControlFlowOutcomes result;

    // Paths that already exited stay unchanged
    result.mayBreakTo = mayBreakTo;
    result.mayContinueTo = mayContinueTo;
    result.mayReturn = mayReturn;
    result.mayYield = mayYield;

    // Only fallthrough paths get the next op's outcomes
    if (mayFallthrough) {
      for (auto &[target, vts] : next.mayBreakTo)
        result.mayBreakTo[target].insert(vts.begin(), vts.end());
      result.mayContinueTo.insert(next.mayContinueTo.begin(),
                                  next.mayContinueTo.end());
      result.mayReturn.insert(next.mayReturn.begin(), next.mayReturn.end());
      result.mayYield.insert(next.mayYield.begin(), next.mayYield.end());
      result.mayFallthrough = next.mayFallthrough;
    } else {
      result.mayFallthrough = false;
    }

    return result;
  }

  //===--- Consume operations (for loops/blocks) ---===//

  /// Consume breaks targeting this op, return their values/types.
  SetVector<ComptimeValueOrType> consumeBreaksTo(Operation *target) {
    auto it = mayBreakTo.find(target);
    if (it == mayBreakTo.end())
      return {};
    SetVector<ComptimeValueOrType> vts = std::move(it->second);
    mayBreakTo.erase(it);
    return vts;
  }

  /// Consume continues targeting this op. Returns true if any were found.
  bool consumeContinuesTo(Operation *target) {
    return mayContinueTo.erase(target);
  }

  //===--- Type unification ---===//

  /// Progressively find a common type across all entries using implicit
  /// coercion rules (e.g. comptime_int + i32 → i32).
  /// Returns the unified type, or VoidType if empty.
  /// On failure, if \p conflicting is non-null, it is set to the two
  /// incompatible types that caused the failure.
  static FailureOr<Type>
  unifyTypes(ArrayRef<ComptimeValueOrType> vts, MLIRContext *ctx,
             std::pair<Type, Type> *conflicting = nullptr) {
    if (vts.empty())
      return (Type)csl::VoidType::get(ctx);

    Type unified = vts[0].getType();
    for (auto &vt : ArrayRef(vts).drop_front()) {
      if (vt.getType() == unified)
        continue;
      auto common = csl::getCommonType(unified, vt.getType());
      if (failed(common)) {
        if (conflicting)
          *conflicting = {unified, vt.getType()};
        return failure();
      }
      unified = *common;
    }
    return unified;
  }

  /// Overload for a plain array of types.
  static FailureOr<Type>
  unifyTypes(ArrayRef<Type> types, MLIRContext *ctx,
             std::pair<Type, Type> *conflicting = nullptr) {
    if (types.empty())
      return (Type)csl::VoidType::get(ctx);

    Type unified = types[0];
    for (Type ty : ArrayRef(types).drop_front()) {
      if (ty == unified)
        continue;
      auto common = csl::getCommonType(unified, ty);
      if (failed(common)) {
        if (conflicting)
          *conflicting = {unified, ty};
        return failure();
      }
      unified = *common;
    }
    return unified;
  }
};

//===----------------------------------------------------------------------===//
// ComptimeEvalResult
//===----------------------------------------------------------------------===//

/// Unified result type for operation evaluation.
///
/// This is what operations return from their evaluate() methods and what
/// evaluateOperation() returns. It carries:
///
/// - **Value channel**: the comptime-known value (rvalue or lvalue), if any
/// - **Type channel**: the inferred type, even when no value is known
/// - **Fatal error**: a ComptimeError for always-fatal failures (semantic,
///   execution, required failure). The operation is invalid.
/// - **Comptime failure**: a ComptimeError with ErrorKind::ComptimeFailure.
///   The operation is valid but couldn't be evaluated at comptime due to
///   runtime dependencies. Carries diagnostic chain for potential
///   fatalization by evaluateOperation() if the context requires comptime.
///   Control flow is still meaningful (defaults to fallthrough).
/// - **Control flow channel**: a ControlFlowOutcomes lattice tracking all
///   possible control flow exits.
///
/// Most leaf ops only use the value/error channels and get implicit
/// fallthrough control flow. Control-flow ops (break, continue, return,
/// yield) and compound ops (if, for, while, labeled_block) use the
/// control flow channel.
struct ComptimeEvalResult {
  /// The evaluated value: nothing, rvalue, or lvalue.
  std::variant<std::monostate, TypedAttr, LValueDescriptor> value;

  /// The inferred type of this operation's result.
  /// Set even when value is not comptime-known. May be null for void ops.
  Type inferredType;

  /// Fatal error (semantic, execution, required failure).
  /// If set, the operation is invalid and control flow is meaningless.
  std::optional<ComptimeError> error;

  /// Optional comptime failure — operation is valid but couldn't be evaluated
  /// at comptime due to runtime dependencies. Carries diagnostic chain for
  /// potential fatalization. Only set when `error` is NOT set.
  /// Control flow remains meaningful (typically fallthrough).
  std::optional<ComptimeError> comptimeFailure;

  /// Control flow outcomes. Defaults to fallthrough for leaf ops.
  ControlFlowOutcomes controlFlow = ControlFlowOutcomes::fallthrough();

  /// If true, the evaluated op was erased or replaced by new ops at the same
  /// position (e.g. inline expansion, dead loop removal). evaluateRegion
  /// should revisit the same position using a pre-saved iterator.
  bool opWasReplaced = false;

  //===--- Backward-compatible constructors ---===//
  // These allow most leaf op models to remain unchanged.

  /// Construct from rvalue with fallthrough.
  ComptimeEvalResult(TypedAttr attr)
      : value(attr), inferredType(attr.getType()) {}

  /// Construct from lvalue with fallthrough.
  ComptimeEvalResult(LValueDescriptor lval)
      : value(lval), inferredType(lval.getLValueType()) {}

  /// Construct void success with fallthrough. Only accepts success().
  ComptimeEvalResult(LogicalResult res) {
    assert(::mlir::succeeded(res) &&
           "Use ComptimeError for failures, not LogicalResult");
  }

  /// Construct from error. Determines whether it is a fatal error or a
  /// comptime failure based on the error kind.
  ComptimeEvalResult(ComptimeError err) : inferredType(err.getInferredType()) {
    if (err.isComptimeFailure()) {
      // Comptime failure: operation is valid, control flow is fallthrough
      comptimeFailure = std::move(err);
    } else {
      // Fatal error: operation is invalid, no meaningful control flow
      error = std::move(err);
      controlFlow = ControlFlowOutcomes(); // default: not fallthrough
    }
  }

  //===--- Control flow factory methods ---===//

  /// Return from function with comptime-known value.
  static ComptimeEvalResult returnWith(TypedAttr val) {
    ComptimeEvalResult r(success());
    r.value = val;
    r.controlFlow =
        ControlFlowOutcomes::returnWith(ComptimeValueOrType::fromValue(val));
    return r;
  }

  /// Return from function without value (void return).
  static ComptimeEvalResult returnVoid(MLIRContext *ctx) {
    ComptimeEvalResult r(success());
    r.controlFlow = ControlFlowOutcomes::returnWith(
        ComptimeValueOrType::fromValue(csl::VoidAttr::get(ctx)));
    return r;
  }

  /// Return from function — value not comptime-known but type is.
  static ComptimeEvalResult returnWithType(Type t) {
    ComptimeEvalResult r(success());
    r.controlFlow =
        ControlFlowOutcomes::returnWith(ComptimeValueOrType::fromType(t));
    return r;
  }

  /// Break to a target operation with a value/type.
  static ComptimeEvalResult breakTo(Operation *target, ComptimeValueOrType vt) {
    ComptimeEvalResult r(success());
    r.controlFlow = ControlFlowOutcomes::breakTo(target, vt);
    return r;
  }

  /// Continue to a target operation.
  static ComptimeEvalResult continueTo(Operation *target) {
    ComptimeEvalResult r(success());
    r.controlFlow = ControlFlowOutcomes::continueTo(target);
    return r;
  }

  /// Yield a comptime-known rvalue (for comptime expression regions).
  static ComptimeEvalResult yieldWith(TypedAttr val) {
    ComptimeEvalResult r(success());
    r.value = val;
    r.controlFlow =
        ControlFlowOutcomes::yieldWith(ComptimeValueOrType::fromValue(val));
    return r;
  }

  /// Yield an lvalue (for module imports).
  static ComptimeEvalResult yieldWith(const LValueDescriptor &lval) {
    ComptimeEvalResult r(success());
    r.value = lval;
    r.controlFlow = ControlFlowOutcomes::yieldWith(
        ComptimeValueOrType::fromType(lval.getLValueType()));
    return r;
  }

  /// The evaluated op was erased or replaced by new ops at the same position
  /// (e.g. inline expansion, dead loop removal). evaluateRegion should
  /// revisit the same block position using a pre-saved iterator.
  static ComptimeEvalResult replaced() {
    ComptimeEvalResult r(success());
    r.opWasReplaced = true;
    return r;
  }

  /// Construct from ControlFlowOutcomes directly (for compound ops).
  /// Always sets VoidType as the inferred type, since compound ops that
  /// propagate control flow (if/switch/for/while) produce no meaningful
  /// result value — any actual result comes from yields, not the op itself.
  static ComptimeEvalResult fromControlFlow(ControlFlowOutcomes cf,
                                            MLIRContext *ctx) {
    // If the outcomes contain a fatal error, convert to error result
    if (cf.hasFatalError()) {
      return ComptimeEvalResult(ComptimeError(cf.getFatalError()));
    }
    ComptimeEvalResult r(success());
    r.controlFlow = std::move(cf);
    r.inferredType = csl::VoidType::get(ctx);
    return r;
  }

  //===--- Query methods ---===//

  /// True if no fatal error.
  bool succeeded() const { return !error.has_value(); }
  /// True if fatal error.
  bool failed() const { return error.has_value(); }
  /// True if fatal error.
  bool isError() const { return error.has_value(); }
  /// True if comptime failure (not fatal, operation is valid).
  bool isComptimeFailure() const { return comptimeFailure.has_value(); }

  /// True if we have a type but no comptime value and no fatal error.
  /// This includes both the comptime failure case and the "type only" case.
  bool isTypeOnly() const {
    return !error.has_value() && std::holds_alternative<std::monostate>(value);
  }

  bool hasRValue() const { return std::holds_alternative<TypedAttr>(value); }
  bool hasLValue() const {
    return std::holds_alternative<LValueDescriptor>(value);
  }

  /// Get the inferred type.
  /// Falls back to deriving from value or error if inferredType is not set.
  Type getType() const {
    if (inferredType)
      return inferredType;
    if (hasRValue())
      return getRValue().getType();
    if (hasLValue())
      return getLValue().getLValueType();
    if (error)
      return error->getInferredType();
    if (comptimeFailure)
      return comptimeFailure->getInferredType();
    return Type();
  }

  TypedAttr getRValue() const {
    assert(hasRValue() && "Not an rvalue");
    return std::get<TypedAttr>(value);
  }

  const LValueDescriptor &getLValue() const {
    assert(hasLValue() && "Not an lvalue");
    return std::get<LValueDescriptor>(value);
  }

  const ComptimeError &getError() const {
    assert(isError() && "Not a fatal error");
    return *error;
  }

  ComptimeError &getError() {
    assert(isError() && "Not a fatal error");
    return *error;
  }

  const ComptimeError &getComptimeFailure() const {
    assert(isComptimeFailure() && "Not a comptime failure");
    return *comptimeFailure;
  }

  ComptimeError &getComptimeFailure() {
    assert(isComptimeFailure() && "Not a comptime failure");
    return *comptimeFailure;
  }

  operator LogicalResult() const { return success(succeeded()); }

  /// Attach an inferred type to an error or comptime failure result.
  ComptimeEvalResult withType(Type t) const {
    ComptimeEvalResult copy = *this;
    copy.inferredType = t;
    if (copy.error)
      copy.error = copy.error->withType(t);
    if (copy.comptimeFailure)
      copy.comptimeFailure = copy.comptimeFailure->withType(t);
    return copy;
  }
};

//===----------------------------------------------------------------------===//
// EvalRequirement (internal to evaluateOperation)
//===----------------------------------------------------------------------===//

/// Requirement level for compile-time evaluation.
/// Used internally by evaluateOperation() to determine error reporting policy.
enum class EvalRequirement {
  Required,    ///< Evaluation is required (failure is an error)
  Optional,    ///< Evaluation is optional (failure means runtime evaluation)
  DoesntMatter ///< Evaluation outcome is a fatal error regardless of
               ///< requirement
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const LValueDescriptor &lval) {
  os << "LValueDescriptor(base=";
  lval.getBaseOperation()->print(os, OpPrintingFlags().skipRegions());
  os << ", path=[";
  for (const auto &elem : lval.getAccessPath()) {
    if (std::holds_alternative<FieldAccess>(elem)) {
      os << "." << std::get<FieldAccess>(elem).fieldName;
    } else if (std::holds_alternative<IndexAccess>(elem)) {
      os << "[" << std::get<IndexAccess>(elem).index << "]";
    }
  }
  os << "], type=" << lval.getValueType() << ")";
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ComptimeError &error) {
  os << "ComptimeError (kind=";
  switch (error.getKind()) {
  case ErrorKind::Semantic:
    os << "Semantic";
    break;
  case ErrorKind::Execution:
    os << "Execution";
    break;
  case ErrorKind::ComptimeFailure:
    os << "ComptimeFailure";
    break;
  case ErrorKind::RequiredFailure:
    os << "RequiredFailure";
    break;
  }
  os << ") at operation ";
  error.getFailedOp()->print(os, OpPrintingFlags().skipRegions());
  os << ": " << error.reasonString();
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ComptimeEvalResult &result) {
  if (result.isError()) {
    os << "Error: " << result.getError();
  } else if (result.isComptimeFailure()) {
    os << "ComptimeFailure: " << result.getComptimeFailure();
  } else if (result.hasRValue()) {
    os << "Success with rvalue: " << result.getRValue();
  } else if (result.hasLValue()) {
    os << "Success with lvalue: " << result.getLValue();
  } else {
    os << "Success (no value)";
  }
  return os;
}

} // namespace parsed_csl
} // namespace mlir

/// Include the generated interface declarations.
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h.inc"

namespace mlir {

class DialectRegistry;

namespace parsed_csl {

/// Register comptime evaluation external models for ParsedCSL operations.
void registerComptimeModels(DialectRegistry &registry);

} // namespace parsed_csl
} // namespace mlir

#endif // PARSED_CSL_COMPTIME_INTERFACES_H
