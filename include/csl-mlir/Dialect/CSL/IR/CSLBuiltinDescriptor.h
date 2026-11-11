//===- CSLBuiltinDescriptor.h - Builtin metadata infrastructure -*- C++ -*-===//
//
// Defines the builtin descriptor infrastructure for CSL builtins, including
// type matching, signature specifications, and comptime behavior categories.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINDESCRIPTOR_H
#define CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINDESCRIPTOR_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TypeName.h"

#include <climits>
#include <memory>
#include <optional>
#include <string>

namespace mlir::csl {

//===----------------------------------------------------------------------===//
// MatchContext - Context passed to matchers during matching
//===----------------------------------------------------------------------===//

/// Context information available to matchers during type matching.
struct MatchContext {
  unsigned argIdx;              ///< Current argument index being matched
  ArrayRef<Type> resolvedTypes; ///< Previously resolved parameter types
  ArrayRef<TypedAttr>
      comptimeValues; ///< Comptime values (NoValueAttr if runtime)
};

//===----------------------------------------------------------------------===//
// MatchResult - Result of a match operation
//===----------------------------------------------------------------------===//

/// Result of matching a type against a matcher.
struct MatchResult {
  bool matched = false;           ///< Whether the type matched
  std::optional<Type> targetType; ///< Target type for casting (if any)

  /// Create a successful match with optional target type.
  static MatchResult success(std::optional<Type> target = std::nullopt) {
    return {true, target};
  }

  /// Create a failed match.
  static MatchResult failure() { return {false, std::nullopt}; }

  /// Implicit conversion to bool for convenience.
  explicit operator bool() const { return matched; }
};

//===----------------------------------------------------------------------===//
// MatcherConcept - Type-erased interface for matchers
//===----------------------------------------------------------------------===//

/// Abstract base class for all matcher implementations.
/// Uses type-erasure pattern to allow polymorphic matcher behavior.
struct MatcherConcept {
  virtual ~MatcherConcept() = default;

  /// Match a type and return the result including target type for casting.
  /// @param type The type to match
  /// @param ctx Context with resolved types and comptime values
  /// @return MatchResult with matched flag and optional target type
  virtual MatchResult match(Type type, const MatchContext &ctx) const = 0;

  /// Get the exact type if this is an exact-type matcher.
  /// Used for converting TypeMatcher to TypeSpec for return types.
  /// @return The exact type, or nullopt if not an exact-type matcher
  virtual std::optional<Type> getExactType() const { return std::nullopt; }

  /// Clone this matcher (for copy semantics).
  virtual std::unique_ptr<MatcherConcept> clone() const = 0;

  /// Get a string representation for error messages.
  virtual std::string toString() const = 0;
};

//===----------------------------------------------------------------------===//
// MatcherBase - CRTP helper for implementing matchers
//===----------------------------------------------------------------------===//

/// CRTP base class that provides clone() implementation.
template <typename Derived>
struct MatcherBase : MatcherConcept {
  std::unique_ptr<MatcherConcept> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

//===----------------------------------------------------------------------===//
// TypeMatcher - Type constraint for builtin parameters
//===----------------------------------------------------------------------===//

/// TypeMatcher specifies what types are acceptable for a builtin parameter.
///
/// Uses type-erased polymorphism to support various matching strategies:
/// - Exact type matching
/// - Type class matching (e.g., any IntegerType)
/// - Reference-based matching (same as previous arg)
/// - Conditional matching (different behavior for comptime vs runtime)
///
/// TypeMatcher also supports:
/// - Coercion flag: when enabled, types that can implicitly cast to the
///   target type are also accepted
/// - Variadic parameters: minCount/maxCount control argument consumption
class TypeMatcher {
public:
  TypeMatcher() = default;
  ~TypeMatcher() = default;

  // Copy operations (deep copy via clone)
  TypeMatcher(const TypeMatcher &other);
  TypeMatcher &operator=(const TypeMatcher &other);

  // Move operations
  TypeMatcher(TypeMatcher &&other) noexcept = default;
  TypeMatcher &operator=(TypeMatcher &&other) noexcept = default;

  /// Check if this is a variadic matcher (can match multiple arguments)
  bool isVariadic() const { return maxCount > 1; }

  /// Match a type and return the result.
  /// @return MatchResult with matched flag and target type for casting
  MatchResult match(Type type, const MatchContext &ctx) const;

  /// Get a string representation of this matcher for error messages.
  std::string toString() const;

  /// Get the exact type if this is an exact-type matcher.
  /// Used for converting TypeMatcher to TypeSpec for return types.
  std::optional<Type> getExactType() const;

  /// Get minimum occurrence count.
  unsigned getMinCount() const { return minCount; }

  /// Get maximum occurrence count.
  unsigned getMaxCount() const { return maxCount; }

  //===--------------------------------------------------------------------===//
  // Factory methods
  //===--------------------------------------------------------------------===//

  /// Match exactly the given type. No coercion allowed.
  static TypeMatcher exact(Type t);

  /// Match types that can coerce to the given type.
  static TypeMatcher coercesTo(Type t);

  /// Match the configuration struct literal of builtin `kind` and coerce
  /// its fields to the types of the configuration (see CSLBuiltinConfig.h).
  static TypeMatcher configOf(BuiltinKind kind);

  /// Match any type.
  static TypeMatcher any();

  /// Match if any of the given matchers match.
  static TypeMatcher anyOf(std::initializer_list<TypeMatcher> matchers);

  /// Match the type of a previous argument.
  static TypeMatcher sameAsTypeOf(unsigned argIdx);

  /// Match types that can coerce to the type of a previous argument.
  static TypeMatcher sameAsTypeOfCoercible(unsigned argIdx);

  /// Match the type specified by the value of a previous argument (TypeAttr).
  static TypeMatcher sameAsValueOf(unsigned argIdx);

  /// Match types that can coerce to the type specified by a previous argument.
  static TypeMatcher sameAsValueOfCoercible(unsigned argIdx);

  /// Match any type that satisfies isa<T>.
  template <typename T>
  static TypeMatcher ofClass();

  /// Conditional matcher: use comptimeMatcher if arg is comptime-known,
  /// otherwise use runtimeMatcher.
  static TypeMatcher ifComptimeKnown(TypeMatcher comptimeMatcher,
                                     TypeMatcher runtimeMatcher);

  /// Match any integer type (IntegerType or ComptimeIntType).
  static TypeMatcher anyInt();

  /// Match any float type (FloatType or ComptimeFloatType).
  static TypeMatcher anyFloat();

  //===--------------------------------------------------------------------===//
  // Modifiers (return new TypeMatcher with modified settings)
  //===--------------------------------------------------------------------===//

  /// Return a copy configured as variadic.
  /// @param minArgs Minimum number of arguments required (default: 0)
  TypeMatcher asVariadic(unsigned minArgs = 0) const;

  TypeMatcher asOptional() const;

  /// Return a copy that preserves LValueType during matching.
  /// By default, LValueType is unwrapped before matching. Use this for
  /// builtins that expect lvalue arguments (like @allocate_fifo buffer).
  TypeMatcher preserveLValue() const;

  /// Check if this matcher should preserve LValueType during matching.
  bool shouldPreserveLValue() const { return preserveLValue_; }

private:
  std::unique_ptr<MatcherConcept> impl;
  unsigned minCount = 1;
  unsigned maxCount = 1;
  bool preserveLValue_ = false;

  /// Private constructor for factory methods.
  explicit TypeMatcher(std::unique_ptr<MatcherConcept> impl);

  /// Helper for creating TypeClassMatcher instances.
  template <typename T>
  static std::unique_ptr<MatcherConcept> createTypeClassMatcher();
};

//===----------------------------------------------------------------------===//
// TypeClassMatcher - Template matcher for type classes
//===----------------------------------------------------------------------===//

/// Matches any type that satisfies mlir::isa<T>.
template <typename T>
struct TypeClassMatcher : MatcherBase<TypeClassMatcher<T>> {
  MatchResult match(Type type, const MatchContext &) const override {
    if (mlir::isa<T>(type))
      return MatchResult::success(); // No specific target type
    return MatchResult::failure();
  }

  std::string toString() const override {
    // Use LLVM's type name demangling
    return ("any " + llvm::getTypeName<T>().str());
  }
};

//===----------------------------------------------------------------------===//
// TypeMatcher template implementations
//===----------------------------------------------------------------------===//

template <typename T>
std::unique_ptr<MatcherConcept> TypeMatcher::createTypeClassMatcher() {
  return std::make_unique<TypeClassMatcher<T>>();
}

template <typename T>
TypeMatcher TypeMatcher::ofClass() {
  return TypeMatcher(createTypeClassMatcher<T>());
}

//===----------------------------------------------------------------------===//
// TypeSpecConcept - Type-erased interface for type specifiers
//===----------------------------------------------------------------------===//

/// Abstract base class for type specifier implementations.
/// TypeSpec specifies/computes a return type from context (not for matching).
struct TypeSpecConcept {
  virtual ~TypeSpecConcept() = default;

  /// Get the specified type given the context.
  /// @param args The argument values (for computing dependent return types)
  /// @return The specified type
  virtual Type getType(ArrayRef<TypedAttr> args) const = 0;

  /// Clone this specifier (for copy semantics).
  virtual std::unique_ptr<TypeSpecConcept> clone() const = 0;
};

/// CRTP base class that provides clone() implementation for TypeSpec.
template <typename Derived>
struct TypeSpecBase : TypeSpecConcept {
  std::unique_ptr<TypeSpecConcept> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }
};

//===----------------------------------------------------------------------===//
// TypeSpec - Specifies/computes a return type
//===----------------------------------------------------------------------===//

/// TypeSpec specifies what return type a builtin signature produces.
/// Unlike TypeMatcher (which matches input types), TypeSpec computes/specifies
/// a type from context without matching.
///
/// Examples:
/// - TypeSpec::exact(i32) - return type is always i32
/// - TypeSpec::sameAsValueOf(0) - return type is the type in arg 0's TypeAttr
/// - TypeSpec::sameAsTypeOf(1) - return type is the type of arg 1
class TypeSpec {
public:
  TypeSpec() = default;
  ~TypeSpec() = default;

  // Copy operations (deep copy via clone)
  TypeSpec(const TypeSpec &other);
  TypeSpec &operator=(const TypeSpec &other);

  // Move operations
  TypeSpec(TypeSpec &&other) noexcept = default;
  TypeSpec &operator=(TypeSpec &&other) noexcept = default;

  /// Implicit constructor from TypeMatcher (must be an exact-type matcher).
  /// Allows using TypeMatcher variables directly as return types in
  /// SignatureSpec.
  TypeSpec(const TypeMatcher &matcher);

  /// Get the specified type given the argument values.
  Type getType(ArrayRef<TypedAttr> args) const;

  //===--------------------------------------------------------------------===//
  // Factory methods
  //===--------------------------------------------------------------------===//

  /// Return type is exactly the given type.
  static TypeSpec exact(Type t);

  /// Return type is the type of a previous argument.
  static TypeSpec sameAsTypeOf(unsigned argIdx);

  /// Return type is the type specified by the value of a previous argument
  /// (TypeAttr).
  static TypeSpec sameAsValueOf(unsigned argIdx);

  /// Return type is computed by a custom function.
  static TypeSpec fromFunc(std::function<Type(ArrayRef<TypedAttr>)> func);

  //===--------------------------------------------------------------------===//
  // Modifiers
  //===--------------------------------------------------------------------===//

  /// Return a copy that preserves LValueType in the computed return type.
  /// By default, LValueType is stripped from the return type (builtins return
  /// rvalues). Use this for builtins that return lvalue references.
  TypeSpec preserveLValue() const;

  /// Whether the computed return type keeps an LValueType.
  bool shouldPreserveLValue() const { return preserveLValue_; }

private:
  std::unique_ptr<TypeSpecConcept> impl;
  bool preserveLValue_ = false;

  /// Private constructor for factory methods.
  explicit TypeSpec(std::unique_ptr<TypeSpecConcept> impl);
};

//===----------------------------------------------------------------------===//
// SignatureSpec - Specification for a single builtin signature
//===----------------------------------------------------------------------===//

/// SignatureSpec defines the parameter and return types for one valid
/// signature of a builtin. Builtins can have multiple valid signatures.
struct SignatureSpec {
  SmallVector<TypeMatcher> paramMatchers; ///< Matchers for each parameter
  TypeSpec returnType;                    ///< Specifies the return type

  /// Constructor for fixed return type.
  SignatureSpec(std::initializer_list<TypeMatcher> params, Type retType)
      : paramMatchers(params.begin(), params.end()),
        returnType(TypeSpec::exact(retType)) {}

  /// Constructor with TypeSpec for return type.
  SignatureSpec(std::initializer_list<TypeMatcher> params, TypeSpec retType)
      : paramMatchers(params.begin(), params.end()),
        returnType(std::move(retType)) {}

  /// Constructor for dynamic return type via lambda.
  SignatureSpec(std::initializer_list<TypeMatcher> params,
                std::function<Type(ArrayRef<TypedAttr>)> retTypeFunc)
      : paramMatchers(params.begin(), params.end()),
        returnType(TypeSpec::fromFunc(std::move(retTypeFunc))) {}
};

//===----------------------------------------------------------------------===//
// BuiltinEvaluationBehavior - How a builtin behaves during comptime evaluation
//===----------------------------------------------------------------------===//

// /// ComptimeBehavior categorizes how builtins behave during compile-time
// /// evaluation in the source-to-source compiler.
// enum class ComptimeBehavior {
//   /// Always evaluated at our comptime, erased from output.
//   /// Examples: @comptime_print, @type_of, @is_same_type
//   MustEvaluate,

//   /// CAN be evaluated if all args are comptime, otherwise kept in output.
//   /// Examples: @get_color (returns ColorAttr if comptime), @as
//   CanEvaluate,

//   /// NOT evaluated by us, always kept in output.
//   /// Examples: @add16, DSD operations, @activate, @unblock
//   DontEvaluate,

//   /// All args must be comptime-known, but NOT evaluated by us.
//   /// Kept in output for final CSL compiler's comptime phase.
//   /// Examples: @bind_local_task, @set_rectangle (in layout block)
//   MustEvaluateAtFinalComptime
// };

// /// ComptimeBehavior categorizes whether our compiler must evaluate,
// /// can evaluate, or must keep a builtin for the final CSL compiler.
// enum class BuiltinEvaluationBehavior {
//   MustEvaluate, // Our compiler must and can evaluate (e.g., @comptime_print)
//   CanEvaluate,  // Our compiler evaluates if args are comptime, otherwise
//   kept
//                 // for final CSL compiler (e.g., @as)
//   RuntimeOnly,  // Our compiler does not evaluate, always kept for final
//                 // CSL compiler (e.g., @add16)
//   MustEvaluateAtFinalComptime // Our compiler does not evaluate, always kept
//   for
//                               // final CSL compiler but args must be
//                               // comptime-known (e.g., @bind_local_task)
// };

/// Categorizes whether a builtin is pure or has side effects at our comptime,
/// final CSL compiler's comptime, or at runtime. Implies whether our compiler
/// must evaluate it (OurComptimeSideEffects), keep it for final CSL compiler
/// (FinalComptimeSideEffects and RuntimeSideEffects), or can erase it if unused
/// (Pure).
enum BuiltinSideEffects {
  Pure,                     ///< No side effects, can be erased if unused (e.g.,
                            ///< @type_of)
  OurComptimeSideEffects,   ///< Has side effects during our comptime eval
                            ///< (e.g., @comptime_print), but can be stripped
                            ///< afterwards. Implies that our compiler must
                            ///< evaluate it.
  FinalComptimeSideEffects, ///< Has side effects during final
                            ///< CSL compiler's comptime eval (e.g.
                            ///< @bind_local_task). Implies that our compiler
                            ///< must keep it for final CSL compiler, and that
                            ///< all args must be comptime known.
  RuntimeSideEffects,       ///< Has side effects at runtime. Implies that our
                            ///< compiler must keep it for final CSL compiler.
  FinalComptimeOrRuntimeSideEffects, ///< Has side effects at final comptime or
                                     ///< runtime, depending on execution
                                     ///< context (e.g., @unblock, @activate)
  OurComptimeOrRuntimeSideEffects,   ///< Has side effects at our comptime or
                                   ///< runtime, depending on execution context
                                   ///< (e.g.,
                                   ///< @assert)
};

/// Contexts where a builtin is allowed to appear.
enum class BuiltinAllowedContext {
  RuntimeOnly,      // Can only appear at runtime (e.g., @add16)
  ComptimeOnly,     // Can only appear in comptime blocks (e.g.,
                    // @bind_local_task)
  RuntimeOrComptime // Can appear in either (e.g., @unblock, @activate)
};

//===----------------------------------------------------------------------===//
// BuiltinDescriptor - Complete metadata for a builtin
//===----------------------------------------------------------------------===//

/// BuiltinDescriptor contains all metadata needed to validate and process
/// a CSL builtin function call.
struct BuiltinDescriptor {
  BuiltinKind kind; ///< The builtin kind enum value
  // BuiltinEvaluationBehavior behavior;    ///< Comptime evaluation behavior
  BuiltinAllowedContext allowedContext;  ///< Allowed contexts
  BuiltinSideEffects sideEffects;        ///< Side effects at various phases
  SmallVector<SignatureSpec> signatures; ///< Valid signatures for this builtin
  /// Argument indices that must be comptime-known even in runtime context.
  /// The parser wraps these in ComptimeExprOp. For example, the type argument
  /// of @bitcast(type, value) is at index 0.
  SmallVector<unsigned, 2> comptimeArgIndices = {};
  /// Argument indices of which only the type is used, e.g. the argument of
  /// @type_of. Such an argument is not evaluated for its value.
  SmallVector<unsigned, 1> typeOnlyArgIndices = {};

  /// Builder method: return a copy with the given comptime argument indices.
  BuiltinDescriptor
  withComptimeArgs(std::initializer_list<unsigned> indices) && {
    comptimeArgIndices.assign(indices);
    return std::move(*this);
  }

  /// Builder method: return a copy with the given type-only argument indices.
  BuiltinDescriptor
  withTypeOnlyArgs(std::initializer_list<unsigned> indices) && {
    typeOnlyArgIndices.assign(indices);
    return std::move(*this);
  }

  /// Get the descriptor for a builtin kind.
  /// The MLIRContext is needed to construct type matchers.
  static const BuiltinDescriptor &get(BuiltinKind kind, MLIRContext *ctx);

  /// Get the builtin name from the TableGen-generated stringifyBuiltinKind().
  StringRef getName() const;

  /// Determine if this builtin must be evaluated at our comptime.
  bool mustEvaluateAtOurComptime(bool inComptimeContext) const {
    return sideEffects == BuiltinSideEffects::OurComptimeSideEffects ||
           (inComptimeContext &&
            sideEffects == BuiltinSideEffects::OurComptimeOrRuntimeSideEffects);
  }

  bool mustEvaluateAtFinalComptime(bool inComptimeContext) const {
    return sideEffects == BuiltinSideEffects::FinalComptimeSideEffects ||
           (inComptimeContext &&
            sideEffects ==
                BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects);
  }

  /// Determine if this builtin must be kept in the output IR for the final
  /// CSL compiler.
  bool mustKeepInOutput(bool inComptimeContext) const {
    return sideEffects == BuiltinSideEffects::FinalComptimeSideEffects ||
           sideEffects == BuiltinSideEffects::RuntimeSideEffects ||
           sideEffects ==
               BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects ||
           (!inComptimeContext &&
            sideEffects == BuiltinSideEffects::OurComptimeOrRuntimeSideEffects);
  }

  /// Whether the given argument should be wrapped in a comptime expression
  /// by the parser. Returns true if the arg index is in comptimeArgIndices,
  /// or if the builtin has OurComptimeSideEffects (all args must be comptime).
  bool shouldWrapArgInComptime(unsigned argIdx) const {
    if (sideEffects == BuiltinSideEffects::OurComptimeSideEffects)
      return true;
    return llvm::is_contained(comptimeArgIndices, argIdx);
  }

  /// Whether the builtin reads the value of the given argument. It does not
  /// if the argument is a location, e.g. the variable whose field @field
  /// accesses, or if only its type is used, as by @type_of. The parser reads
  /// every other argument where the source evaluates it.
  bool readsArgument(unsigned argIdx) const {
    if (llvm::is_contained(typeOnlyArgIndices, argIdx))
      return false;
    return llvm::none_of(signatures, [&](const SignatureSpec &signature) {
      return argIdx < signature.paramMatchers.size() &&
             signature.paramMatchers[argIdx].shouldPreserveLValue();
    });
  }

  /// Whether the result may be a location, e.g. the field of a variable that
  /// @field accesses.
  bool mayReturnLocation() const {
    return llvm::any_of(signatures, [](const SignatureSpec &signature) {
      return signature.returnType.shouldPreserveLValue();
    });
  }

  bool hasRuntimeSideEffects(bool inComptimeContext) const {
    return sideEffects == BuiltinSideEffects::RuntimeSideEffects ||
           (!inComptimeContext &&
            sideEffects ==
                BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects) ||
           (!inComptimeContext &&
            sideEffects == BuiltinSideEffects::OurComptimeOrRuntimeSideEffects);
  }
};

//===----------------------------------------------------------------------===//
// SignatureMatchResult - Result of signature matching
//===----------------------------------------------------------------------===//

/// SignatureMatchResult contains the outcome of matching argument types
/// against a builtin's signatures.
struct SignatureMatchResult {
  bool isValid;                         ///< Match succeeded
  SmallVector<Type> resolvedParamTypes; ///< Resolved param types
  SmallVector<std::pair<unsigned, Type>>
      requiredCasts;                 ///< (argIdx, targetType)
  std::string errorMessage;          ///< Error if !isValid
  Type inferredReturnType;           ///< Inferred return type
  std::optional<SignatureSpec> spec; ///< The matched signature spec

  /// Create a success result.
  static SignatureMatchResult
  success(SignatureSpec spec, Type resultType, ArrayRef<Type> paramTypes,
          ArrayRef<std::pair<unsigned, Type>> casts);

  /// Create a failure result with an error message.
  static SignatureMatchResult failure(StringRef message);
};

//===----------------------------------------------------------------------===//
// BuiltinSignatureMatcher - Main API for signature matching
//===----------------------------------------------------------------------===//

/// BuiltinSignatureMatcher provides the main API for matching argument types
/// against builtin signatures and determining required implicit casts.
class BuiltinSignatureMatcher {
public:
  /// Match argument types against a builtin's signatures.
  /// @param kind The builtin kind
  /// @param argValues The values of the args. Either the actual comptime-known
  /// argument values, or NoValueAttr with the runtime type.
  /// @param ctx The MLIR context
  /// @return Match result with resolved types and required casts
  static SignatureMatchResult
  match(BuiltinKind kind, ArrayRef<TypedAttr> argValues, MLIRContext *ctx);

  /// Infer the return type for a builtin call.
  /// @param kind The builtin kind
  /// @param argValues The values of the args. Either the actual comptime-known
  /// argument values, or NoValueAttr with the runtime type.
  /// @param ctx The MLIR context
  /// @return The inferred return type, or failure if no match
  static FailureOr<Type> inferReturnType(BuiltinKind kind,
                                         ArrayRef<TypedAttr> argValues,
                                         MLIRContext *ctx);

  /// Get the required implicit casts for a builtin call.
  /// @param kind The builtin kind
  /// @param argValues The values of the args. Either the actual comptime-known
  /// argument values, or NoValueAttr with the runtime type.
  /// @param ctx The MLIR context
  /// @return List of (argIndex, targetType) pairs for required casts
  static FailureOr<SmallVector<std::pair<unsigned, Type>>>
  getRequiredCasts(BuiltinKind kind, ArrayRef<TypedAttr> argValues,
                   MLIRContext *ctx);
};

} // namespace mlir::csl

#endif // CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINDESCRIPTOR_H
