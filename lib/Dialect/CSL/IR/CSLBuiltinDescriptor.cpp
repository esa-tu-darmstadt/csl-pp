//===- CSLBuiltinDescriptor.cpp - Builtin metadata implementation ---------===//
//
// Implementation of the builtin descriptor infrastructure for CSL builtins.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinConfig.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

#define DEBUG_TYPE "builtin-matcher"

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Concrete Matcher Implementations
//===----------------------------------------------------------------------===//

namespace {

/// Matches exactly a specific type, optionally allowing coercion.
struct ExactMatcher : MatcherBase<ExactMatcher> {
  Type targetType;
  bool coercionAllowed;

  explicit ExactMatcher(Type t, bool coercion = false)
      : targetType(t), coercionAllowed(coercion) {}

  MatchResult match(Type type, const MatchContext &ctx) const override {
    if (type == targetType)
      return MatchResult::success(targetType);
    if (coercionAllowed) {
      bool comptimeKnown =
          !mlir::isa<NoValueAttr>(ctx.comptimeValues[ctx.argIdx]);
      if (isImplicitCastAllowed(type, targetType, comptimeKnown))
        return MatchResult::success(targetType);
    }
    return MatchResult::failure();
  }

  std::optional<Type> getExactType() const override { return targetType; }

  std::string toString() const override {
    if (coercionAllowed)
      return "coerces to " + prettyPrintType(targetType);
    return prettyPrintType(targetType);
  }
};

/// Matches the type of a previous argument, optionally allowing coercion.
struct SameAsTypeOfMatcher : MatcherBase<SameAsTypeOfMatcher> {
  unsigned argIdx;
  bool coercionAllowed;

  explicit SameAsTypeOfMatcher(unsigned idx, bool coercion = false)
      : argIdx(idx), coercionAllowed(coercion) {}

  MatchResult match(Type type, const MatchContext &ctx) const override {
    Type target = ctx.resolvedTypes[argIdx];
    if (type == target)
      return MatchResult::success(target);
    if (coercionAllowed) {
      bool comptimeKnown =
          !mlir::isa<NoValueAttr>(ctx.comptimeValues[ctx.argIdx]);
      if (isImplicitCastAllowed(type, target, comptimeKnown))
        return MatchResult::success(target);
    }
    return MatchResult::failure();
  }

  std::string toString() const override {
    std::string result = "same type as argument " + std::to_string(argIdx);
    if (coercionAllowed)
      result = "coerces to " + result;
    return result;
  }
};

/// Matches the type specified by a TypeAttr value of a previous argument,
/// optionally allowing coercion.
struct SameAsValueOfMatcher : MatcherBase<SameAsValueOfMatcher> {
  unsigned argIdx;
  bool coercionAllowed;

  explicit SameAsValueOfMatcher(unsigned idx, bool coercion = false)
      : argIdx(idx), coercionAllowed(coercion) {}

  MatchResult match(Type type, const MatchContext &ctx) const override {
    if (auto typeAttr = dyn_cast<csl::TypeAttr>(ctx.comptimeValues[argIdx])) {
      Type target = typeAttr.getValue();
      if (type == target)
        return MatchResult::success(target);
      if (coercionAllowed) {
        bool comptimeKnown =
            !mlir::isa<NoValueAttr>(ctx.comptimeValues[ctx.argIdx]);
        if (isImplicitCastAllowed(type, target, comptimeKnown))
          return MatchResult::success(target);
      }
    }
    return MatchResult::failure();
  }

  std::string toString() const override {
    std::string result =
        "same type as value of argument " + std::to_string(argIdx);
    if (coercionAllowed)
      result = "coerces to " + result;
    return result;
  }
};

/// Matches any type.
struct AnyMatcher : MatcherBase<AnyMatcher> {
  MatchResult match(Type, const MatchContext &) const override {
    return MatchResult::success(); // No specific target type
  }

  std::string toString() const override { return "any type"; }
};

/// Matches the configuration struct literal of a builtin, e.g. the second
/// argument of @get_dsd. Its target type has the same fields, each coerced to
/// the type the configuration gives it, so that NormalizeCasts casts e.g. a
/// comptime `.stride = -1` of a mem1d_dsd to `i8`. The comptime evaluator
/// reports unknown or mistyped fields (see checkConfig).
struct ConfigMatcher : MatcherBase<ConfigMatcher> {
  BuiltinKind kind;

  explicit ConfigMatcher(BuiltinKind kind) : kind(kind) {}

  MatchResult match(Type type, const MatchContext &ctx) const override {
    if (!mlir::isa<StructType>(type))
      return MatchResult::failure();
    const ConfigSchema *schema =
        getConfigSchema(kind, ctx.argIdx, ctx.comptimeValues);
    if (!schema)
      return MatchResult::success();
    FailureOr<Type> coerced = coerceConfigType(type, *schema);
    if (failed(coerced))
      return MatchResult::failure();
    if (*coerced == type)
      return MatchResult::success();
    return MatchResult::success(*coerced);
  }

  std::string toString() const override { return "configuration struct"; }
};

/// Matches if any of the nested matchers match.
struct AnyOfMatcher : MatcherBase<AnyOfMatcher> {
  std::vector<TypeMatcher> nestedMatchers;

  explicit AnyOfMatcher(std::vector<TypeMatcher> matchers)
      : nestedMatchers(std::move(matchers)) {}

  MatchResult match(Type type, const MatchContext &ctx) const override {
    // Return the result of the first matcher that matches
    for (const TypeMatcher &m : nestedMatchers) {
      MatchResult result = m.match(type, ctx);
      if (result.matched)
        return result;
    }
    return MatchResult::failure();
  }

  std::string toString() const override {
    std::string result = "any of (";
    for (size_t i = 0; i < nestedMatchers.size(); ++i) {
      result += nestedMatchers[i].toString();
      if (i + 1 < nestedMatchers.size())
        result += ", ";
    }
    result += ")";
    return result;
  }
};

/// Conditional matcher: uses different matchers for comptime vs runtime.
struct ComptimeOrRuntimeMatcher : MatcherBase<ComptimeOrRuntimeMatcher> {
  TypeMatcher comptimeMatcher;
  TypeMatcher runtimeMatcher;

  ComptimeOrRuntimeMatcher(TypeMatcher comptime, TypeMatcher runtime)
      : comptimeMatcher(std::move(comptime)),
        runtimeMatcher(std::move(runtime)) {}

  MatchResult match(Type type, const MatchContext &ctx) const override {
    bool isComptime = !isa<NoValueAttr>(ctx.comptimeValues[ctx.argIdx]);
    return isComptime ? comptimeMatcher.match(type, ctx)
                      : runtimeMatcher.match(type, ctx);
  }

  std::string toString() const override {
    return "if comptime: " + comptimeMatcher.toString() +
           ", else: " + runtimeMatcher.toString();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// TypeMatcher Implementation
//===----------------------------------------------------------------------===//

TypeMatcher::TypeMatcher(std::unique_ptr<MatcherConcept> impl)
    : impl(std::move(impl)) {}

TypeMatcher::TypeMatcher(const TypeMatcher &other)
    : impl(other.impl ? other.impl->clone() : nullptr),
      minCount(other.minCount), maxCount(other.maxCount),
      preserveLValue_(other.preserveLValue_) {}

TypeMatcher &TypeMatcher::operator=(const TypeMatcher &other) {
  if (this != &other) {
    impl = other.impl ? other.impl->clone() : nullptr;
    minCount = other.minCount;
    maxCount = other.maxCount;
    preserveLValue_ = other.preserveLValue_;
  }
  return *this;
}

MatchResult TypeMatcher::match(Type type, const MatchContext &ctx) const {
  if (!impl)
    return MatchResult::failure();
  return impl->match(type, ctx);
}

std::string TypeMatcher::toString() const {
  if (!impl)
    return "<invalid matcher>";
  std::string result;
  if (getMinCount() == 0 && getMaxCount() == 1)
    result += "optional ";
  else if (isVariadic())
    result += "variadic ";
  result += impl->toString();
  return result;
}

std::optional<Type> TypeMatcher::getExactType() const {
  if (!impl)
    return std::nullopt;
  return impl->getExactType();
}

//===----------------------------------------------------------------------===//
// TypeMatcher Factory Methods
//===----------------------------------------------------------------------===//

TypeMatcher TypeMatcher::exact(Type t) {
  return TypeMatcher(std::make_unique<ExactMatcher>(t));
}

TypeMatcher TypeMatcher::coercesTo(Type t) {
  return TypeMatcher(std::make_unique<ExactMatcher>(t, /*coercion=*/true));
}

TypeMatcher TypeMatcher::any() {
  return TypeMatcher(std::make_unique<AnyMatcher>());
}

TypeMatcher TypeMatcher::configOf(BuiltinKind kind) {
  return TypeMatcher(std::make_unique<ConfigMatcher>(kind));
}

TypeMatcher TypeMatcher::anyOf(std::initializer_list<TypeMatcher> matchers) {
  std::vector<TypeMatcher> matcherVec(matchers.begin(), matchers.end());
  return TypeMatcher(std::make_unique<AnyOfMatcher>(std::move(matcherVec)));
}

TypeMatcher TypeMatcher::sameAsTypeOf(unsigned argIdx) {
  return TypeMatcher(std::make_unique<SameAsTypeOfMatcher>(argIdx));
}

TypeMatcher TypeMatcher::sameAsTypeOfCoercible(unsigned argIdx) {
  return TypeMatcher(
      std::make_unique<SameAsTypeOfMatcher>(argIdx, /*coercion=*/true));
}

TypeMatcher TypeMatcher::sameAsValueOf(unsigned argIdx) {
  return TypeMatcher(std::make_unique<SameAsValueOfMatcher>(argIdx));
}

TypeMatcher TypeMatcher::sameAsValueOfCoercible(unsigned argIdx) {
  return TypeMatcher(
      std::make_unique<SameAsValueOfMatcher>(argIdx, /*coercion=*/true));
}

TypeMatcher TypeMatcher::ifComptimeKnown(TypeMatcher comptimeMatcher,
                                         TypeMatcher runtimeMatcher) {
  return TypeMatcher(std::make_unique<ComptimeOrRuntimeMatcher>(
      std::move(comptimeMatcher), std::move(runtimeMatcher)));
}

TypeMatcher TypeMatcher::anyInt() {
  return anyOf({ofClass<IntegerType>(), ofClass<ComptimeIntType>()});
}

TypeMatcher TypeMatcher::anyFloat() {
  return anyOf({ofClass<FloatType>(), ofClass<ComptimeFloatType>()});
}

//===----------------------------------------------------------------------===//
// TypeMatcher Modifiers
//===----------------------------------------------------------------------===//

TypeMatcher TypeMatcher::asVariadic(unsigned minArgs) const {
  TypeMatcher copy(*this);
  copy.minCount = minArgs;
  copy.maxCount = UINT_MAX;
  return copy;
}

TypeMatcher TypeMatcher::asOptional() const {
  TypeMatcher copy(*this);
  copy.minCount = 0;
  copy.maxCount = 1;
  return copy;
}

TypeMatcher TypeMatcher::preserveLValue() const {
  TypeMatcher copy(*this);
  copy.preserveLValue_ = true;
  return copy;
}

//===----------------------------------------------------------------------===//
// TypeSpec Concrete Implementations
//===----------------------------------------------------------------------===//

namespace {

/// Returns a fixed type.
struct ExactTypeSpec : TypeSpecBase<ExactTypeSpec> {
  Type type;

  explicit ExactTypeSpec(Type t) : type(t) {}

  Type getType(ArrayRef<TypedAttr>) const override { return type; }
};

/// Returns the type of a previous argument.
struct SameAsTypeOfSpec : TypeSpecBase<SameAsTypeOfSpec> {
  unsigned argIdx;

  explicit SameAsTypeOfSpec(unsigned idx) : argIdx(idx) {}

  Type getType(ArrayRef<TypedAttr> args) const override {
    return args[argIdx].getType();
  }
};

/// Returns the type specified by the value of a previous argument (TypeAttr).
struct SameAsValueOfSpec : TypeSpecBase<SameAsValueOfSpec> {
  unsigned argIdx;

  explicit SameAsValueOfSpec(unsigned idx) : argIdx(idx) {}

  Type getType(ArrayRef<TypedAttr> args) const override {
    auto typeAttr = cast<csl::TypeAttr>(args[argIdx]);
    return typeAttr.getValue();
  }
};

/// Returns a type computed by a custom function.
struct FuncTypeSpec : TypeSpecBase<FuncTypeSpec> {
  std::function<Type(ArrayRef<TypedAttr>)> func;

  explicit FuncTypeSpec(std::function<Type(ArrayRef<TypedAttr>)> f)
      : func(std::move(f)) {}

  Type getType(ArrayRef<TypedAttr> args) const override { return func(args); }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// TypeSpec Implementation
//===----------------------------------------------------------------------===//

TypeSpec::TypeSpec(std::unique_ptr<TypeSpecConcept> impl)
    : impl(std::move(impl)) {}

TypeSpec::TypeSpec(const TypeSpec &other)
    : impl(other.impl ? other.impl->clone() : nullptr),
      preserveLValue_(other.preserveLValue_) {}

TypeSpec::TypeSpec(const TypeMatcher &matcher) {
  auto exactType = matcher.getExactType();
  assert(exactType && "Can only convert exact-type TypeMatcher to TypeSpec");
  impl = std::make_unique<ExactTypeSpec>(*exactType);
}

TypeSpec &TypeSpec::operator=(const TypeSpec &other) {
  if (this != &other) {
    impl = other.impl ? other.impl->clone() : nullptr;
    preserveLValue_ = other.preserveLValue_;
  }
  return *this;
}

Type TypeSpec::getType(ArrayRef<TypedAttr> args) const {
  assert(impl && "TypeSpec has no implementation");
  Type result = impl->getType(args);
  if (!preserveLValue_)
    if (auto lvalueType = dyn_cast<LValueType>(result))
      result = lvalueType.getValueType();
  return result;
}

TypeSpec TypeSpec::exact(Type t) {
  return TypeSpec(std::make_unique<ExactTypeSpec>(t));
}

TypeSpec TypeSpec::sameAsTypeOf(unsigned argIdx) {
  return TypeSpec(std::make_unique<SameAsTypeOfSpec>(argIdx));
}

TypeSpec TypeSpec::sameAsValueOf(unsigned argIdx) {
  return TypeSpec(std::make_unique<SameAsValueOfSpec>(argIdx));
}

TypeSpec TypeSpec::fromFunc(std::function<Type(ArrayRef<TypedAttr>)> func) {
  return TypeSpec(std::make_unique<FuncTypeSpec>(std::move(func)));
}

TypeSpec TypeSpec::preserveLValue() const {
  TypeSpec copy(*this);
  copy.preserveLValue_ = true;
  return copy;
}

//===----------------------------------------------------------------------===//
// SignatureMatchResult Implementation
//===----------------------------------------------------------------------===//

SignatureMatchResult
SignatureMatchResult::success(SignatureSpec spec, Type resultType,
                              ArrayRef<Type> paramTypes,
                              ArrayRef<std::pair<unsigned, Type>> casts) {
  SignatureMatchResult result;
  result.isValid = true;
  result.inferredReturnType = resultType;
  result.spec = std::move(spec);
  result.resolvedParamTypes.assign(paramTypes.begin(), paramTypes.end());
  result.requiredCasts.assign(casts.begin(), casts.end());
  return result;
}

SignatureMatchResult SignatureMatchResult::failure(StringRef message) {
  SignatureMatchResult result;
  result.isValid = false;
  result.errorMessage = message.str();
  return result;
}

//===----------------------------------------------------------------------===//
// BuiltinDescriptor Implementation
//===----------------------------------------------------------------------===//

StringRef BuiltinDescriptor::getName() const {
  return stringifyBuiltinKind(kind);
}

/// Build a descriptor for a specific builtin kind.
static std::optional<BuiltinDescriptor> buildDescriptor(BuiltinKind kind,
                                                        MLIRContext *ctx) {
  // DSD type matchers
  TypeMatcher mem1dDsd = TypeMatcher::exact(Mem1dDsdType::get(ctx));
  TypeMatcher mem4dDsd = TypeMatcher::exact(Mem4dDsdType::get(ctx));
  TypeMatcher fabinDsd = TypeMatcher::exact(FabinDsdType::get(ctx));
  TypeMatcher faboutDsd = TypeMatcher::exact(FaboutDsdType::get(ctx));
  TypeMatcher fifoDsd = TypeMatcher::exact(FifoDsdType::get(ctx));
  TypeMatcher anyMemDsd = TypeMatcher::anyOf({mem1dDsd, mem4dDsd});
  TypeMatcher anyFabDsd = TypeMatcher::anyOf({fabinDsd, faboutDsd});
  TypeMatcher anyDsd =
      TypeMatcher::anyOf({mem1dDsd, mem4dDsd, fabinDsd, faboutDsd, fifoDsd});

  // Used by @get_* builtins - accept any int if comptime-known, else coerce to
  // u16
  Type u16Type = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  TypeMatcher comptimeKnownIntOrU16 = TypeMatcher::ifComptimeKnown(
      TypeMatcher::anyInt(), TypeMatcher::coercesTo(u16Type));

  // Numeric types allow for type coercion (i.e. accept comptime_int as well)
  TypeMatcher i8 =
      TypeMatcher::coercesTo(IntegerType::get(ctx, 8, IntegerType::Signed));
  TypeMatcher u8 =
      TypeMatcher::coercesTo(IntegerType::get(ctx, 8, IntegerType::Unsigned));
  TypeMatcher i16 =
      TypeMatcher::coercesTo(IntegerType::get(ctx, 16, IntegerType::Signed));
  TypeMatcher u16 = TypeMatcher::coercesTo(u16Type);
  TypeMatcher i32 =
      TypeMatcher::coercesTo(IntegerType::get(ctx, 32, IntegerType::Signed));
  TypeMatcher u32 =
      TypeMatcher::coercesTo(IntegerType::get(ctx, 32, IntegerType::Unsigned));
  TypeMatcher f16 = TypeMatcher::coercesTo(Float16Type::get(ctx));
  TypeMatcher f32 = TypeMatcher::coercesTo(Float32Type::get(ctx));

  // Mutable pointer types (both array = [*] i32 and non-array = *i32). Allow
  // for type coercion.
  auto constructAnyPtrMatchers = [&](Type baseType) {
    return TypeMatcher::anyOf(
        {TypeMatcher::coercesTo(PointerType::get(baseType, false, false)),
         TypeMatcher::coercesTo(PointerType::get(baseType, true, true))});
  };
  TypeMatcher anyI8Ptr =
      constructAnyPtrMatchers(IntegerType::get(ctx, 8, IntegerType::Signed));
  TypeMatcher anyU8Ptr =
      constructAnyPtrMatchers(IntegerType::get(ctx, 8, IntegerType::Unsigned));
  TypeMatcher anyI16Ptr =
      constructAnyPtrMatchers(IntegerType::get(ctx, 16, IntegerType::Signed));
  TypeMatcher anyU16Ptr =
      constructAnyPtrMatchers(IntegerType::get(ctx, 16, IntegerType::Unsigned));
  TypeMatcher anyI32Ptr =
      constructAnyPtrMatchers(IntegerType::get(ctx, 32, IntegerType::Signed));
  TypeMatcher anyU32Ptr =
      constructAnyPtrMatchers(IntegerType::get(ctx, 32, IntegerType::Unsigned));
  TypeMatcher anyF16Ptr = constructAnyPtrMatchers(Float16Type::get(ctx));
  TypeMatcher anyF32Ptr = constructAnyPtrMatchers(Float32Type::get(ctx));

  TypeMatcher color = TypeMatcher::exact(ColorType::get(ctx));
  TypeMatcher localTaskId = TypeMatcher::exact(LocalTaskIdType::get(ctx));
  TypeMatcher dataTaskId = TypeMatcher::exact(DataTaskIdType::get(ctx));
  TypeMatcher controlTaskId = TypeMatcher::exact(ControlTaskIdType::get(ctx));
  TypeMatcher utId = TypeMatcher::exact(UtIdType::get(ctx));
  TypeMatcher boolean = TypeMatcher::exact(IntegerType::get(ctx, 1));
  TypeMatcher type = TypeMatcher::exact(TypeType::get(ctx));
  TypeMatcher comptimeString = TypeMatcher::exact(ComptimeStringType::get(ctx));
  TypeMatcher comptimeInt = TypeMatcher::coercesTo(ComptimeIntType::get(ctx));
  TypeMatcher comptimeFloat =
      TypeMatcher::coercesTo(ComptimeFloatType::get(ctx));
  TypeMatcher voidTy = TypeMatcher::exact(VoidType::get(ctx));
  auto ui8Ty = IntegerType::get(ctx, 8, IntegerType::Unsigned);
  TypeMatcher manyPtrMutableU8 = TypeMatcher::exact(
      PointerType::get(ui8Ty, /*isConst=*/false, /*isArray=*/true));

  TypeMatcher anyFunction = TypeMatcher::ofClass<FunctionType>();

  // Range type matcher using type class
  TypeMatcher anyRange = TypeMatcher::ofClass<RangeType>();

  // Struct type matcher using type class
  TypeMatcher anyStruct = TypeMatcher::ofClass<StructType>();

  auto makeStructType = [ctx](ArrayRef<StringRef> names,
                              ArrayRef<Type> fieldTypes) -> StructType {
    SmallVector<StringAttr> nameAttrs;
    llvm::transform(
        names, std::back_inserter(nameAttrs),
        [ctx](StringRef name) { return StringAttr::get(ctx, name); });
    return StructType::get(ctx, nameAttrs, fieldTypes);
  };
  Type rectangleExtentType =
      makeStructType({"width", "height"}, {u16Type, u16Type});
  Type rectangleType = makeStructType(
      {"width", "height", "fabric", "offsets"},
      {u16Type, u16Type, rectangleExtentType, rectangleExtentType});
  TypeMatcher rectangleStruct = TypeMatcher::exact(rectangleType);

  // DSR type matchers
  TypeMatcher dsrDest = TypeMatcher::exact(DsrDestType::get(ctx));
  TypeMatcher dsrSrc0 = TypeMatcher::exact(DsrSrc0Type::get(ctx));
  TypeMatcher dsrSrc1 = TypeMatcher::exact(DsrSrc1Type::get(ctx));
  TypeMatcher dsrFifoDest = TypeMatcher::exact(DsrFifoDestType::get(ctx));
  TypeMatcher dsrFifoSrc1 = TypeMatcher::exact(DsrFifoSrc1Type::get(ctx));
  TypeMatcher xdsrType = TypeMatcher::exact(XdsrType::get(ctx));
  TypeMatcher srType = TypeMatcher::exact(SrType::get(ctx));
  TypeMatcher anyDsr =
      TypeMatcher::anyOf({dsrDest, dsrSrc0, dsrSrc1, dsrFifoDest, dsrFifoSrc1});

  TypeMatcher config = TypeMatcher::configOf(kind);
  TypeMatcher dsdConfig = config.asOptional();

  // TODO: This could be more precise
  TypeMatcher dsd = TypeMatcher::anyOf({anyDsd, anyDsr, xdsrType});

  // Queue type matchers
  TypeMatcher inputQueue = TypeMatcher::exact(InputQueueType::get(ctx));
  TypeMatcher outputQueue = TypeMatcher::exact(OutputQueueType::get(ctx));

  // LValue type matcher (for builtins expecting lvalue arguments)
  // Uses preserveLValue() to prevent automatic unwrapping during matching
  TypeMatcher anyLValue = TypeMatcher::ofClass<LValueType>().preserveLValue();

  // Matcher for pointer types (and array types, as they can coerce to pointers)
  TypeMatcher anyPtr = TypeMatcher::anyOf(
      {TypeMatcher::ofClass<PointerType>(), TypeMatcher::ofClass<ArrayType>()});

  TypeMatcher anyFuture = TypeMatcher::ofClass<FutureType>();

  switch (kind) {
    //===------------------------------------------------------------------===//
    // OurComptimeSideEffects - Side effects at our comptime (must evaluate)
    //===------------------------------------------------------------------===//

  case BuiltinKind::ComptimePrint:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeSideEffects,
        {SignatureSpec{{TypeMatcher::any().asVariadic()}, voidTy}}};

  case BuiltinKind::ComptimeAssert:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeSideEffects,
        {SignatureSpec{{boolean}, voidTy},
         SignatureSpec{{boolean, TypeMatcher::any()}, voidTy}},
    };

  case BuiltinKind::ImportModule:
    // Return type depends on the imported module - handled specially during
    // evaluation by ImportModuleEvaluator::inferResultType. Return a
    // placeholder ImportedModuleType here (instance ID 0 is a placeholder).
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::OurComptimeSideEffects,
        // Actual return type is determined during comptime eval.
        // First signature: just filename
        // Second signature: filename + compile-time-known anonymous struct for
        // parameter bindings.
        {SignatureSpec{{comptimeString}, voidTy},
         SignatureSpec{{comptimeString, anyStruct}, voidTy}}};

    //===------------------------------------------------------------------===//
    // FinalComptimeSideEffects - Side effects at final CSL compiler's comptime
    //===------------------------------------------------------------------===//

  case BuiltinKind::SetRectangle:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{comptimeInt, comptimeInt}, voidTy}}};

  case BuiltinKind::SetTileCode:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{comptimeInt, comptimeInt, comptimeString, anyStruct},
                       voidTy}}};

  case BuiltinKind::SetColorConfig:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{comptimeInt, comptimeInt, color, anyStruct}, voidTy}}};

  case BuiltinKind::ExportSymbol:
    // Custom evaluator handles validation - accepts any type
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{anyFunction, comptimeString.asOptional()}, voidTy},
         SignatureSpec{{anyLValue, comptimeString.asOptional()}, voidTy}}};

  case BuiltinKind::SetLocalColorConfig:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::ComptimeOnly,
                             BuiltinSideEffects::FinalComptimeSideEffects,
                             {SignatureSpec{{color, anyStruct}, voidTy}}};

  case BuiltinKind::InitializeQueue:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::ComptimeOnly,
                             BuiltinSideEffects::FinalComptimeSideEffects,
                             {SignatureSpec{{inputQueue, config}, voidTy},
                              SignatureSpec{{outputQueue, config}, voidTy}}};

  case BuiltinKind::InternalRpc:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        // @__internal_rpc(fn: fn([*]const u32) void) void;
        {SignatureSpec{{TypeMatcher::exact(FunctionType::get(
                           ctx,
                           {PointerType::get(
                               IntegerType::get(ctx, 32, IntegerType::Unsigned),
                               true, true)},
                           {VoidType::get(ctx)}))},
                       voidTy},
         SignatureSpec{{TypeMatcher::exact(FunctionType::get(
                           ctx,
                           {PointerType::get(
                               IntegerType::get(ctx, 32, IntegerType::Unsigned),
                               true, true)},
                           {}))},
                       voidTy}}};

  case BuiltinKind::InternalPromoteToConstGlobal:
    // @__internal_promote_to_const_global(s: comptime_string) *const [N]u8
    // Actual return type is determined during comptime eval.
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::ComptimeOnly,
                             BuiltinSideEffects::OurComptimeSideEffects,
                             {SignatureSpec{{comptimeString}, voidTy}}};

  case BuiltinKind::SetTeardownHandler:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        // @set_teardown_handler(fn: fn() void, color) void;
        {SignatureSpec{{TypeMatcher::exact(
                            FunctionType::get(ctx, {}, {VoidType::get(ctx)})),
                        color},
                       voidTy},
         SignatureSpec{
             {TypeMatcher::exact(FunctionType::get(ctx, {}, {})), color},
             voidTy}}};

  case BuiltinKind::ExportName:
    // @export_name(name: comptime_string, type: type) void
    // @export_name(name: comptime_string, type: type, isMutable: bool) void
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{comptimeString, type}, voidTy},
         SignatureSpec{{comptimeString, type, boolean}, voidTy}}};

  case BuiltinKind::SetControlTaskTable:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{}, voidTy}, SignatureSpec{{anyStruct}, voidTy}}};

  case BuiltinKind::SetEmptyQueueHandler:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::FinalComptimeSideEffects,
        {SignatureSpec{{anyFunction, inputQueue}, voidTy},
         SignatureSpec{{anyFunction, outputQueue}, voidTy}}};

  case BuiltinKind::Rpc:
    // @rpc(task_id) -> void
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::ComptimeOnly,
                             BuiltinSideEffects::FinalComptimeSideEffects,
                             {SignatureSpec{{dataTaskId}, voidTy}}};

    //===------------------------------------------------------------------===//
    //  Pure - No side effects (may evaluate, strip if unused)
    //===------------------------------------------------------------------===//
  case BuiltinKind::TypeOf:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{TypeMatcher::any()}, type}},
    }
        .withTypeOnlyArgs({0});

  case BuiltinKind::FutureOf:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyFunction}, type}},
    };

  case BuiltinKind::Await: {
    // @await(future) or @await(future, .{ .offload = true })
    // -> result_type (extracted from FutureType)
    auto awaitReturnType = [](ArrayRef<TypedAttr> args) -> Type {
      if (args.size() < 1)
        return VoidType::get(args[0].getContext());
      auto futureTy = dyn_cast<FutureType>(args[0].getType());
      if (!futureTy)
        return VoidType::get(args[0].getContext());
      Type resultTy = futureTy.getResultType();
      if (!resultTy)
        return VoidType::get(args[0].getContext());
      return resultTy;
    };
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyFuture}, awaitReturnType},
         SignatureSpec{{anyFuture, anyStruct}, awaitReturnType}},
    };
  }

  case BuiltinKind::AwaitParallel: {
    // @await_parallel(future1, future2, ...) -> tuple of the callee results,
    // with a void field for each callee without one.
    auto awaitParallelReturnType = [](ArrayRef<TypedAttr> args) -> Type {
      SmallVector<Type> resultTypes;
      for (TypedAttr arg : args) {
        auto futureTy = cast<FutureType>(arg.getType());
        Type resultTy = futureTy.getResultType();
        resultTypes.push_back(resultTy ? resultTy
                                       : VoidType::get(arg.getContext()));
      }
      return StructType::get(args[0].getContext(), resultTypes);
    };
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyFuture.asVariadic(2)}, awaitParallelReturnType}},
    };
  }

  case BuiltinKind::CurrentTask:
    // @current_task() -> local_task_id
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{}, localTaskId}},
    };

  case BuiltinKind::CoroRuntimeInit:
    // @coro_runtime_init() -> void
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{}, voidTy}},
    };

  case BuiltinKind::FrameAddress:
    // @frame_address() -> [*]u8
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{}, manyPtrMutableU8}},
    };

  case BuiltinKind::IsSameType:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{type, type}, boolean}}}
        .withComptimeArgs({0, 1});

  case BuiltinKind::IsArch:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{comptimeString}, boolean}}}
        .withComptimeArgs({0});

  case BuiltinKind::IsComptime:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{}, boolean}}};

  case BuiltinKind::Fp16:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{}, type}}};

  case BuiltinKind::Strlen:
    // @strlen(str) -> comptime_int
    // str: comptime_string
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{comptimeString}, comptimeInt}}};

  case BuiltinKind::Strcat:
    // @strcat(str1, str2, ..., strN) -> comptime_string
    // All arguments must be comptime_string, variadic (0 or more args)
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeString.asVariadic()}, comptimeString}}};

  case BuiltinKind::GetStringFromByte:
    // @get_string_from_byte(byte) -> comptime_string
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{comptimeInt}, comptimeString}}};

  case BuiltinKind::Dimensions:
    // @dimensions(array_type) -> [rank]u32
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type},
                       [](ArrayRef<TypedAttr> args) -> Type {
                         auto typeAttr = dyn_cast<csl::TypeAttr>(args[0]);
                         if (!typeAttr)
                           return Type{};
                         auto arrType =
                             dyn_cast<ArrayType>(typeAttr.getValue());
                         if (!arrType)
                           return Type{};
                         int64_t rank = arrType.getRank();
                         Type u32Ty = IntegerType::get(args[0].getContext(), 32,
                                                       IntegerType::Unsigned);
                         return ArrayType::get({rank}, u32Ty);
                       }}}}
        .withComptimeArgs({0});

  case BuiltinKind::ElementCount:
    // @element_count(array_type) -> u32
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{type}, u32}}}
        .withComptimeArgs({0});

  case BuiltinKind::ElementType:
    // @element_type(array_type) -> type
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{type}, type}}}
        .withComptimeArgs({0});

  case BuiltinKind::Rank:
    // @rank(array_type) -> u16
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{type}, u16}}}
        .withComptimeArgs({0});

  case BuiltinKind::GetArray:
    // @get_array(string) -> [@strlen(string)]u8
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeString}, [](ArrayRef<TypedAttr> args) -> Type {
                         auto strAttr = dyn_cast<ComptimeStringAttr>(args[0]);
                         if (!strAttr)
                           return Type{};
                         int64_t len = strAttr.getValue().size();
                         Type u8Ty = IntegerType::get(args[0].getContext(), 8,
                                                      IntegerType::Unsigned);
                         return ArrayType::get({len}, u8Ty);
                       }}}};

  case BuiltinKind::ImportComptimeValue:
    // @import_comptime_value(type, filename) -> value of type
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, comptimeString}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

  case BuiltinKind::Ptrcast:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, anyPtr}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

  case BuiltinKind::Bitcast:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, TypeMatcher::any()}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});
  case BuiltinKind::GetFilterId:
    // TODO: Builtin is actually allowed at comptime as well!
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{color}, u16}}};
  case BuiltinKind::GetInt:
    // @get_int(enum_value) -> underlying_int_type
    // Custom evaluator handles type inference since return type depends on
    // argument
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{
            {TypeMatcher::any()}, [](ArrayRef<TypedAttr> args) -> Type {
              Type valueType = args[0].getType();
              if (auto enumType = dyn_cast<EnumType>(valueType)) {
                return enumType.getUnderlyingType();
              } else if (auto intType = dyn_cast<IntegerType>(valueType)) {
                // Allow passing integer directly
                return intType;
              } else {
                // All other types return u16
                return IntegerType::get(args[0].getContext(), 16,
                                        IntegerType::Unsigned);
              }
            }}}};

  case BuiltinKind::GetColor:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{comptimeKnownIntOrU16}, color}}};

  case BuiltinKind::GetInputQueue:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeKnownIntOrU16}, inputQueue}}};

  case BuiltinKind::GetOutputQueue:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeKnownIntOrU16}, outputQueue}}};

  case BuiltinKind::GetLocalTaskId:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeKnownIntOrU16}, localTaskId}}};

  case BuiltinKind::GetDataTaskId:
    // WSE-2: takes color, WSE-3: takes input_queue
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{TypeMatcher::anyOf({color, inputQueue})}, dataTaskId}}};

  case BuiltinKind::GetControlTaskId:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeKnownIntOrU16}, controlTaskId}}};

  case BuiltinKind::GetUtId:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{comptimeKnownIntOrU16}, utId}}};

  case BuiltinKind::HasField:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyStruct, comptimeString}, boolean}}};

  case BuiltinKind::Field: {
    // @field(base, field_name) -> field value (or lvalue to field)
    // Two specs: one for lvalue<struct> base, one for plain struct base.

    // Helper to look up field type from a StructType by name, handling tuples.
    auto lookupFieldType = [](StructType structType,
                              StringRef fieldName) -> Type {
      Type fieldType = structType.getFieldType(fieldName);
      if (fieldType)
        return fieldType;
      if (structType.isTuple()) {
        unsigned index;
        if (!fieldName.getAsInteger(10, index) &&
            index < structType.getNumFields())
          return structType.getFieldType(index);
      }
      return Type();
    };

    // Spec 1: lvalue<struct> → lvalue<field_type>
    // anyLValue already has preserveLValue(), so it sees the raw lvalue type.
    auto lvalueFieldRetType =
        TypeSpec::fromFunc([lookupFieldType](ArrayRef<TypedAttr> args) -> Type {
          auto fieldNameAttr = dyn_cast<ComptimeStringAttr>(args[1]);
          if (!fieldNameAttr)
            return Type{};
          auto lvalueType = dyn_cast<LValueType>(args[0].getType());
          if (!lvalueType)
            return Type{};
          auto structType = dyn_cast<StructType>(lvalueType.getValueType());
          if (!structType)
            return Type{};
          Type fieldType =
              lookupFieldType(structType, fieldNameAttr.getValue());
          if (!fieldType)
            return Type{};
          return LValueType::get(fieldType, lvalueType.getIsConst());
        }).preserveLValue();

    // Spec 2: struct → field_type (rvalue)
    // preserveLValue on matcher ensures lvalue<struct> inputs are NOT matched
    // here (they go to Spec 1 instead).
    auto rvalueFieldRetType =
        TypeSpec::fromFunc([lookupFieldType](ArrayRef<TypedAttr> args) -> Type {
          auto fieldNameAttr = dyn_cast<ComptimeStringAttr>(args[1]);
          if (!fieldNameAttr)
            return Type{};
          StringRef fieldName = fieldNameAttr.getValue();
          Type argType = args[0].getType();
          if (auto structType = dyn_cast<StructType>(argType))
            return lookupFieldType(structType, fieldName);
          return Type{};
        });

    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyLValue, comptimeString}, lvalueFieldRetType},
         SignatureSpec{{anyStruct.preserveLValue(), comptimeString},
                       rvalueFieldRetType}}};
  }

  case BuiltinKind::Range:
    // (type, value of type, value of type, value of type) -> range of type
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, TypeMatcher::sameAsValueOfCoercible(0),
                        TypeMatcher::sameAsValueOfCoercible(0),
                        TypeMatcher::sameAsValueOfCoercible(0)},
                       [](ArrayRef<TypedAttr> args) {
                         Type elemType =
                             cast<csl::TypeAttr>(args[0]).getValue();
                         return csl::RangeType::get(elemType);
                       }},
         SignatureSpec{{type, TypeMatcher::sameAsValueOfCoercible(0)},
                       [](ArrayRef<TypedAttr> args) {
                         Type elemType =
                             cast<csl::TypeAttr>(args[0]).getValue();
                         return csl::RangeType::get(elemType);
                       }}}}
        .withComptimeArgs({0});
  case BuiltinKind::RangeStart:
  case BuiltinKind::RangeStep:
  case BuiltinKind::RangeStop:
    // (range of type) -> value of type
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyRange}, [](ArrayRef<TypedAttr> args) {
                         auto rangeType =
                             cast<csl::RangeType>(args[0].getType());
                         return rangeType.getElementType();
                       }}}};

  case BuiltinKind::GetDsd:
    // Maybe AnyValueOf(mem1d_dsd, mem4d_dsd, fabin_dsd, fabout_dsd) for first
    // arg somehow? Just add additional verifier method?
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, config}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

  case BuiltinKind::IncrementDsdOffset:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyMemDsd, i16, type}, TypeSpec::sameAsTypeOf(0)}}}
        .withComptimeArgs({2});

  case BuiltinKind::SetDsdBaseAddr:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyMemDsd, anyPtr}, TypeSpec::sameAsTypeOf(0)},
         SignatureSpec{{anyMemDsd, TypeMatcher::anyInt()},
                       TypeSpec::sameAsTypeOf(0)}}};

  case BuiltinKind::SetDsdLength:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyDsd, u16}, TypeSpec::sameAsTypeOf(0)}}};

  case BuiltinKind::SetDsdStride:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{mem1dDsd, i8}, TypeSpec::sameAsTypeOf(0)}}};

  case BuiltinKind::GetTensorPtr:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{TypeMatcher::anyInt()},
                       PointerType::get(u16Type, false, true)}}};

  case BuiltinKind::GetDsr:
    // @get_dsr(dsr_type, dsr_id) - create DSR from type and integer ID
    // @get_dsr(fifo_dsr_type, non_fifo_dsr) - convert non-FIFO DSR to FIFO DSR
    // Custom evaluator handles the second overload
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{type, comptimeKnownIntOrU16},
                                            TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

  case BuiltinKind::GetXdsr:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{comptimeKnownIntOrU16}, xdsrType}}};

  case BuiltinKind::GetSr:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{comptimeKnownIntOrU16}, srType}}};

  case BuiltinKind::LoadToDsr:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects,
        {SignatureSpec{{anyDsr, dsd}, voidTy},
         SignatureSpec{{anyDsr, dsd, config}, voidTy}}};

  case BuiltinKind::LoadToDsrXdsrSr:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects,
        {SignatureSpec{{anyDsr, xdsrType, anyStruct, mem4dDsd, dsdConfig},
                       voidTy}}}
        .withComptimeArgs({0, 1, 2});

  case BuiltinKind::GetRectangle:
    // The value is compile-time known, but has an ordinary concrete anonymous
    // struct type in SDK 2.10 and later.
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{}, rectangleStruct}}};

  case BuiltinKind::AllocateFifo: {
    TypeSpec fifoDsd = TypeSpec::exact(FifoDsdType::get(ctx));
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::ComptimeOnly,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{anyLValue, config.asOptional()}, fifoDsd}}};
  }

  case BuiltinKind::Random16:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{}, u16}}};

  case BuiltinKind::GetSymbolId:
    // @get_symbol_id(symbol) -> u16
    // symbol is a reference to an advertised global device symbol
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{TypeMatcher::any()}, u16}}};

  case BuiltinKind::GetSymbolValue:
    // @get_symbol_value(type, id) -> value of type
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, u16}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

  case BuiltinKind::HasExportedTensors:
    // @has_exported_tensors() -> bool
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOrComptime,
                             BuiltinSideEffects::Pure,
                             {SignatureSpec{{}, boolean}}};

    //===------------------------------------------------------------------===//
    // RuntimeSideEffects - Side effects at runtime (must keep)
    //===------------------------------------------------------------------===//
  case BuiltinKind::GetConfig:
  case BuiltinKind::GetConfigUnchecked:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{u16}, u16}, SignatureSpec{{u16, anyStruct}, u16}}};
  case BuiltinKind::SetConfig:
  case BuiltinKind::SetConfigUnchecked:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {SignatureSpec{{u16, u16}, voidTy},
                              SignatureSpec{{u16, u16, anyStruct}, voidTy}}};

  // 16-bit DSD unary ops (count leading/trailing zeros, popcount, int-to-float)
  case BuiltinKind::Clz:
  case BuiltinKind::Ctz:
  case BuiltinKind::Popcnt:
  case BuiltinKind::Xp162fh:
  case BuiltinKind::Xp162fs:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {
                                 SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
                                 SignatureSpec{{dsd, i16, dsdConfig}, boolean},
                                 SignatureSpec{{dsd, u16, dsdConfig}, boolean},
                             }};

  // 16-bit DSD arithmetic ops
  case BuiltinKind::Add16:
  case BuiltinKind::Addc16:
  case BuiltinKind::Sub16:
  case BuiltinKind::And16:
  case BuiltinKind::Or16:
  case BuiltinKind::Xor16:
  case BuiltinKind::Sar16:
  case BuiltinKind::Sll16:
  case BuiltinKind::Slr16:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, i16, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, u16, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, dsd, i16, dsdConfig}, boolean},
            SignatureSpec{{dsd, dsd, u16, dsdConfig}, boolean},
        }};

    // DSD half-half float ops
  case BuiltinKind::Faddh:
  case BuiltinKind::Fsubh:
  case BuiltinKind::Fmulh:
  case BuiltinKind::Fmaxh:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f16, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, dsd, f16, dsdConfig}, boolean},
            SignatureSpec{{anyF16Ptr, f16, dsd, dsdConfig}, boolean},
        }};

    // DSD single-single float ops
  case BuiltinKind::Fadds:
  case BuiltinKind::Fsubs:
  case BuiltinKind::Fmuls:
  case BuiltinKind::Fmaxs:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f32, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, dsd, f32, dsdConfig}, boolean},
            SignatureSpec{{anyF32Ptr, f32, dsd, dsdConfig}, boolean},
        }};

    // DSD single-half float ops
  case BuiltinKind::Faddhs:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f16, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, dsd, f16, dsdConfig}, boolean},
            SignatureSpec{{anyF32Ptr, f32, dsd, dsdConfig}, boolean},
        }};

  // 16-bit DSD move ops
  case BuiltinKind::Mov16:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{anyI16Ptr, dsd, dsdConfig}, boolean},
            SignatureSpec{{anyU16Ptr, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, i16, dsdConfig}, boolean},
            SignatureSpec{{dsd, u16, dsdConfig}, boolean},
        }};

  // 32-bit DSD move ops
  case BuiltinKind::Mov32:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{anyI32Ptr, dsd, dsdConfig}, boolean},
            SignatureSpec{{anyU32Ptr, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, i32, dsdConfig}, boolean},
            SignatureSpec{{dsd, u32, dsdConfig}, boolean},
        }};

    // 16-bit floating point multiply-add.
  case BuiltinKind::Fmach:
  case BuiltinKind::Fmachs:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{dsd, dsd, dsd, f16, dsdConfig}, boolean}}};

  // 32-bit floating point multiply-add.
  case BuiltinKind::Fmacs:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{dsd, dsd, dsd, f32, dsdConfig}, boolean}}};

    // 32-bit float DSD move ops
  case BuiltinKind::Fmovs:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{anyF32Ptr, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f32, dsdConfig}, boolean},
        }};

    // 16-bit float DSD move ops
  case BuiltinKind::Fmovh:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{anyF16Ptr, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f16, dsdConfig}, boolean},
        }};
    // DSD unary f32 ops: negate, absolute value, convert single to half
  case BuiltinKind::Fnegs:
  case BuiltinKind::Fabss:
  case BuiltinKind::Fs2h:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {
                                 SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
                                 SignatureSpec{{dsd, f32, dsdConfig}, boolean},
                             }};

    // DSD unary f16 ops: negate, absolute value, convert half to single
  case BuiltinKind::Fnegh:
  case BuiltinKind::Fabsh:
  case BuiltinKind::Fh2s:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {
                                 SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
                                 SignatureSpec{{dsd, f16, dsdConfig}, boolean},
                             }};

    // DSD float convert half to 16-bit integer
  case BuiltinKind::Fh2xp16:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f16, dsdConfig}, boolean},
            SignatureSpec{{anyI16Ptr, f16, dsdConfig}, boolean},
        }};

    // DSD float normalize (16-bit)
  case BuiltinKind::Fnormh:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyF16Ptr, f16, dsdConfig}, boolean}}};

    // DSD float normalize (32-bit)
  case BuiltinKind::Fnorms:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyF32Ptr, f32, dsdConfig}, boolean}}};

    // DSD float convert single to 16-bit integer
  case BuiltinKind::Fs2xp16:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {
            SignatureSpec{{dsd, dsd, dsdConfig}, boolean},
            SignatureSpec{{dsd, f32, dsdConfig}, boolean},
            SignatureSpec{{anyI16Ptr, f32, dsdConfig}, boolean},
        }};

    // 16-bit float scale (multiply by 2^i16)
  case BuiltinKind::Fscaleh:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyF16Ptr, f16, i16, dsdConfig}, boolean}}};

    // 32-bit float scale (multiply by 2^i16)
  case BuiltinKind::Fscales:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyF32Ptr, f32, i16, dsdConfig}, boolean}}};

    // DSD filter (drop data wavelets until N control wavelets arrive)
  case BuiltinKind::Dfilt:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {SignatureSpec{{dsd, config}, voidTy}}};

  case BuiltinKind::SetActivePrng:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {SignatureSpec{{u16}, voidTy}}};

  case BuiltinKind::QueueFlush:
    return BuiltinDescriptor{kind,
                             BuiltinAllowedContext::RuntimeOnly,
                             BuiltinSideEffects::RuntimeSideEffects,
                             {SignatureSpec{{inputQueue}, voidTy},
                              SignatureSpec{{outputQueue}, voidTy}}};

  case BuiltinKind::SetFifoReadLength:
  case BuiltinKind::SetFifoWriteLength:
    // @set_fifo_read_length(fifo_dsd, u16) -> fifo_dsd
    // @set_fifo_write_length(fifo_dsd, u16) -> fifo_dsd
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{fifoDsd, u16}, TypeSpec::sameAsTypeOf(0)}}};

    // @map(callback, inputs...) -> void
  case BuiltinKind::Map:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOnly,
        BuiltinSideEffects::RuntimeSideEffects,
        {SignatureSpec{{anyFunction, TypeMatcher::any().asVariadic(1)},
                       voidTy}}};

    //===------------------------------------------------------------------===//
    // FinalComptimeOrRuntimeSideEffects - Side effects at final CSL
    // compiler's comptime or at runtime (must keep)
    //===------------------------------------------------------------------===//

  case BuiltinKind::BindLocalTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects,
        {SignatureSpec{{anyFunction, localTaskId}, voidTy}}};

  case BuiltinKind::BindDataTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects,
        {SignatureSpec{{anyFunction, dataTaskId}, voidTy}}};

  case BuiltinKind::BindControlTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects,
        {SignatureSpec{{anyFunction, controlTaskId}, voidTy}}};

  case BuiltinKind::Activate:
  case BuiltinKind::Block:
  case BuiltinKind::Unblock:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::FinalComptimeOrRuntimeSideEffects,
        {SignatureSpec{{localTaskId}, voidTy},
         SignatureSpec{{dataTaskId}, voidTy}, SignatureSpec{{color}, voidTy},
         SignatureSpec{{inputQueue}, voidTy},
         SignatureSpec{{utId}, voidTy}}}; // Task side effects

    //===------------------------------------------------------------------===//
    // OurComptimeOrRuntimeSideEffects - Side effects at our comptime or at
    // runtime (may evaluate, strip if evaluated)
    //===------------------------------------------------------------------===//

  case BuiltinKind::Assert:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{boolean}, voidTy}}};

    //===------------------------------------------------------------------===//
    // Array initialization builtins
    //===------------------------------------------------------------------===//

  case BuiltinKind::Zeros:
    // @zeros(type) -> value of that type (array initialized to zeros)
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

  case BuiltinKind::Constants:
    // @constants(type, value) -> value of that type (array initialized to
    // value)
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::Pure,
        {SignatureSpec{{type, TypeMatcher::any()}, TypeSpec::sameAsValueOf(0)}}}
        .withComptimeArgs({0});

    //===------------------------------------------------------------------===//
    // Resource claim/release builtins
    //===------------------------------------------------------------------===//

  case BuiltinKind::ClaimColor:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{}, color}}};
  case BuiltinKind::ReleaseColor:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{color}, voidTy}}};

  case BuiltinKind::ClaimLocalTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{}, localTaskId}}};
  case BuiltinKind::ReleaseLocalTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{localTaskId}, voidTy}}};

  case BuiltinKind::ClaimDataTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{}, dataTaskId}}};
  case BuiltinKind::ReleaseDataTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{dataTaskId}, voidTy}}};

  case BuiltinKind::ClaimInputQueue:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{}, inputQueue}}};
  case BuiltinKind::ReleaseInputQueue:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{inputQueue}, voidTy}}};

  case BuiltinKind::ClaimOutputQueue:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{}, outputQueue}}};
  case BuiltinKind::ReleaseOutputQueue:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{outputQueue}, voidTy}}};

  case BuiltinKind::ClaimControlTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{}, controlTaskId}}};
  case BuiltinKind::ReleaseControlTask:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{controlTaskId}, voidTy}}};

  // @count_free(T) -> u16, where T is the ID type of a resource kind. It
  // reads the allocator state, so it must stay in order with the claims and
  // releases around it.
  case BuiltinKind::CountFree:
    return BuiltinDescriptor{
        kind,
        BuiltinAllowedContext::RuntimeOrComptime,
        BuiltinSideEffects::OurComptimeOrRuntimeSideEffects,
        {SignatureSpec{{type}, TypeMatcher::exact(u16Type)}}}
        .withComptimeArgs({0});

  default:
    return std::nullopt;
  }
}

const BuiltinDescriptor &BuiltinDescriptor::get(BuiltinKind kind,
                                                MLIRContext *ctx) {
  CSLDialect *dialect = ctx->getLoadedDialect<CSLDialect>();
  if (!dialect) [[unlikely]] {
    llvm_unreachable(
        "csl dialect must be loaded before bulding a builtin descriptor");
  }
  if (dialect->builtinDescriptors.empty()) [[unlikely]] {
    llvm_unreachable("must be registered before this function can be used");
  }

  assert(static_cast<unsigned>(kind) < dialect->builtinDescriptors.size() &&
         "builtin descriptor index out of bounds");

  std::optional<BuiltinDescriptor> &maybeDesc =
      dialect->builtinDescriptors[static_cast<unsigned>(kind)];

  assert(maybeDesc.has_value() &&
         "builtin descriptor for this kind was not registered");

  return *maybeDesc;
}

void CSLDialect::registerBuiltinDescriptors() {
  unsigned numBuiltins =
      static_cast<unsigned>(getMaxEnumValForBuiltinKind() + 1);
  builtinDescriptors.reserve(numBuiltins);
  for (unsigned i = 0; i < numBuiltins; ++i) {
    builtinDescriptors.push_back(
        buildDescriptor(static_cast<BuiltinKind>(i), getContext()));
  }
}

//===----------------------------------------------------------------------===//
// BuiltinSignatureMatcher Implementation
//===----------------------------------------------------------------------===//

/// Try to match argument types against a signature spec with variadic
/// support. Greedy: each variadic matcher consumes as many args as it can (up
/// to max).
static std::optional<SignatureMatchResult>
tryMatchSignature(const SignatureSpec &spec, ArrayRef<TypedAttr> argValues,
                  MLIRContext * /*ctx*/, unsigned sigIdx) {
  LLVM_DEBUG(llvm::dbgs() << "  Trying signature #" << sigIdx << ": (");
  LLVM_DEBUG({
    for (size_t i = 0; i < spec.paramMatchers.size(); ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << spec.paramMatchers[i].toString();
    }
    llvm::dbgs() << ")\n";
  });

  SmallVector<Type> resolvedTypes;
  SmallVector<std::pair<unsigned, Type>> requiredCasts;

  size_t argIdx = 0;
  size_t matcherIdx = 0;
  for (const TypeMatcher &matcher : spec.paramMatchers) {
    LLVM_DEBUG(llvm::dbgs() << "    Matcher #" << matcherIdx << " ["
                            << matcher.toString() << "]:\n");

    // Consume as many matching args as allowed (greedy)
    unsigned consumed = 0;
    while (argIdx < argValues.size() && consumed < matcher.getMaxCount()) {
      Type argType = argValues[argIdx].getType();

      // Unless the matcher explicitly preserves LValueType, unwrap it
      // Most builtins expect rvalue types, but some (like @allocate_fifo)
      // expect lvalue references
      if (!matcher.shouldPreserveLValue()) {
        if (auto lvalueType = dyn_cast<LValueType>(argType))
          argType = lvalueType.getValueType();
      }

      // Build context for this argument
      MatchContext matchCtx{static_cast<unsigned>(argIdx), resolvedTypes,
                            argValues};

      LLVM_DEBUG({
        llvm::dbgs() << "      Arg #" << argIdx << " ("
                     << prettyPrintType(argType);
        if (!isa<NoValueAttr>(argValues[argIdx]))
          llvm::dbgs() << "=" << prettyPrintValue(argValues[argIdx]);
        llvm::dbgs() << "): ";
      });

      MatchResult result = matcher.match(argType, matchCtx);
      if (!result.matched) {
        LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
        break;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "OK";
        if (result.targetType && argType != *result.targetType)
          llvm::dbgs() << " (cast to " << prettyPrintType(*result.targetType)
                       << ")";
        llvm::dbgs() << "\n";
      });

      // Determine resolved type and required cast
      Type resolvedType = argType;
      if (result.targetType && argType != *result.targetType) {
        // Need a cast to the target type
        requiredCasts.push_back(
            {static_cast<unsigned>(argIdx), *result.targetType});
        resolvedType = *result.targetType;
      }
      resolvedTypes.push_back(resolvedType);

      ++argIdx;
      ++consumed;
    }

    // Check minimum count satisfied
    if (consumed < matcher.getMinCount()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    -> Matcher failed: consumed " << consumed
                 << " args, need at least " << matcher.getMinCount() << "\n");
      return std::nullopt;
    }
    ++matcherIdx;
  }

  // All arguments must be consumed
  if (argIdx != argValues.size()) {
    LLVM_DEBUG(llvm::dbgs() << "  -> Signature failed: " << argIdx << " of "
                            << argValues.size() << " args consumed\n");
    return std::nullopt;
  }

  LLVM_DEBUG(llvm::dbgs() << "  -> Signature #" << sigIdx << " MATCHED\n");

  Type inferredReturnType = spec.returnType.getType(argValues);
  return SignatureMatchResult::success(spec, inferredReturnType, resolvedTypes,
                                       requiredCasts);
}

SignatureMatchResult
BuiltinSignatureMatcher::match(BuiltinKind kind, ArrayRef<TypedAttr> argValues,
                               MLIRContext *ctx) {
  const BuiltinDescriptor &desc = BuiltinDescriptor::get(kind, ctx);

  LLVM_DEBUG({
    llvm::dbgs() << "Matching @" << desc.getName() << " with args: (";
    for (size_t i = 0; i < argValues.size(); ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << prettyPrintType(argValues[i].getType());
      // Show comptime value if known
      if (!isa<NoValueAttr>(argValues[i]))
        llvm::dbgs() << "=" << prettyPrintValue(argValues[i]);
    }
    llvm::dbgs() << ")\n";
  });

  unsigned sigIdx = 0;
  for (const SignatureSpec &spec : desc.signatures) {
    if (auto result = tryMatchSignature(spec, argValues, ctx, sigIdx++))
      return *result;
  }

  LLVM_DEBUG(llvm::dbgs() << "-> No matching signature found\n");

  // No match - build error message
  if (desc.signatures.empty()) {
    std::string errMsg = "Oh no! builtin @";
    errMsg += desc.getName().str();
    errMsg += " has no signatures defined yet. PLEASE file a bug report.";
    return SignatureMatchResult::failure(errMsg);
  }

  // @await_parallel matches any number of async calls but one.
  if (desc.kind == BuiltinKind::AwaitParallel && argValues.size() < 2)
    return SignatureMatchResult::failure(
        "@await_parallel runs two or more async calls at the same time; to "
        "await a single call, use @await");

  std::string errMsg = "no matching signature for @";
  errMsg += desc.getName().str();
  errMsg += " with argument types (";
  for (size_t i = 0; i < argValues.size(); ++i) {
    if (i > 0)
      errMsg += ", ";
    if (!isa<NoValueAttr>(argValues[i]))
      errMsg += "comptime known ";
    errMsg += prettyPrintType(argValues[i].getType());
  }
  errMsg += "). Available signatures are:\n";
  for (const SignatureSpec &spec : desc.signatures) {
    errMsg += "  - (";
    for (size_t i = 0; i < spec.paramMatchers.size(); ++i) {
      if (i > 0)
        errMsg += ", ";
      errMsg += spec.paramMatchers[i].toString();
    }
    errMsg += ")\n";
  }

  return SignatureMatchResult::failure(errMsg);
}

FailureOr<Type> BuiltinSignatureMatcher::inferReturnType(
    BuiltinKind kind, ArrayRef<TypedAttr> argValues, MLIRContext *ctx) {
  auto result = match(kind, argValues, ctx);
  if (!result.isValid)
    return failure();

  return result.inferredReturnType;
}

FailureOr<SmallVector<std::pair<unsigned, Type>>>
BuiltinSignatureMatcher::getRequiredCasts(BuiltinKind kind,
                                          ArrayRef<TypedAttr> argValues,
                                          MLIRContext *ctx) {
  auto result = match(kind, argValues, ctx);
  if (!result.isValid)
    return failure();
  return result.requiredCasts;
}
