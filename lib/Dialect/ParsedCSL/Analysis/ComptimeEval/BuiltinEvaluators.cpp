//===- BuiltinEvaluators.cpp - Builtin Comptime Evaluators ---------------===//
//
// Per-builtin comptime evaluator implementations.
// Uses CSLBuiltinDescriptor for declarative validation/type inference,
// with custom evaluators for builtins that need imperative evaluation logic.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/BuiltinEvaluators.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinConfig.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/CSL/Utils/CoroRuntimeSources.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Target/CSL/ParseCSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace mlir;
using namespace mlir::parsed_csl;
using namespace mlir::csl;

namespace {

/// Convert the anonymous option struct used by CSL source syntax into the
/// compact attribute stored in comptime DSD values. DSD configuration structs
/// have no expected field type, so their nested `.simd_mode` value is otherwise
/// retained as a StructValueAttr rather than implicitly becoming SimdModeAttr.
static SimdModeAttr getSimdModeAttr(TypedAttr value) {
  if (auto mode = mlir::dyn_cast_if_present<SimdModeAttr>(value))
    return mode;

  auto option = mlir::dyn_cast_if_present<StructValueAttr>(value);
  if (!option)
    return {};

  constexpr std::pair<StringLiteral, SimdModeKind> modes[] = {
      {"simd_32", SimdModeKind::Simd32},
      {"simd_64", SimdModeKind::Simd64},
      {"simd_32_or_64", SimdModeKind::Simd32Or64},
      {"simd_16", SimdModeKind::Simd16},
      {"simd_off", SimdModeKind::SimdOff},
      {"simd_max", SimdModeKind::SimdMax},
  };
  for (auto [fieldName, mode] : modes) {
    auto enabled =
        mlir::dyn_cast_if_present<IntegerAttr>(option.getField(fieldName));
    if (enabled && enabled.getValue().getBoolValue())
      return SimdModeAttr::get(value.getContext(), mode);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BuiltinEvaluatorDefaults - CRTP base with default implementations
//===----------------------------------------------------------------------===//

/// CRTP base class providing default implementations that use
/// CSLBuiltinDescriptor for declarative validation and type inference.
template <typename Derived>
struct BuiltinEvaluatorDefaults
    : public ComptimeEvaluableOpInterface::ExternalModel<Derived,
                                                         BuiltinCallOp> {

  /// Gathers argument values/types for signature matching.
  /// Returns failure with error if type inference fails for any argument.
  static std::variant<ComptimeError, SmallVector<TypedAttr>>
  gatherArgValues(Operation *op, ComptimeEvalContext &context) {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    SmallVector<TypedAttr> argValues;
    for (Value arg : builtinCallOp.getArgs()) {
      // Get as lvalue if possible
      if (auto lvalue = context.getComptimeLValue(arg); succeeded(lvalue)) {
        if (auto lvalueAttr = lvalue->toAttr()) {
          argValues.push_back(*lvalueAttr);
          continue;
        }
      }

      // Get rvalue (loading lvalues if needed)
      FailureOr<TypedAttr> argValue = context.getOrLoadComptimeRValue(arg);
      if (succeeded(argValue)) {
        argValues.push_back(*argValue);
        continue;
      }

      // Use NoValueAttr with type if not comptime-known
      // Use getInferredType (not getInferredRValueType) to preserve LValueType.
      // The signature matcher will unwrap LValueType for matchers that don't
      // explicitly use preserveLValue().
      FailureOr<Type> argType = context.getInferredType(arg);
      if (failed(argType)) {
        return ComptimeError::semanticError(op, context)
               << "cannot infer type of argument";
      }
      argValues.push_back(csl::NoValueAttr::get(*argType));
    }
    return argValues;
  }

  /// Check the configuration struct arguments of a builtin, such as the
  /// configuration of @get_dsd, against their fields (see csl::checkConfig).
  static std::optional<ComptimeError>
  checkConfigArgs(Operation *op, ComptimeEvalContext &context,
                  csl::BuiltinKind kind, ArrayRef<TypedAttr> argValues) {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    bool isWse3 =
        context.getGlobalEvalContext()->getOptions().cslFlags.targetArch ==
        csl::TargetArch::WSE3;
    for (auto [idx, argValue] : llvm::enumerate(argValues)) {
      const csl::ConfigSchema *schema =
          csl::getConfigSchema(kind, idx, argValues);
      if (!schema)
        continue;
      Value arg = builtinCallOp.getArgs()[idx];
      bool comptimeKnown = !mlir::isa<csl::NoValueAttr>(argValue);
      // A field of a literal is comptime-known if the value it is built from
      // is.
      auto isFieldComptimeKnown = [&](ArrayRef<StringRef> path) {
        if (comptimeKnown)
          return true;
        Value current = arg;
        for (StringRef name : path) {
          auto literal = current.getDefiningOp<ConstructStructOp>();
          if (!literal)
            return false;
          ArrayRef<Attribute> names = literal.getFieldNames().getValue();
          auto it = llvm::find_if(names, [&](Attribute fieldName) {
            return mlir::cast<StringAttr>(fieldName).getValue() == name;
          });
          if (it == names.end())
            return false;
          current = literal.getFieldValues()[it - names.begin()];
        }
        return context.isRValueComptimeKnown(current);
      };
      if (std::optional<std::string> message =
              csl::checkConfig(unwrapLValueType(argValue.getType()), *schema,
                               isWse3, isFieldComptimeKnown))
        return ComptimeError::semanticError(op, context) << *message;
    }
    return std::nullopt;
  }

  /// Default validation using BuiltinSignatureMatcher.
  /// Returns nullopt on success, ComptimeError on failure.
  static std::optional<ComptimeError>
  defaultValidation(Operation *op, ComptimeEvalContext &context,
                    ArrayRef<TypedAttr> argValues) {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto builtinKind = builtinCallOp.resolveBuiltin();
    if (!builtinKind) {
      return ComptimeError::semanticError(op, context)
             << "unknown builtin function: @" << builtinCallOp.getBuiltinName();
    }

    const BuiltinDescriptor &desc =
        BuiltinDescriptor::get(*builtinKind, op->getContext());

    // Configuration structs first: their diagnostics are more precise than
    // a mismatching signature.
    if (auto error = checkConfigArgs(op, context, *builtinKind, argValues))
      return error;

    // Match signature using BuiltinSignatureMatcher
    SignatureMatchResult result = BuiltinSignatureMatcher::match(
        *builtinKind, argValues, op->getContext());

    if (!result.isValid) {
      return ComptimeError::semanticError(op, context) << result.errorMessage;
    }

    bool inComptimeContext = context.isComptimeContext();
    if (desc.allowedContext == BuiltinAllowedContext::ComptimeOnly &&
        !inComptimeContext) {
      return ComptimeError::semanticError(op, context)
             << "builtin can only be used in comptime context";
    }

    if (desc.allowedContext == BuiltinAllowedContext::RuntimeOnly &&
        inComptimeContext) {
      return ComptimeError::semanticError(op, context)
             << "builtin can only be used in runtime context";
    }

    return std::nullopt;
  }

  /// Overload that gathers argument values automatically before validating.
  /// Returns nullopt on success, ComptimeError on failure.
  static std::optional<ComptimeError>
  defaultValidation(Operation *op, ComptimeEvalContext &context) {
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    return defaultValidation(
        op, context, std::get<SmallVector<TypedAttr>>(argValuesOrError));
  }

  /// Default type inference using BuiltinSignatureMatcher.
  static Type defaultInferResultType(Operation *op,
                                     ComptimeEvalContext &context,
                                     ArrayRef<TypedAttr> argValues) {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // resolveBuiltin() must succeed here — defaultValidation already checked.
    auto builtinKind = *builtinCallOp.resolveBuiltin();

    // Get return type from signature matcher
    FailureOr<Type> returnType = BuiltinSignatureMatcher::inferReturnType(
        builtinKind, argValues, op->getContext());
    assert(succeeded(returnType) &&
           "inferReturnType should not fail here — defaultValidation already "
           "checked");

    return *returnType;
  }

  /// Default comptime evaluation.
  /// Returns success for final-comptime builtins, type-only for runtime
  /// builtins.
  static ComptimeEvalResult defaultEvaluation(Operation *op,
                                              ComptimeEvalContext &context,
                                              Type resultType) {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    // resolveBuiltin() must succeed here — defaultValidation already checked.
    auto builtinKind = *builtinCallOp.resolveBuiltin();
    const BuiltinDescriptor &desc =
        BuiltinDescriptor::get(builtinKind, op->getContext());

    // A task bound to a known ID, at comptime or at runtime, is not available
    // to the runtime allocator.
    if (builtinKind == csl::BuiltinKind::BindLocalTask ||
        builtinKind == csl::BuiltinKind::BindDataTask ||
        builtinKind == csl::BuiltinKind::BindControlTask) {
      FailureOr<TypedAttr> taskId =
          context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[1]);
      if (succeeded(taskId))
        context.getGlobalEvalContext()->getResourceBitmaps().reserveBoundTask(
            context.getGlobalEvalContext()->getOptions().cslFlags.targetArch,
            *taskId);
    }

    bool inComptimeContext = context.isComptimeContext();
    if (desc.hasRuntimeSideEffects(inComptimeContext)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "builtin has runtime side effects and cannot be evaluated at "
                "comptime";
    }

    // Record builtin calls that must be evaluated at final comptime
    if (inComptimeContext &&
        desc.mustEvaluateAtFinalComptime(inComptimeContext)) {
      // Gather arguments
      auto argValuesOrError = gatherArgValues(op, context);
      assert(!std::holds_alternative<ComptimeError>(argValuesOrError) &&
             "gatherArgValues should not fail here — defaultValidation already "
             "checked");
      auto origArgValues = std::get<SmallVector<TypedAttr>>(argValuesOrError);
      SignatureMatchResult result = BuiltinSignatureMatcher::match(
          builtinKind, origArgValues, op->getContext());
      SmallVector<TypedAttr> argValues;

      for (auto [index, argValue] : llvm::enumerate(origArgValues)) {
        if (isa<csl::NoValueAttr>(argValue)) {
          return ComptimeError::comptimeFailure(op, context, resultType)
                 << "builtin argument is not comptime-known";
        }

        if (isa<LValueType>(argValue.getType()) &&
            !isa<LValueType>(result.resolvedParamTypes[index])) {
          // Load lvalue if the parameter type doesn't preserve lvalues
          auto loadedValue =
              context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[index]);
          if (failed(loadedValue)) {
            return ComptimeError::comptimeFailure(op, context, resultType)
                   << "failed to load lvalue argument";
          }
          argValue = *loadedValue;
        }

        argValues.push_back(argValue);
      }
      context.getGlobalEvalContext()->recordBuiltinCall(builtinKind, argValues,
                                                        resultType, op);

      // This is somewhat of a special case: We dont have a value, but comptime
      // eval still succeeded and we know the type
      ComptimeEvalResult evalRes(success());
      evalRes.inferredType = resultType;
      return evalRes;
    }

    // Not comptime-evaluable
    return ComptimeError::comptimeFailure(op, context, resultType)
           << "builtin call has no comptime model attached. Report this to "
              "compiler developers";
  }

  /// Result of running validation and type inference phases.
  struct ValidationResult {
    SmallVector<TypedAttr> argValues;
    Type resultType;
  };

  /// Run validation and type inference, returning either an error or the
  /// validation result. Derived classes can use this to avoid boilerplate.
  static std::variant<ComptimeError, ValidationResult>
  runValidationAndTypeInference(Operation *op, ComptimeEvalContext &context) {
    // Gather arguments
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError)) {
      return std::get<ComptimeError>(argValuesOrError);
    }
    SmallVector<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(std::move(argValuesOrError));

    // Validation
    if (auto error = defaultValidation(op, context, argValues)) {
      return *error;
    }

    // Type Inference
    Type resultType = defaultInferResultType(op, context, argValues);

    return ValidationResult{std::move(argValues), resultType};
  }

  /// Unified evaluate method combining validation, type inference, and
  /// evaluation.
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError)) {
      return std::get<ComptimeError>(validationOrError);
    }
    auto &validation = std::get<ValidationResult>(validationOrError);
    return defaultEvaluation(op, context, validation.resultType);
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto builtinKind = builtinCallOp.resolveBuiltin();
    if (!builtinKind)
      return false; // Unknown builtin — will error during evaluate()

    const BuiltinDescriptor &desc =
        BuiltinDescriptor::get(*builtinKind, op->getContext());

    return desc.mustEvaluateAtOurComptime(context.isComptimeContext());
  }
};

//===----------------------------------------------------------------------===//
// DefaultBuiltinEvaluator - uses all defaults
//===----------------------------------------------------------------------===//

struct DefaultBuiltinEvaluator
    : public BuiltinEvaluatorDefaults<DefaultBuiltinEvaluator> {};

//===----------------------------------------------------------------------===//
// TypeOfEvaluator
//===----------------------------------------------------------------------===//

struct TypeOfEvaluator : public BuiltinEvaluatorDefaults<TypeOfEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // ========== PHASE 1: Gather arguments ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError)) {
      return std::get<ComptimeError>(argValuesOrError);
    }
    ArrayRef<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);

    // ========== PHASE 2: Validation ==========
    if (auto error = defaultValidation(op, context, argValues)) {
      return *error;
    }

    // ========== PHASE 3: Type Inference ==========
    Type resultType = defaultInferResultType(op, context, argValues);

    // ========== PHASE 4: Comptime Evaluation ==========
    // TypeOf builtin: get the (rvalue) type of the argument
    FailureOr<Type> argType =
        context.getInferredRValueType(builtinCallOp.getArgs()[0]);
    if (failed(argType)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "failed to determine type of argument";
    }
    return cast<TypedAttr>(csl::TypeAttr::get(*argType));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// FutureOfEvaluator
//===----------------------------------------------------------------------===//

struct FutureOfEvaluator : public BuiltinEvaluatorDefaults<FutureOfEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    // ========== PHASE 1: Gather arguments ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    ArrayRef<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);

    // ========== PHASE 2: Validation ==========
    if (auto error = defaultValidation(op, context, argValues))
      return *error;

    // ========== PHASE 3+4: Comptime Evaluation ==========
    // The argument must be a function reference (FuncRefAttr).
    TypedAttr argAttr = argValues[0];
    auto funcRef = mlir::dyn_cast<csl::FuncRefAttr>(argAttr);
    if (!funcRef) {
      return ComptimeError::semanticError(op, context)
             << "@future_of argument must be a function reference";
    }

    // Look up the GenericFuncOp and verify it's async.
    auto cu = op->getParentOfType<csl::CompilationUnitOp>();
    auto funcOrError = lookupFuncByRefAttr(funcRef, cu);
    if (failed(funcOrError)) {
      return ComptimeError::semanticError(op, context)
             << "@future_of: could not resolve function '"
             << funcRef.getSymbolName().getValue() << "'";
    }
    auto *generic = std::get_if<GenericFuncOp>(&*funcOrError);
    if (!generic) {
      return ComptimeError::semanticError(op, context)
             << "@future_of argument must reference a GenericFuncOp";
    }
    if (!generic->getIsAsync()) {
      return ComptimeError::semanticError(op, context)
             << "@future_of argument must reference an async function or task";
    }

    // The FuncRefAttr carries the original FunctionType (before the async
    // rewrite wraps the result in FutureType). Extract the return type.
    // Generic (comptime/anytype) async functions are not supported.
    if (funcRef.isGeneric()) {
      return ComptimeError::semanticError(op, context)
             << "@future_of argument must reference a non-generic function";
    }

    FunctionType funcType = funcRef.getFunctionType();
    Type asyncResultTy = funcType.getNumResults() > 0
                             ? funcType.getResult(0)
                             : csl::VoidType::get(op->getContext());

    auto futureTy =
        csl::FutureType::get(op->getContext(), funcRef.getModuleIdentifier(),
                             funcRef.getSymbolName(), asyncResultTy);
    return cast<TypedAttr>(csl::TypeAttr::get(futureTy));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ImportModuleEvaluator
//===----------------------------------------------------------------------===//

struct ImportModuleEvaluator
    : public BuiltinEvaluatorDefaults<ImportModuleEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // ========== PHASE 1: Gather arguments ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError)) {
      return std::get<ComptimeError>(argValuesOrError);
    }
    ArrayRef<TypedAttr> argValuesRef =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);

    // ========== PHASE 2: Validation ==========
    if (auto error = defaultValidation(op, context, argValuesRef)) {
      return *error;
    }

    // ========== PHASE 3: Type Inference ==========
    Type resultType = ImportedModuleType::get(op->getContext());

    // ========== PHASE 4: Comptime Evaluation ==========
    // Get comptime argument values
    llvm::SmallVector<TypedAttr> argValues;
    for (Value arg : builtinCallOp.getArgs()) {
      auto argValue = context.getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(arg)
               << "argument is not comptime-known";
      }
      argValues.push_back(*argValue);
    }

    auto filenameAttr = cast<ComptimeStringAttr>(argValues[0]);
    llvm::StringRef filename = filenameAttr.getValue();

    // Get param bindings if present (optional second argument)
    llvm::DenseMap<llvm::StringRef, csl::ParamBinding> paramBindings;
    if (argValues.size() > 1) {
      auto structAttr = mlir::dyn_cast<StructValueAttr>(argValues[1]);
      if (!structAttr) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "@import_module param_binding must be a struct";
      }

      // Check it's not a tuple (must have named fields)
      if (structAttr.isTuple()) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "@import_module param_binding must be a struct with named "
                  "fields, not a tuple";
      }

      // Get location from the struct argument for better diagnostics
      Location bindingLoc = builtinCallOp.getArgs()[1].getLoc();

      // Extract field name-value pairs with location
      auto fieldNames = structAttr.getFieldNames();
      auto fieldValues = structAttr.getFieldValues();
      for (size_t i = 0; i < structAttr.getNumFields(); ++i) {
        paramBindings.insert({fieldNames[i], {fieldValues[i], bindingLoc}});
      }
    }

    // Get the module resolver from the context
    auto *resolver = context.getModuleResolver();
    if (!resolver) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "@import_module requires CslModuleResolver to be available";
    }

    // Detect system import: angle brackets inside the string "<path>"
    bool isSystemImport = filename.starts_with("<") && filename.ends_with(">");

    std::string includedFrom;
    unsigned bufferID = 0;

    if (isSystemImport) {
      // System import: strip angle brackets and resolve through resolver
      StringRef modulePath = filename.drop_front(1).drop_back(1);

      // The coroutine runtime is served from memory by the resolver, so it is
      // exempt from the import-path requirement below -- it is part of the
      // compiler and does not live on any search path.
      if (!csl::isCoroRuntimeImportPath(modulePath) &&
          !resolver->hasSystemImportPaths()) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "system imports (e.g. <memcpy/memcpy>) require "
                  "--system-module-sif or --system-import-path to be "
                  "configured";
      }

      auto result = resolver->resolveSystemModule(modulePath);
      if (failed(result)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "failed to find system module: " << modulePath
               << " (searched in --system-import-path directories)";
      }
      bufferID = result->bufferID;
      includedFrom = result->resolvedPath;
    } else {
      // Regular import: resolve through resolver. Pass the importing file's
      // directory so sibling imports inside libraries (e.g. lib/pe.csl
      // importing lib/helper.csl by its bare name) resolve regardless of
      // where the top-level PE program lives.
      llvm::StringRef importerDir;
      if (auto fileLoc = mlir::dyn_cast<FileLineColLoc>(op->getLoc())) {
        importerDir =
            llvm::sys::path::parent_path(fileLoc.getFilename().getValue());
      }
      auto result = resolver->resolveUserModule(filename, importerDir);
      if (failed(result)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "failed to load module file: " << filename;
      }
      bufferID = result->bufferID;
      includedFrom = result->resolvedPath;
    }

    auto sourceMgr = resolver->getSourceMgr();

    // Create the ModuleOp at scope level (before the GlobalOp whose init
    // region contains this import_module call).
    OpBuilder builder(op->getContext());
    auto globalOp = op->getParentOfType<GlobalOp>();
    if (!globalOp) {
      return ComptimeError::semanticError(op, context)
             << "@import_module must be used in a global initializer";
    }
    builder.setInsertionPoint(globalOp);

    // Create a unique instance ID for this module instantiation
    static std::atomic<int64_t> nextModuleIdentifier{0};
    int64_t moduleIdentifier = nextModuleIdentifier.fetch_add(1);
    auto moduleOp = parsed_csl::ModuleOp::create(
        builder, op->getLoc(), includedFrom, moduleIdentifier, {});

    // Make the module handle available while its body is evaluated. SDK
    // modules may bind a callback from the importing module and analyze that
    // callback in a top-level comptime block; the callback can legitimately
    // refer back to this module handle. This is not a value cycle because the
    // ModuleOp and its stable identifier already exist.
    auto moduleRef =
        ModuleRefAttr::get(moduleOp->getContext(), moduleIdentifier);
    globalOp.setInferredType(csl::LValueType::get(
        csl::ImportedModuleType::get(moduleOp->getContext()),
        globalOp.getIsConst()));
    globalOp.setInferredInitializerAttr(moduleRef);

    // Parse the CSL file into the module's region
    const auto &opts = context.getGlobalEvalContext()->getOptions();
    if (failed(csl::parseCSLIntoRegion(sourceMgr, bufferID,
                                       moduleOp.getRegion(), paramBindings, 0,
                                       includedFrom, moduleOp, opts.lspMode,
                                       opts.cslFlags, isSystemImport))) {
      if (opts.lspMode)
        op->emitError("error parsing imported module here");
      else
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "failed to parse module file: " << filename;
    }

    // Evaluate the module's body
    ControlFlowOutcomes outcome = context.evaluateModule(moduleOp);
    if (outcome.hasFatalError()) {
      op->emitError("error evaluating imported module here");
      if (!opts.lspMode)
        return outcome.getFatalError().withType(resultType);
    }

    // Return a ModuleRefAttr pointing to the module
    return cast<TypedAttr>(moduleRef);
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// IsArchEvaluator
//===----------------------------------------------------------------------===//

struct IsArchEvaluator : public BuiltinEvaluatorDefaults<IsArchEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // ========== PHASE 1-3: Validation & Type Inference ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    ArrayRef<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);
    if (auto error = defaultValidation(op, context, argValues))
      return *error;
    Type resultType = defaultInferResultType(op, context, argValues);

    // ========== PHASE 4: Comptime Evaluation ==========
    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    llvm::StringRef archArg = cast<ComptimeStringAttr>(*argValue).getValue();
    llvm::StringRef targetArch = csl::stringifyTargetArch(
        context.getGlobalEvalContext()->getOptions().cslFlags.targetArch);

    bool result = (archArg == targetArch);
    return cast<TypedAttr>(BoolAttr::get(op->getContext(), result));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true; // is_arch must always be evaluated at comptime
  }
};

//===----------------------------------------------------------------------===//
// IsComptimeEvaluator
//===----------------------------------------------------------------------===//

struct IsComptimeEvaluator
    : public BuiltinEvaluatorDefaults<IsComptimeEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    // ========== PHASE 1-3: Validation & Type Inference ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    ArrayRef<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);
    if (auto error = defaultValidation(op, context, argValues))
      return *error;

    // ========== PHASE 4: Comptime Evaluation ==========
    // Returns true if evaluated in a comptime context, false otherwise.
    bool result = context.isComptimeContext();
    return cast<TypedAttr>(BoolAttr::get(op->getContext(), result));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true; // is_comptime must always be evaluated at comptime
  }
};

//===----------------------------------------------------------------------===//
// StrlenEvaluator
//===----------------------------------------------------------------------===//

struct StrlenEvaluator : public BuiltinEvaluatorDefaults<StrlenEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // ========== PHASE 1-3: Validation & Type Inference ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    ArrayRef<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);
    if (auto error = defaultValidation(op, context, argValues))
      return *error;
    Type resultType = defaultInferResultType(op, context, argValues);

    // ========== PHASE 4: Comptime Evaluation ==========
    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    llvm::StringRef strValue = cast<ComptimeStringAttr>(*argValue).getValue();
    APInt length(64, strValue.size(), /*isSigned=*/true);
    return cast<TypedAttr>(ComptimeIntAttr::get(op->getContext(), length));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true; // strlen must always be evaluated at comptime
  }
};

//===----------------------------------------------------------------------===//
// StrcatEvaluator
//===----------------------------------------------------------------------===//

struct StrcatEvaluator : public BuiltinEvaluatorDefaults<StrcatEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // ========== PHASE 1-3: Validation & Type Inference ==========
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    ArrayRef<TypedAttr> argValues =
        std::get<SmallVector<TypedAttr>>(argValuesOrError);
    if (auto error = defaultValidation(op, context, argValues))
      return *error;
    Type resultType = defaultInferResultType(op, context, argValues);

    // ========== PHASE 4: Comptime Evaluation ==========
    std::string result;
    for (Value arg : builtinCallOp.getArgs()) {
      auto argValue = context.getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(arg)
               << "argument is not comptime-known";
      }

      llvm::StringRef strValue = cast<ComptimeStringAttr>(*argValue).getValue();
      result += strValue;
    }

    return cast<TypedAttr>(ComptimeStringAttr::get(op->getContext(), result));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true; // strcat must always be evaluated at comptime
  }
};

//===----------------------------------------------------------------------===//
// GetRectangleEvaluator
//===----------------------------------------------------------------------===//

struct GetRectangleEvaluator
    : public BuiltinEvaluatorDefaults<GetRectangleEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    // Validation result (argValues, resultType) not needed for @get_rectangle
    (void)std::get<ValidationResult>(validationOrError);

    MLIRContext *ctx = op->getContext();
    const auto &flags = context.getGlobalEvalContext()->getOptions().cslFlags;

    // Check if rectangle is configured
    if (!flags.rectangle) {
      return ComptimeError::semanticError(op, context)
             << "@get_rectangle requires rectangle to be specified "
                "(@set_rectangle)";
    }

    Type u16Type = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    // Helper to create u16 attribute
    auto makeU16 = [&](uint16_t value) -> TypedAttr {
      return IntegerAttr::get(u16Type, value);
    };

    // Build fabric nested struct: .{ .width = ..., .height = ... }
    SmallVector<TypedAttr> fabricFields = {makeU16(flags.fabricDims.width),
                                           makeU16(flags.fabricDims.height)};
    SmallVector<StringAttr> fabricFieldNames = {StringAttr::get(ctx, "width"),
                                                StringAttr::get(ctx, "height")};
    auto fabricStruct =
        StructValueAttr::get(ctx, fabricFields,
                             /*def_location=*/{}, fabricFieldNames);

    // Build offsets nested struct: .{ .width = ..., .height = ... }
    SmallVector<TypedAttr> offsetsFields = {
        makeU16(flags.fabricOffsets.width),
        makeU16(flags.fabricOffsets.height)};
    SmallVector<StringAttr> offsetsFieldNames = {
        StringAttr::get(ctx, "width"), StringAttr::get(ctx, "height")};
    auto offsetsStruct =
        StructValueAttr::get(ctx, offsetsFields,
                             /*def_location=*/{}, offsetsFieldNames);

    // Build top-level struct:
    // .{ .width = ..., .height = ..., .fabric = ..., .offsets = ... }
    SmallVector<TypedAttr> topFields = {makeU16(flags.rectangle.width),
                                        makeU16(flags.rectangle.height),
                                        fabricStruct, offsetsStruct};
    SmallVector<StringAttr> topFieldNames = {
        StringAttr::get(ctx, "width"), StringAttr::get(ctx, "height"),
        StringAttr::get(ctx, "fabric"), StringAttr::get(ctx, "offsets")};
    return cast<TypedAttr>(StructValueAttr::get(ctx, topFields,
                                                /*def_location=*/{},
                                                topFieldNames));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true; // get_rectangle must always be evaluated at comptime
  }
};

//===----------------------------------------------------------------------===//
// SetRectangleEvaluator
//===----------------------------------------------------------------------===//

struct SetRectangleEvaluator
    : public BuiltinEvaluatorDefaults<SetRectangleEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    // Load the actual rvalue for each argument (gatherArgValues may return
    // lvalue representations like GlobalRefAttr for global params)
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();

    auto extractInt =
        [&](unsigned idx) -> std::variant<ComptimeError, int64_t> {
      FailureOr<TypedAttr> val = context.getOrLoadComptimeRValue(args[idx]);
      if (failed(val)) {
        return ComptimeError::semanticError(op, context)
               << "@set_rectangle argument " << idx << " is not comptime-known";
      }
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(*val))
        return intAttr.getValue().getSExtValue();
      if (auto comptimeAttr = mlir::dyn_cast<csl::ComptimeIntAttr>(*val))
        return comptimeAttr.getValue().getSExtValue();
      return ComptimeError::semanticError(op, context)
             << "@set_rectangle argument " << idx << " is not an integer";
    };

    auto widthOrErr = extractInt(0);
    if (std::holds_alternative<ComptimeError>(widthOrErr))
      return std::get<ComptimeError>(widthOrErr);
    auto heightOrErr = extractInt(1);
    if (std::holds_alternative<ComptimeError>(heightOrErr))
      return std::get<ComptimeError>(heightOrErr);

    int64_t width = std::get<int64_t>(widthOrErr);
    int64_t height = std::get<int64_t>(heightOrErr);

    if (width <= 0 || height <= 0) {
      return ComptimeError::semanticError(op, context)
             << "@set_rectangle dimensions must be positive, got " << width
             << "x" << height;
    }

    // Store the rectangle dimensions so @get_rectangle can find them
    context.getGlobalEvalContext()->setRectangle(static_cast<uint16_t>(width),
                                                 static_cast<uint16_t>(height));

    // Delegate to default evaluation to record the builtin call
    return defaultEvaluation(op, context, validation.resultType);
  }
};

//===----------------------------------------------------------------------===//
// IsSameTypeEvaluator
//===----------------------------------------------------------------------===//

struct IsSameTypeEvaluator
    : public BuiltinEvaluatorDefaults<IsSameTypeEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    llvm::SmallVector<TypedAttr> argValues;
    for (auto [i, arg] : llvm::enumerate(builtinCallOp.getArgs())) {
      auto argValue = context.getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        return ComptimeError::comptimeFailure(op, context,
                                              validation.resultType)
                   .withDependency(arg)
               << "argument " << (i + 1) << " is not comptime-known";
      }
      argValues.push_back(*argValue);
    }

    Type typeA = cast<csl::TypeAttr>(argValues[0]).getValue();
    Type typeB = cast<csl::TypeAttr>(argValues[1]).getValue();
    bool isSame = (typeA == typeB);
    return cast<TypedAttr>(BoolAttr::get(op->getContext(), isSame));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ComptimeAssertEvaluator
//===----------------------------------------------------------------------===//

struct ComptimeAssertEvaluator
    : public BuiltinEvaluatorDefaults<ComptimeAssertEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    llvm::SmallVector<TypedAttr> argValues;
    for (auto [i, arg] : llvm::enumerate(builtinCallOp.getArgs())) {
      auto argValue = context.getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        return ComptimeError::comptimeFailure(op, context,
                                              validation.resultType)
                   .withDependency(arg)
               << "argument " << (i + 1) << " is not comptime-known";
      }
      argValues.push_back(*argValue);
    }

    TypedAttr conditionAttr = argValues[0];
    assert(conditionAttr.getType() == IntegerType::get(op->getContext(), 1) &&
           "comptime_assert condition must be boolean");
    auto conditionInt = cast<IntegerAttr>(conditionAttr);
    if (conditionInt.getValue().getBoolValue())
      return (TypedAttr)VoidAttr::get(op->getContext());

    std::string message = "assertion failed";
    if (argValues.size() == 2) {
      auto messageAttr = cast<ComptimeStringAttr>(argValues[1]);
      message = messageAttr.getValue().str();
    }
    return ComptimeError::executionError(op, context) << message;
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// ComptimePrintEvaluator
//===----------------------------------------------------------------------===//

struct ComptimePrintEvaluator
    : public BuiltinEvaluatorDefaults<ComptimePrintEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    llvm::SmallVector<TypedAttr> argValues;
    for (auto [i, arg] : llvm::enumerate(builtinCallOp.getArgs())) {
      auto argValue = context.getOrLoadComptimeRValue(arg);
      if (failed(argValue)) {
        return ComptimeError::comptimeFailure(op, context,
                                              validation.resultType)
                   .withDependency(arg)
               << "argument " << (i + 1) << " is not comptime-known";
      }
      argValues.push_back(*argValue);
    }

    // Emit remarks at the @comptime_print call location, not the argument's
    // definition location
    for (TypedAttr value : argValues) {
      mlir::emitRemark(op->getLoc()) << prettyPrintValue(value);
    }
    return (TypedAttr)VoidAttr::get(op->getContext());
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// GetColorEvaluator
//===----------------------------------------------------------------------===//

struct GetColorEvaluator : public BuiltinEvaluatorDefaults<GetColorEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    // Get the color ID argument value
    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    // Cast to u16 using the folder helper
    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u16Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "failed to cast color ID to u16";
    }
    diag.abandon();

    // Get the value as uint16_t
    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint16_t colorId = static_cast<uint16_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(csl::ColorAttr::get(op->getContext(), colorId));
  }
};

//===----------------------------------------------------------------------===//
// GetInputQueueEvaluator
//===----------------------------------------------------------------------===//

struct GetInputQueueEvaluator
    : public BuiltinEvaluatorDefaults<GetInputQueueEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    // Get the queue ID argument value
    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    // Cast to u16 using the folder helper
    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u16Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "failed to cast input queue ID to u16";
    }
    diag.abandon();

    // Get the value as uint16_t
    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint16_t queueId = static_cast<uint16_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(csl::InputQueueAttr::get(op->getContext(), queueId));
  }
};

//===----------------------------------------------------------------------===//
// GetOutputQueueEvaluator
//===----------------------------------------------------------------------===//

struct GetOutputQueueEvaluator
    : public BuiltinEvaluatorDefaults<GetOutputQueueEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    // Get the queue ID argument value
    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    // Cast to u16 using the folder helper
    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u16Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "failed to cast output queue ID to u16";
    }
    diag.abandon();

    // Get the value as uint16_t
    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint16_t queueId = static_cast<uint16_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(
        csl::OutputQueueAttr::get(op->getContext(), queueId));
  }
};

//===----------------------------------------------------------------------===//
// GetLocalTaskIdEvaluator
//===----------------------------------------------------------------------===//

struct GetLocalTaskIdEvaluator
    : public BuiltinEvaluatorDefaults<GetLocalTaskIdEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u16Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "failed to cast local task ID to u16";
    }
    diag.abandon();

    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint16_t taskId = static_cast<uint16_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(csl::LocalTaskIdAttr::get(op->getContext(), taskId));
  }
};

//===----------------------------------------------------------------------===//
// GetUtIdEvaluator - @get_ut_id builtin (WSE-3 microthread ID)
//===----------------------------------------------------------------------===//

struct GetUtIdEvaluator : public BuiltinEvaluatorDefaults<GetUtIdEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u16Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "failed to cast microthread ID to u16";
    }
    diag.abandon();

    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint16_t utId = static_cast<uint16_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(csl::UtIdAttr::get(op->getContext(), utId));
  }
};

//===----------------------------------------------------------------------===//
// GetDataTaskIdEvaluator
//===----------------------------------------------------------------------===//

struct GetDataTaskIdEvaluator
    : public BuiltinEvaluatorDefaults<GetDataTaskIdEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    // DataTaskId is created from a color (WSE-2) or input_queue (WSE-3)
    // Store the source attribute directly to enable proper emission
    TypedAttr source = *argValue;
    if (!mlir::isa<csl::ColorAttr, csl::InputQueueAttr>(source)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "@get_data_task_id expects a color or input_queue argument";
    }

    return cast<TypedAttr>(csl::DataTaskIdAttr::get(op->getContext(), source));
  }
};

//===----------------------------------------------------------------------===//
// GetControlTaskIdEvaluator
//===----------------------------------------------------------------------===//

struct GetControlTaskIdEvaluator
    : public BuiltinEvaluatorDefaults<GetControlTaskIdEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u16Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
             << "failed to cast control task ID to u16";
    }
    diag.abandon();

    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint16_t taskId = static_cast<uint16_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(
        csl::ControlTaskIdAttr::get(op->getContext(), taskId));
  }
};

//===----------------------------------------------------------------------===//
// GetDsrEvaluator - @get_dsr builtin
//===----------------------------------------------------------------------===//

struct GetDsrEvaluator : public BuiltinEvaluatorDefaults<GetDsrEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();

    // ========== PHASE 1: Validation ==========
    if (args.size() != 2) {
      return ComptimeError::semanticError(op, context)
             << "@get_dsr requires exactly 2 arguments";
    }

    // First argument must be comptime-known type
    FailureOr<TypedAttr> arg0Value = context.getOrLoadComptimeRValue(args[0]);
    if (failed(arg0Value)) {
      return ComptimeError::semanticError(op, context)
             << "first argument must be comptime-known";
    }

    auto typeAttr = mlir::dyn_cast<csl::TypeAttr>(*arg0Value);
    if (!typeAttr) {
      return ComptimeError::semanticError(op, context)
             << "first argument must be a DSR type";
    }

    Type dsrType = typeAttr.getValue();

    // Check if it's a valid DSR type
    if (!mlir::isa<DsrDestType, DsrSrc0Type, DsrSrc1Type, DsrFifoDestType,
                   DsrFifoSrc1Type>(dsrType)) {
      return ComptimeError::semanticError(op, context)
             << "first argument must be a DSR type (dsr_dest, dsr_src0, "
                "dsr_src1, dsr_fifo_dest, or dsr_fifo_src1)";
    }

    // Check second argument based on DSR type
    if (mlir::isa<DsrFifoDestType, DsrFifoSrc1Type>(dsrType)) {
      FailureOr<Type> arg1Type = context.getInferredRValueType(args[1]);
      if (failed(arg1Type)) {
        return ComptimeError::semanticError(op, context)
               << "cannot infer type of second argument";
      }

      if (mlir::isa<DsrFifoDestType>(dsrType)) {
        if (!mlir::isa<DsrDestType>(*arg1Type)) {
          return ComptimeError::semanticError(op, context)
                 << "@get_dsr(dsr_fifo_dest, ...) requires dsr_dest argument, "
                    "got "
                 << csl::prettyPrintType(*arg1Type);
        }
      } else {
        if (!mlir::isa<DsrSrc1Type>(*arg1Type)) {
          return ComptimeError::semanticError(op, context)
                 << "@get_dsr(dsr_fifo_src1, ...) requires dsr_src1 argument, "
                    "got "
                 << csl::prettyPrintType(*arg1Type);
        }
      }
    }

    // ========== PHASE 2: Type Inference ==========
    Type resultType = dsrType;

    // ========== PHASE 3: Comptime Evaluation ==========
    auto arg1Value = context.getOrLoadComptimeRValue(args[1]);
    if (failed(arg1Value)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[1])
             << "second argument is not comptime-known";
    }

    if (mlir::isa<DsrFifoDestType, DsrFifoSrc1Type>(dsrType)) {
      uint8_t dsrId;
      if (auto destAttr = mlir::dyn_cast<DsrDestAttr>(*arg1Value)) {
        dsrId = destAttr.getValue();
      } else if (auto src1Attr = mlir::dyn_cast<DsrSrc1Attr>(*arg1Value)) {
        dsrId = src1Attr.getValue();
      } else {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "invalid DSR value for FIFO conversion";
      }

      if (mlir::isa<DsrFifoDestType>(dsrType))
        return cast<TypedAttr>(DsrFifoDestAttr::get(op->getContext(), dsrId));
      return cast<TypedAttr>(DsrFifoSrc1Attr::get(op->getContext(), dsrId));
    }

    IntegerType u8Type =
        IntegerType::get(op->getContext(), 8, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*arg1Value, u8Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast DSR ID to u8";
    }
    diag.abandon();

    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint8_t dsrId = static_cast<uint8_t>(intAttr.getValue().getZExtValue());

    if (mlir::isa<DsrDestType>(dsrType))
      return cast<TypedAttr>(DsrDestAttr::get(op->getContext(), dsrId));
    if (mlir::isa<DsrSrc0Type>(dsrType))
      return cast<TypedAttr>(DsrSrc0Attr::get(op->getContext(), dsrId));
    if (mlir::isa<DsrSrc1Type>(dsrType))
      return cast<TypedAttr>(DsrSrc1Attr::get(op->getContext(), dsrId));

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "unexpected DSR type in @get_dsr";
  }
};

//===----------------------------------------------------------------------===//
// GetXdsrEvaluator - @get_xdsr builtin
//===----------------------------------------------------------------------===//

struct GetXdsrEvaluator : public BuiltinEvaluatorDefaults<GetXdsrEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    Type resultType = XdsrType::get(op->getContext());

    if (auto error = defaultValidation(op, context))
      return *error;

    auto argValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[0])
             << "argument is not comptime-known";
    }

    IntegerType u8Type =
        IntegerType::get(op->getContext(), 8, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u8Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast XDSR ID to u8";
    }
    diag.abandon();

    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint8_t xdsrId = static_cast<uint8_t>(intAttr.getValue().getZExtValue());

    return cast<TypedAttr>(XdsrAttr::get(op->getContext(), xdsrId));
  }
};

//===----------------------------------------------------------------------===//
// GetSrEvaluator - @get_sr builtin
//===----------------------------------------------------------------------===//

struct GetSrEvaluator : public BuiltinEvaluatorDefaults<GetSrEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    Type resultType = SrType::get(op->getContext());

    if (auto error = defaultValidation(op, context))
      return *error;

    auto argValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[0])
             << "argument is not comptime-known";
    }

    IntegerType u8Type =
        IntegerType::get(op->getContext(), 8, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedValue =
        csl::evaluateCast(*argValue, u8Type, &diag);
    if (failed(castedValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast SR ID to u8";
    }
    diag.abandon();

    auto intAttr = mlir::cast<IntegerAttr>(*castedValue);
    uint8_t srId = static_cast<uint8_t>(intAttr.getValue().getZExtValue());
    return cast<TypedAttr>(SrAttr::get(op->getContext(), srId));
  }
};

//===----------------------------------------------------------------------===//
// GetDsdEvaluator - @get_dsd builtin
//===----------------------------------------------------------------------===//

struct GetDsdEvaluator : public BuiltinEvaluatorDefaults<GetDsdEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    MLIRContext *ctx = op->getContext();

    if (auto error = defaultValidation(op, context))
      return *error;

    // Get DSD type from first argument
    auto typeValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(typeValue)) {
      return ComptimeError::semanticError(op, context)
             << "DSD type must be comptime-known";
    }
    Type dsdType = mlir::cast<csl::TypeAttr>(*typeValue).getValue();
    // The result type is the DSD type from the first argument
    Type resultType = dsdType;

    // Get configuration struct from second argument
    auto configValue = context.getOrLoadComptimeRValue(args[1]);
    if (failed(configValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[1])
             << "DSD configuration struct is not comptime-known";
    }

    auto configStruct = mlir::dyn_cast<StructValueAttr>(*configValue);
    if (!configStruct) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "DSD configuration must be a struct";
    }

    // Helper to extract integer field value
    auto getIntField = [&](StringRef name,
                           int64_t defaultVal) -> FailureOr<int64_t> {
      TypedAttr field = configStruct.getField(name);
      if (!field)
        return defaultVal;
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(field))
        return intAttr.getValue().getSExtValue();
      if (auto comptimeInt = mlir::dyn_cast<ComptimeIntAttr>(field))
        return comptimeInt.getValue().getSExtValue();
      return failure();
    };

    // Helper to extract bool field value
    auto getBoolField = [&](StringRef name, bool defaultVal) -> bool {
      TypedAttr field = configStruct.getField(name);
      if (!field)
        return defaultVal;
      if (auto boolAttr = mlir::dyn_cast<BoolAttr>(field))
        return boolAttr.getValue();
      return defaultVal;
    };

    // Handle based on DSD type
    if (mlir::isa<Mem1dDsdType>(dsdType)) {
      // Check for tensor_access field first
      TypedAttr taField = configStruct.getField("tensor_access");
      TypedAttr baseAddr;
      uint16_t extent = 1;
      int8_t stride = 0;
      int16_t offset = 0;

      if (taField) {
        // tensor_access provides base_address, offset, stride, extent
        auto taStruct = mlir::dyn_cast<StructValueAttr>(taField);
        if (!taStruct) {
          return ComptimeError::comptimeFailure(op, context, resultType)
                 << "tensor_access field must be a struct";
        }
        baseAddr = taStruct.getField("base_address");
        if (auto extentVal = getIntField("extent", 1); succeeded(extentVal))
          extent = static_cast<uint16_t>(*extentVal);
        if (TypedAttr extentField = taStruct.getField("extent")) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(extentField))
            extent = static_cast<uint16_t>(intAttr.getValue().getZExtValue());
        }
        if (TypedAttr strideField = taStruct.getField("stride")) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(strideField))
            stride = static_cast<int8_t>(intAttr.getValue().getSExtValue());
        }
        if (TypedAttr offsetField = taStruct.getField("offset")) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(offsetField))
            offset = static_cast<int16_t>(intAttr.getValue().getSExtValue());
        }
      } else {
        // Use individual fields
        baseAddr = configStruct.getField("base_address");
        if (auto extentVal = getIntField("extent", 1); succeeded(extentVal))
          extent = static_cast<uint16_t>(*extentVal);
        if (auto strideVal = getIntField("stride", 1); succeeded(strideVal))
          stride = static_cast<int8_t>(*strideVal);
        if (auto offsetVal = getIntField("offset", 0); succeeded(offsetVal))
          offset = static_cast<int16_t>(*offsetVal);
      }

      if (!baseAddr) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "mem1d_dsd requires tensor_access or explicit base_address, "
                  "extent, stride";
      }

      bool waveletIndexOffset = getBoolField("wavelet_index_offset", false);

      return cast<TypedAttr>(Mem1dDsdAttr::get(ctx, baseAddr, extent, stride,
                                               offset, waveletIndexOffset));
    }

    if (mlir::isa<Mem4dDsdType>(dsdType)) {
      // Check for tensor_access field
      TypedAttr taField = configStruct.getField("tensor_access");
      TypedAttr baseAddr;
      SmallVector<uint16_t> extents;
      SmallVector<int16_t> strides;
      int16_t offset = 0;

      if (taField) {
        auto taStruct = mlir::dyn_cast<StructValueAttr>(taField);
        if (!taStruct) {
          return ComptimeError::comptimeFailure(op, context, resultType)
                 << "tensor_access field must be a struct";
        }
        baseAddr = taStruct.getField("base_address");

        // Extract extents tuple
        if (TypedAttr extentField = taStruct.getField("extent")) {
          if (auto extentStruct =
                  mlir::dyn_cast<StructValueAttr>(extentField)) {
            for (TypedAttr e : extentStruct.getFieldValues()) {
              if (auto intAttr = mlir::dyn_cast<IntegerAttr>(e))
                extents.push_back(
                    static_cast<uint16_t>(intAttr.getValue().getZExtValue()));
            }
          } else if (auto intAttr = mlir::dyn_cast<IntegerAttr>(extentField)) {
            extents.push_back(
                static_cast<uint16_t>(intAttr.getValue().getZExtValue()));
          }
        }

        // Extract strides tuple
        if (TypedAttr strideField = taStruct.getField("stride")) {
          if (auto strideStruct =
                  mlir::dyn_cast<StructValueAttr>(strideField)) {
            for (TypedAttr s : strideStruct.getFieldValues()) {
              if (auto intAttr = mlir::dyn_cast<IntegerAttr>(s))
                strides.push_back(
                    static_cast<int16_t>(intAttr.getValue().getSExtValue()));
            }
          } else if (auto intAttr = mlir::dyn_cast<IntegerAttr>(strideField)) {
            strides.push_back(
                static_cast<int16_t>(intAttr.getValue().getSExtValue()));
          }
        }

        if (TypedAttr offsetField = taStruct.getField("offset")) {
          if (auto intAttr = mlir::dyn_cast<IntegerAttr>(offsetField))
            offset = static_cast<int16_t>(intAttr.getValue().getSExtValue());
        }
      } else {
        baseAddr = configStruct.getField("base_address");
        if (auto offsetVal = getIntField("offset", 0); succeeded(offsetVal))
          offset = static_cast<int16_t>(*offsetVal);
        // For mem4d without tensor_access, extents/strides come from config
        // This is a simplified implementation
      }

      if (!baseAddr) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "mem4d_dsd requires tensor_access or explicit base_address, "
                  "extent, stride";
      }

      bool waveletIndexOffset = getBoolField("wavelet_index_offset", false);

      return cast<TypedAttr>(Mem4dDsdAttr::get(ctx, baseAddr, extents, strides,
                                               offset, waveletIndexOffset));
    }

    if (mlir::isa<FabinDsdType>(dsdType)) {
      TypedAttr colorField = configStruct.getField("fabric_color");
      auto colorAttr = mlir::dyn_cast_if_present<ColorAttr>(colorField);

      InputQueueAttr inputQueue;
      if (TypedAttr iqField = configStruct.getField("input_queue"))
        inputQueue = mlir::dyn_cast<InputQueueAttr>(iqField);

      if (!colorAttr && !inputQueue)
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "fabin_dsd requires fabric_color or input_queue";

      if (!configStruct.getField("extent")) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "fabin_dsd requires extent";
      }
      auto extentVal = getIntField("extent", 1);
      if (failed(extentVal)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "fabin_dsd extent must be an integer";
      }
      uint16_t extent = static_cast<uint16_t>(*extentVal);

      // Optional fields
      PriorityAttr priority;
      if (TypedAttr pField = configStruct.getField("priority"))
        priority = mlir::dyn_cast<PriorityAttr>(pField);

      bool controlTransform = getBoolField("control_transform", false);

      SimdModeAttr simdMode;
      if (TypedAttr smField = configStruct.getField("simd_mode"))
        simdMode = getSimdModeAttr(smField);

      return cast<TypedAttr>(FabinDsdAttr::get(ctx, colorAttr, extent,
                                               inputQueue, priority,
                                               controlTransform, simdMode));
    }

    if (mlir::isa<FaboutDsdType>(dsdType)) {
      TypedAttr colorField = configStruct.getField("fabric_color");
      auto colorAttr = mlir::dyn_cast_if_present<ColorAttr>(colorField);

      OutputQueueAttr outputQueue;
      if (TypedAttr oqField = configStruct.getField("output_queue"))
        outputQueue = mlir::dyn_cast<OutputQueueAttr>(oqField);

      if (!colorAttr && !outputQueue)
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "fabout_dsd requires fabric_color or output_queue";

      if (!configStruct.getField("extent")) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "fabout_dsd requires extent";
      }
      auto extentVal = getIntField("extent", 1);
      if (failed(extentVal)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "fabout_dsd extent must be an integer";
      }
      uint16_t extent = static_cast<uint16_t>(*extentVal);

      // Optional fields
      bool control = getBoolField("control", false);
      bool waveletIndexOffset = getBoolField("wavelet_index_offset", false);
      bool advanceSwitch = getBoolField("advance_switch", false);
      bool controlTransform = getBoolField("control_transform", false);

      SimdModeAttr simdMode;
      if (TypedAttr smField = configStruct.getField("simd_mode"))
        simdMode = getSimdModeAttr(smField);

      ZeroAttr zero;
      if (TypedAttr zField = configStruct.getField("zero"))
        zero = mlir::dyn_cast<ZeroAttr>(zField);

      return cast<TypedAttr>(FaboutDsdAttr::get(
          ctx, colorAttr, extent, outputQueue, control, waveletIndexOffset,
          advanceSwitch, controlTransform, simdMode, zero));
    }

    // FIFO DSD is created via @allocate_fifo, not @get_dsd

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "unsupported DSD type for @get_dsd";
  }
};

//===----------------------------------------------------------------------===//
// AllocateFifoEvaluator - @allocate_fifo builtin
//===----------------------------------------------------------------------===//

struct AllocateFifoEvaluator
    : public BuiltinEvaluatorDefaults<AllocateFifoEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    MLIRContext *ctx = op->getContext();
    Type resultType = FifoDsdType::get(ctx);

    if (auto error = defaultValidation(op, context))
      return *error;

    // First argument: buffer (should be a global array reference)
    FailureOr<LValueDescriptor> bufferLValue =
        context.getComptimeLValue(args[0]);
    if (failed(bufferLValue) || !bufferLValue->isGlobalBased()) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "@allocate_fifo buffer must be a global array";
    }

    GlobalOp globalOp = bufferLValue->getBaseGlobal();
    std::optional<uint64_t> moduleId = std::nullopt;
    if (auto module = globalOp->getParentOfType<parsed_csl::ModuleOp>()) {
      moduleId = module.getModuleIdentifier();
    }

    auto bufferLValueType = llvm::dyn_cast_if_present<LValueType>(
        *context.getInferredType(args[0]));
    auto bufferRef = GlobalRefAttr::get(
        ctx, moduleId, FlatSymbolRefAttr::get(globalOp.getSymNameAttr()),
        bufferLValueType);

    // Optional configuration fields
    FuncRefAttr activatePop;
    FuncRefAttr activatePush;
    PriorityAttr priority;
    SimdModeAttr simdMode;

    // Second argument: configuration struct (optional)
    if (args.size() > 1) {
      auto configValue = context.getOrLoadComptimeRValue(args[1]);
      if (failed(configValue)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(args[1])
               << "FIFO configuration struct is not comptime-known";
      }

      auto configStruct = mlir::dyn_cast<StructValueAttr>(*configValue);
      if (!configStruct) {
        return ComptimeError::comptimeFailure(op, context, resultType)
               << "FIFO configuration must be a struct";
      }

      // Extract optional fields from config struct
      if (TypedAttr popField = configStruct.getField("activate_pop"))
        activatePop = mlir::dyn_cast<FuncRefAttr>(popField);

      if (TypedAttr pushField = configStruct.getField("activate_push"))
        activatePush = mlir::dyn_cast<FuncRefAttr>(pushField);

      if (TypedAttr pField = configStruct.getField("priority"))
        priority = mlir::dyn_cast<PriorityAttr>(pField);

      if (TypedAttr smField = configStruct.getField("simd_mode"))
        simdMode = getSimdModeAttr(smField);
    }

    return cast<TypedAttr>(FifoDsdAttr::get(ctx, bufferRef, activatePop,
                                            activatePush, priority, simdMode));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    // @allocate_fifo is comptime-only
    return true;
  }
};

//===----------------------------------------------------------------------===//
// LoadToDsrEvaluator - @load_to_dsr builtin
//===----------------------------------------------------------------------===//

struct LoadToDsrEvaluator
    : public BuiltinEvaluatorDefaults<LoadToDsrEvaluator> {
  // Uses default validateSemantics (signature matcher)
  // Uses default inferResultType (void)
  // Uses default evaluateAtComptime (records for final comptime)
};

struct LoadToDsrXdsrSrEvaluator
    : public BuiltinEvaluatorDefaults<LoadToDsrXdsrSrEvaluator> {
  // Uses descriptor validation and is retained for final-comptime or runtime
  // evaluation, just like @load_to_dsr.
};

//===----------------------------------------------------------------------===//
// RangeEvaluator - @range builtin
//===----------------------------------------------------------------------===//

struct RangeEvaluator : public BuiltinEvaluatorDefaults<RangeEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();

    if (auto error = defaultValidation(op, context))
      return *error;

    // Get element type from first arg (must be TypeAttr)
    auto typeValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(typeValue)) {
      return ComptimeError::semanticError(op, context)
             << "element type is not comptime-known";
    }
    Type elemType = cast<csl::TypeAttr>(*typeValue).getValue();
    Type resultType = csl::RangeType::get(op->getContext(), elemType);

    TypedAttr startAttr, stopAttr, stepAttr;

    if (args.size() == 4) {
      // Full form: @range(type, start, stop, step)
      auto startVal = context.getOrLoadComptimeRValue(args[1]);
      auto stopVal = context.getOrLoadComptimeRValue(args[2]);
      auto stepVal = context.getOrLoadComptimeRValue(args[3]);

      if (failed(startVal)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(args[1])
               << "start is not comptime-known";
      }
      if (failed(stopVal)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(args[2])
               << "stop is not comptime-known";
      }
      if (failed(stepVal)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(args[3])
               << "step is not comptime-known";
      }

      startAttr = *startVal;
      stopAttr = *stopVal;
      stepAttr = *stepVal;
    } else {
      // Short form: @range(type, stop) - start=0, step=1
      auto stopVal = context.getOrLoadComptimeRValue(args[1]);
      if (failed(stopVal)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(args[1])
               << "stop is not comptime-known";
      }

      // Create zero and one for start and step
      startAttr = csl::ComptimeIntAttr::get(op->getContext(), APInt(64, 0));
      stopAttr = *stopVal;
      stepAttr = csl::ComptimeIntAttr::get(op->getContext(), APInt(64, 1));
    }

    // Cast all values to elemType to validate they fit
    InFlightDiagnostic diag = op->emitRemark();
    auto castStart = csl::evaluateCast(startAttr, elemType, &diag);
    auto castStop = csl::evaluateCast(stopAttr, elemType, &diag);
    auto castStep = csl::evaluateCast(stepAttr, elemType, &diag);
    diag.abandon();

    if (failed(castStart) || failed(castStop) || failed(castStep)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast range values to element type";
    }

    // Extract int64_t values from the casted attributes
    // Handle both IntegerAttr (for runtime integer types) and
    // ComptimeIntAttr (for comptime_int type)
    auto extractInt = [](TypedAttr attr) -> int64_t {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr))
        return intAttr.getValue().getSExtValue();
      if (auto comptimeAttr = mlir::dyn_cast<csl::ComptimeIntAttr>(attr))
        return comptimeAttr.getValue().getSExtValue();
      llvm_unreachable("expected IntegerAttr or ComptimeIntAttr");
    };

    int64_t start = extractInt(*castStart);
    int64_t stop = extractInt(*castStop);
    int64_t step = extractInt(*castStep);

    return cast<TypedAttr>(
        csl::RangeAttr::get(op->getContext(), elemType, start, stop, step));
  }
};

//===----------------------------------------------------------------------===//
// RangeAccessorEvaluator - @range_start, @range_stop, @range_step builtins
//===----------------------------------------------------------------------===//

enum class RangeAccessor { Start, Stop, Step };

template <RangeAccessor Accessor>
struct RangeAccessorEvaluator
    : public BuiltinEvaluatorDefaults<RangeAccessorEvaluator<Accessor>> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    using Base = BuiltinEvaluatorDefaults<RangeAccessorEvaluator<Accessor>>;
    if (auto error = Base::defaultValidation(op, context))
      return *error;

    // Try to get the range type for inferring result type
    FailureOr<Type> argType =
        context.getInferredRValueType(builtinCallOp.getArgs()[0]);
    Type resultType;
    if (succeeded(argType)) {
      if (auto rangeType = mlir::dyn_cast<csl::RangeType>(*argType))
        resultType = rangeType.getElementType();
    }

    auto rangeValue =
        context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(rangeValue)) {
      if (!resultType) {
        return ComptimeError::semanticError(op, context)
               << "cannot infer result type of range accessor";
      }
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "range is not comptime-known";
    }

    auto rangeAttr = cast<csl::RangeAttr>(*rangeValue);
    Type elemType = rangeAttr.getElementType();

    int64_t value;
    if constexpr (Accessor == RangeAccessor::Start)
      value = rangeAttr.getStart();
    else if constexpr (Accessor == RangeAccessor::Stop)
      value = rangeAttr.getStop();
    else
      value = rangeAttr.getStep();

    // Return as IntegerAttr with the element type
    auto intType = mlir::cast<IntegerType>(elemType);
    return cast<TypedAttr>(IntegerAttr::get(
        intType, APInt(intType.getWidth(), value, intType.isSigned())));
  }
};

// Convenience type aliases
using RangeStartEvaluator = RangeAccessorEvaluator<RangeAccessor::Start>;
using RangeStopEvaluator = RangeAccessorEvaluator<RangeAccessor::Stop>;
using RangeStepEvaluator = RangeAccessorEvaluator<RangeAccessor::Step>;

//===----------------------------------------------------------------------===//
// ArrayInitEvaluator - @zeros and @constants builtins
//===----------------------------------------------------------------------===//

struct ArrayInitEvaluator
    : public BuiltinEvaluatorDefaults<ArrayInitEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    BuiltinKind kind = *builtinCallOp.resolveBuiltin();

    if (auto error = defaultValidation(op, context))
      return *error;

    // Get the type argument (first arg)
    auto typeValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(typeValue)) {
      return ComptimeError::semanticError(op, context)
             << "type argument is not comptime-known";
    }

    auto typeAttr = dyn_cast<csl::TypeAttr>(*typeValue);
    if (!typeAttr) {
      return ComptimeError::semanticError(op, context)
             << "first argument must be a type";
    }

    auto arrayType = dyn_cast<csl::ArrayType>(typeAttr.getValue());
    if (!arrayType) {
      return ComptimeError::semanticError(op, context)
             << "@" << csl::stringifyBuiltinKind(kind)
             << " requires an array type argument, got " << typeAttr.getValue();
    }

    // Result type is the array type
    Type resultType = arrayType;
    Type elemType = arrayType.getElementType();
    TypedAttr fillValue;

    if (kind == BuiltinKind::Zeros) {
      // @zeros: create default zero value for element type
      fillValue = getDefaultZero(elemType, op->getContext());
      if (!fillValue) {
        return ComptimeError::semanticError(op, context).withType(resultType)
               << "@zeros: cannot create zero value for element type "
               << elemType;
      }
    } else {
      // @constants: get fill value from second argument
      assert(kind == BuiltinKind::Constants);
      auto fillAttr = context.getOrLoadComptimeRValue(args[1]);
      if (failed(fillAttr)) {
        return ComptimeError::comptimeFailure(op, context, resultType)
                   .withDependency(args[1])
               << "fill value is not comptime-known";
      }

      // Cast fill value to element type if needed
      if (fillAttr->getType() != elemType) {
        InFlightDiagnostic diag = op->emitRemark();
        auto castedAttr = csl::evaluateImplicitCast(*fillAttr, elemType, &diag);
        diag.abandon();
        if (failed(castedAttr)) {
          return ComptimeError::semanticError(op, context).withType(resultType)
                 << "@constants: failed to cast fill value from "
                 << csl::prettyPrintType(fillAttr->getType()) << " to "
                 << csl::prettyPrintType(elemType);
        }
        fillValue = *castedAttr;
      } else {
        fillValue = *fillAttr;
      }
    }

    // Create array filled with the fill value
    int64_t numElements = arrayType.getNumElements();
    SmallVector<TypedAttr> elements(numElements, fillValue);

    return cast<TypedAttr>(csl::ArrayValueAttr::get(
        op->getContext(), arrayType.getShape(), elemType, elements));
  }

private:
  // Helper to create a default zero value for a given type
  static TypedAttr getDefaultZero(Type type, MLIRContext *ctx) {
    if (auto intType = dyn_cast<IntegerType>(type)) {
      return IntegerAttr::get(type, APInt(intType.getWidth(), 0));
    }
    if (auto floatType = dyn_cast<FloatType>(type)) {
      return FloatAttr::get(type,
                            APFloat::getZero(floatType.getFloatSemantics()));
    }
    if (mlir::isa<csl::ComptimeIntType>(type)) {
      return csl::ComptimeIntAttr::get(ctx, APInt(64, 0));
    }
    if (mlir::isa<csl::ComptimeFloatType>(type)) {
      return csl::ComptimeFloatAttr::get(ctx, APFloat(0.0));
    }
    // Note: Boolean (i1) is handled by the IntegerType case above
    // Could add more types (nested arrays, etc.) if needed
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// GetIntEvaluator - @get_int builtin
//===----------------------------------------------------------------------===//

struct GetIntEvaluator : public BuiltinEvaluatorDefaults<GetIntEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    // ========== PHASE 1: Validation ==========
    if (builtinCallOp.getArgs().size() != 1) {
      return ComptimeError::semanticError(op, context)
             << "@get_int expects exactly one argument";
    }

    // Get the argument's inferred rvalue type (unwraps LValueType)
    auto argType = context.getInferredRValueType(builtinCallOp.getArgs()[0]);
    if (failed(argType)) {
      return ComptimeError::semanticError(op, context)
             << "@get_int: failed to infer argument type";
    }

    // Check for supported types
    if (!mlir::isa<EnumType, IntegerType, ColorType, ControlTaskIdType,
                   DataTaskIdType, InputQueueType, OutputQueueType,
                   LocalTaskIdType, UtIdType>(*argType)) {
      return ComptimeError::semanticError(op, context)
             << "@get_int: unsupported argument type "
             << csl::prettyPrintType(*argType);
    }

    // ========== PHASE 2: Type Inference ==========
    IntegerType u16Type =
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned);

    // Determine result type based on argument type
    Type resultType = u16Type; // default for most cases
    if (auto enumType = mlir::dyn_cast<EnumType>(*argType)) {
      resultType = enumType.getUnderlyingType();
    } else if (auto intType = mlir::dyn_cast<IntegerType>(*argType)) {
      resultType = intType;
    }

    // ========== PHASE 3: Comptime Evaluation ==========
    // Get the argument value
    auto argValue = context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(builtinCallOp.getArgs()[0])
             << "argument is not comptime-known";
    }

    // Enum: return underlying int value
    if (auto enumValue = mlir::dyn_cast<EnumValueAttr>(*argValue)) {
      IntegerType underlyingType = enumValue.getEnumType().getUnderlyingType();
      return cast<TypedAttr>(
          IntegerAttr::get(underlyingType, APInt(underlyingType.getWidth(),
                                                 enumValue.getIntValue(),
                                                 underlyingType.isSigned())));
    }

    // Integer: return unchanged
    if (mlir::isa<IntegerAttr>(*argValue))
      return *argValue;

    // Color: extract u16 value
    if (auto colorAttr = mlir::dyn_cast<ColorAttr>(*argValue))
      return cast<TypedAttr>(IntegerAttr::get(u16Type, colorAttr.getValue()));

    // LocalTaskId: extract u16 value
    if (auto taskIdAttr = mlir::dyn_cast<LocalTaskIdAttr>(*argValue))
      return cast<TypedAttr>(IntegerAttr::get(u16Type, taskIdAttr.getValue()));

    // ControlTaskId: extract u16 value
    if (auto taskIdAttr = mlir::dyn_cast<ControlTaskIdAttr>(*argValue))
      return cast<TypedAttr>(IntegerAttr::get(u16Type, taskIdAttr.getValue()));

    // UtId: extract u16 value
    if (auto utIdAttr = mlir::dyn_cast<UtIdAttr>(*argValue))
      return cast<TypedAttr>(IntegerAttr::get(u16Type, utIdAttr.getValue()));

    // InputQueue: extract u8 value, return as u16
    if (auto queueAttr = mlir::dyn_cast<InputQueueAttr>(*argValue))
      return cast<TypedAttr>(IntegerAttr::get(u16Type, queueAttr.getValue()));

    // OutputQueue: extract u8 value, return as u16
    if (auto queueAttr = mlir::dyn_cast<OutputQueueAttr>(*argValue))
      return cast<TypedAttr>(IntegerAttr::get(u16Type, queueAttr.getValue()));

    // DataTaskId: extract value from source (color or input_queue)
    if (auto dataTaskIdAttr = mlir::dyn_cast<DataTaskIdAttr>(*argValue)) {
      if (auto colorAttr = dataTaskIdAttr.getColor())
        return cast<TypedAttr>(IntegerAttr::get(u16Type, colorAttr.getValue()));
      if (auto queueAttr = dataTaskIdAttr.getInputQueue())
        return cast<TypedAttr>(IntegerAttr::get(u16Type, queueAttr.getValue()));
    }

    return ComptimeError::semanticError(op, context)
           << "@get_int: unsupported argument type";
  }
};

//===----------------------------------------------------------------------===//
// Fp16
//===----------------------------------------------------------------------===//

struct Fp16Evaluator : public BuiltinEvaluatorDefaults<Fp16Evaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    MLIRContext *ctx = op->getContext();
    Type fp16Type;
    switch (context.getGlobalEvalContext()->getOptions().cslFlags.fp16Format) {
    case csl::Fp16Format::F16:
      fp16Type = mlir::Float16Type::get(ctx);
      break;
    case csl::Fp16Format::BF16:
      fp16Type = mlir::BFloat16Type::get(ctx);
      break;
    case csl::Fp16Format::CB16:
      fp16Type = csl::CB16Type::get(ctx);
      break;
    }
    return cast<TypedAttr>(csl::TypeAttr::get(ctx, fp16Type));
  }
};

//===----------------------------------------------------------------------===//
// HasField
//===----------------------------------------------------------------------===//

struct HasFieldEvaluator : public BuiltinEvaluatorDefaults<HasFieldEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    Type resultType = IntegerType::get(op->getContext(), 1);

    if (auto error = defaultValidation(op, context))
      return *error;

    FailureOr<Type> operandType =
        context.getInferredRValueType(builtinCallOp.getArgs()[0]);
    if (failed(operandType)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to infer operand type";
    }

    FailureOr<TypedAttr> fieldNameValue =
        context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[1]);
    if (failed(fieldNameValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "field name is not comptime-known";
    }

    auto fieldNameAttr =
        mlir::dyn_cast<csl::ComptimeStringAttr>(*fieldNameValue);
    if (!fieldNameAttr) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "field name must be a string";
    }

    StringRef fieldName = fieldNameAttr.getValue();
    bool hasField = false;

    if (auto structType = mlir::dyn_cast<StructType>(*operandType)) {
      hasField = llvm::is_contained(structType.getFieldNames(), fieldName);
    } else {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "operand type must be a struct";
    }

    return cast<TypedAttr>(BoolAttr::get(op->getContext(), hasField));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// BitcastEvaluator
//===----------------------------------------------------------------------===//

struct BitcastEvaluator : public BuiltinEvaluatorDefaults<BitcastEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();

    if (auto error = defaultValidation(op, context))
      return *error;

    // First argument is the target type (must be comptime-known).
    FailureOr<TypedAttr> typeValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(typeValue)) {
      return ComptimeError::semanticError(op, context)
             << "target type is not comptime-known";
    }
    auto typeAttr = mlir::dyn_cast<csl::TypeAttr>(*typeValue);
    if (!typeAttr) {
      return ComptimeError::semanticError(op, context)
             << "first argument must be a type";
    }
    Type targetType = typeAttr.getValue();

    // Source must be numeric.
    FailureOr<Type> srcType = context.getInferredRValueType(args[1]);
    if (failed(srcType)) {
      return ComptimeError::comptimeFailure(op, context, targetType)
                 .withDependency(args[1])
             << "failed to infer source type";
    }

    auto srcIntType = mlir::dyn_cast<IntegerType>(*srcType);
    auto srcFloatType = mlir::dyn_cast<FloatType>(*srcType);
    if (!srcIntType && !srcFloatType) {
      return ComptimeError::semanticError(op, context)
             << "bitcast source must be a numeric type";
    }
    unsigned srcBits =
        srcIntType ? srcIntType.getWidth() : srcFloatType.getWidth();

    // Target can be pointer, integer, or float.
    auto targetIntType = mlir::dyn_cast<IntegerType>(targetType);
    auto targetFloatType = mlir::dyn_cast<FloatType>(targetType);
    bool targetIsPtr = mlir::isa<csl::PointerType>(targetType);

    if (!targetIntType && !targetFloatType && !targetIsPtr) {
      return ComptimeError::semanticError(op, context)
             << "bitcast target must be a pointer, integer, or float type";
    }

    // Pointer targets require integer source; bit width check is deferred
    // to runtime.
    if (targetIsPtr) {
      if (!srcIntType) {
        return ComptimeError::semanticError(op, context)
               << "bitcast to pointer requires an integer source";
      }
      return ComptimeError::comptimeFailure(op, context, targetType)
             << "pointer bitcast cannot be evaluated at comptime";
    }

    unsigned targetBits =
        targetIntType ? targetIntType.getWidth() : targetFloatType.getWidth();
    if (targetBits != srcBits) {
      return ComptimeError::semanticError(op, context)
             << "bitcast requires matching bit widths, got " << srcBits
             << " and " << targetBits;
    }

    // Try to evaluate at comptime.
    FailureOr<TypedAttr> srcValue = context.getOrLoadComptimeRValue(args[1]);
    if (failed(srcValue)) {
      return ComptimeError::comptimeFailure(op, context, targetType)
                 .withDependency(args[1])
             << "source value is not comptime-known";
    }

    // Extract raw bits from source.
    APInt rawBits;
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(*srcValue)) {
      rawBits = intAttr.getValue();
    } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(*srcValue)) {
      rawBits = csl::floatBitcastToAPInt(floatAttr.getValue(), srcFloatType);
    } else {
      return ComptimeError::comptimeFailure(op, context, targetType)
             << "unsupported source value for comptime bitcast";
    }

    // Construct result with target type.
    if (targetIntType)
      return cast<TypedAttr>(IntegerAttr::get(targetIntType, rawBits));
    return cast<TypedAttr>(FloatAttr::get(
        targetFloatType, csl::apIntBitcastToFloat(rawBits, targetFloatType)));
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false;
  }
};

//===----------------------------------------------------------------------===//
// FieldEvaluator
//===----------------------------------------------------------------------===//

struct FieldEvaluator : public BuiltinEvaluatorDefaults<FieldEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);

    if (auto error = defaultValidation(op, context))
      return *error;

    // Get the field name (must be comptime-known).
    FailureOr<TypedAttr> fieldNameValue =
        context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[1]);
    if (failed(fieldNameValue)) {
      return ComptimeError::semanticError(op, context)
             << "field name is not comptime-known";
    }
    auto fieldNameAttr =
        mlir::dyn_cast<csl::ComptimeStringAttr>(*fieldNameValue);
    if (!fieldNameAttr) {
      return ComptimeError::semanticError(op, context)
             << "field name must be a comptime_string";
    }
    StringRef fieldName = fieldNameAttr.getValue();

    // Get the base type, preserving LValueType for constness tracking.
    FailureOr<Type> baseType =
        context.getInferredType(builtinCallOp.getArgs()[0]);
    if (failed(baseType)) {
      return ComptimeError::semanticError(op, context)
             << "failed to infer struct operand type";
    }

    // Unpack lvalue if needed, but keep lvalue type info for constness.
    Type valueType = *baseType;
    auto lvalueType = mlir::dyn_cast<LValueType>(valueType);
    if (lvalueType)
      valueType = lvalueType.getValueType();

    // Helper to look up a field in a StructValueAttr by name, handling tuples
    // by parsing the name as a numeric index.
    auto lookupField = [](StructValueAttr structAttr,
                          StringRef name) -> TypedAttr {
      // Try named lookup first.
      TypedAttr fieldValue = structAttr.getField(name);
      if (fieldValue)
        return fieldValue;
      // For tuples, parse the name as a numeric index.
      if (structAttr.isTuple()) {
        unsigned index;
        if (!name.getAsInteger(10, index) && index < structAttr.getNumFields())
          return structAttr.getField(index);
      }
      return TypedAttr();
    };

    // Helper to look up a field type in a StructType by name, handling tuples.
    auto lookupFieldType = [](StructType structType, StringRef name) -> Type {
      Type fieldType = structType.getFieldType(name);
      if (fieldType)
        return fieldType;
      if (structType.isTuple()) {
        unsigned index;
        if (!name.getAsInteger(10, index) && index < structType.getNumFields())
          return structType.getFieldType(index);
      }
      return Type();
    };

    // Determine the field type based on the struct type.
    Type resultType;
    if (auto structType = mlir::dyn_cast<StructType>(valueType)) {
      Type fieldType = lookupFieldType(structType, fieldName);
      if (!fieldType) {
        return ComptimeError::semanticError(op, context)
               << "struct has no field named '" << fieldName << "'";
      }
      // If base was an lvalue, return an lvalue type preserving constness.
      if (lvalueType) {
        resultType = LValueType::get(fieldType, lvalueType.getIsConst());
      } else {
        resultType = fieldType;
      }
    } else {
      return ComptimeError::semanticError(op, context)
             << "operand type must be a struct";
    }

    // If base is an lvalue, try to return an lvalue descriptor for the field.
    // This allows @field(var, "name") = value to work as assignment.
    if (lvalueType) {
      FailureOr<LValueDescriptor> lvalueDescr =
          context.getComptimeLValue(builtinCallOp.getArgs()[0]);
      if (succeeded(lvalueDescr)) {
        Type fieldType = mlir::cast<LValueType>(resultType).getValueType();
        // For tuples, the field name is a numeric string (e.g., "0", "1").
        // Use IndexAccess instead of FieldAccess since tuples have no named
        // fields and loadLValue resolves IndexAccess by position.
        unsigned tupleIndex;
        if (!fieldName.getAsInteger(10, tupleIndex))
          return lvalueDescr->extendWithIndex(static_cast<int64_t>(tupleIndex),
                                              fieldType);
        return lvalueDescr->extendWithField(fieldName, fieldType, context);
      }
    }

    // Try to get comptime rvalue.
    FailureOr<TypedAttr> structValue =
        context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (succeeded(structValue)) {
      auto structAttr = mlir::dyn_cast<StructValueAttr>(*structValue);
      if (structAttr) {
        TypedAttr fieldValue = lookupField(structAttr, fieldName);
        if (fieldValue)
          return fieldValue;
      }
    }

    // Not comptime-evaluable — return type only.
    return ComptimeError::comptimeFailure(op, context, resultType)
               .withDependency(builtinCallOp.getArgs()[0])
           << "struct value is not comptime-known";
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return false;
  }
};

//===----------------------------------------------------------------------===//
// SetDsdStrideEvaluator - @set_dsd_stride builtin
//===----------------------------------------------------------------------===//

struct SetDsdStrideEvaluator
    : public BuiltinEvaluatorDefaults<SetDsdStrideEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    MLIRContext *ctx = op->getContext();

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);
    Type resultType = validation.resultType;

    auto dsdValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(dsdValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[0])
             << "DSD argument is not comptime-known";
    }

    auto strideValue = context.getOrLoadComptimeRValue(args[1]);
    if (failed(strideValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[1])
             << "stride argument is not comptime-known";
    }

    IntegerType i8Type = IntegerType::get(ctx, 8, IntegerType::Signed);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedStride =
        csl::evaluateCast(*strideValue, i8Type, &diag);
    if (failed(castedStride)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast stride to i8" << std::move(diag);
    }
    diag.abandon();

    int8_t stride = static_cast<int8_t>(
        mlir::cast<IntegerAttr>(*castedStride).getValue().getSExtValue());

    auto mem1dDsd = mlir::cast<Mem1dDsdAttr>(*dsdValue);
    return cast<TypedAttr>(Mem1dDsdAttr::get(
        ctx, mem1dDsd.getBaseAddress(), mem1dDsd.getExtent(), stride,
        mem1dDsd.getOffset(), mem1dDsd.getWaveletIndexOffset()));
  }
};

//===----------------------------------------------------------------------===//
// SetDsdLengthEvaluator - @set_dsd_length builtin
//===----------------------------------------------------------------------===//

struct SetDsdLengthEvaluator
    : public BuiltinEvaluatorDefaults<SetDsdLengthEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    MLIRContext *ctx = op->getContext();

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);
    Type resultType = validation.resultType;

    auto dsdValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(dsdValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[0])
             << "DSD argument is not comptime-known";
    }

    auto lengthValue = context.getOrLoadComptimeRValue(args[1]);
    if (failed(lengthValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[1])
             << "length argument is not comptime-known";
    }

    IntegerType u16Type = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedLength =
        csl::evaluateCast(*lengthValue, u16Type, &diag);
    if (failed(castedLength)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast length to u16" << std::move(diag);
    }
    diag.abandon();

    uint16_t length = static_cast<uint16_t>(
        mlir::cast<IntegerAttr>(*castedLength).getValue().getZExtValue());

    if (auto mem1dDsd = mlir::dyn_cast<Mem1dDsdAttr>(*dsdValue)) {
      return cast<TypedAttr>(Mem1dDsdAttr::get(
          ctx, mem1dDsd.getBaseAddress(), length, mem1dDsd.getStride(),
          mem1dDsd.getOffset(), mem1dDsd.getWaveletIndexOffset()));
    }

    if (auto fabinDsd = mlir::dyn_cast<FabinDsdAttr>(*dsdValue)) {
      return cast<TypedAttr>(FabinDsdAttr::get(
          ctx, fabinDsd.getFabricColor(), length, fabinDsd.getInputQueue(),
          fabinDsd.getPriority(), fabinDsd.getControlTransform(),
          fabinDsd.getSimdMode()));
    }

    if (auto faboutDsd = mlir::dyn_cast<FaboutDsdAttr>(*dsdValue)) {
      return cast<TypedAttr>(FaboutDsdAttr::get(
          ctx, faboutDsd.getFabricColor(), length, faboutDsd.getOutputQueue(),
          faboutDsd.getControl(), faboutDsd.getWaveletIndexOffset(),
          faboutDsd.getAdvanceSwitch(), faboutDsd.getControlTransform(),
          faboutDsd.getSimdMode(), faboutDsd.getZero()));
    }

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "comptime evaluation of @set_dsd_length not supported for this "
              "DSD type";
  }
};

//===----------------------------------------------------------------------===//
// SetDsdBaseAddrEvaluator - @set_dsd_base_addr builtin
//===----------------------------------------------------------------------===//

struct SetDsdBaseAddrEvaluator
    : public BuiltinEvaluatorDefaults<SetDsdBaseAddrEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    MLIRContext *ctx = op->getContext();

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);
    Type resultType = validation.resultType;

    auto dsdValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(dsdValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[0])
             << "DSD argument is not comptime-known";
    }

    auto addrValue = context.getOrLoadComptimeRValue(args[1]);
    if (failed(addrValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[1])
             << "base address argument is not comptime-known";
    }

    if (auto mem1dDsd = mlir::dyn_cast<Mem1dDsdAttr>(*dsdValue)) {
      return cast<TypedAttr>(Mem1dDsdAttr::get(
          ctx, *addrValue, mem1dDsd.getExtent(), mem1dDsd.getStride(),
          mem1dDsd.getOffset(), mem1dDsd.getWaveletIndexOffset()));
    }

    if (auto mem4dDsd = mlir::dyn_cast<Mem4dDsdAttr>(*dsdValue)) {
      return cast<TypedAttr>(Mem4dDsdAttr::get(
          ctx, *addrValue, mem4dDsd.getExtents(), mem4dDsd.getStrides(),
          mem4dDsd.getOffset(), mem4dDsd.getWaveletIndexOffset()));
    }

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "comptime evaluation of @set_dsd_base_addr not supported for "
              "this DSD type";
  }
};

//===----------------------------------------------------------------------===//
// IncrementDsdOffsetEvaluator - @increment_dsd_offset builtin
//===----------------------------------------------------------------------===//

struct IncrementDsdOffsetEvaluator
    : public BuiltinEvaluatorDefaults<IncrementDsdOffsetEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto args = builtinCallOp.getArgs();
    MLIRContext *ctx = op->getContext();

    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);
    Type resultType = validation.resultType;

    auto dsdValue = context.getOrLoadComptimeRValue(args[0]);
    if (failed(dsdValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[0])
             << "DSD argument is not comptime-known";
    }

    auto offsetValue = context.getOrLoadComptimeRValue(args[1]);
    if (failed(offsetValue)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
                 .withDependency(args[1])
             << "offset argument is not comptime-known";
    }

    // Get the element type (third argument) to compute the word-unit scale
    // factor.
    // @increment_dsd_offset(dsd, k, T) increments the DSD's base address by
    // k * sizeof(T) bytes. Since Mem1dDsdAttr.offset is in 2-byte (i16) word
    // units, the offset increment in word units is k * sizeof(T) / 2.
    auto typeValue = context.getOrLoadComptimeRValue(args[2]);
    if (failed(typeValue)) {
      return ComptimeError::semanticError(op, context)
             << "element type argument is not comptime-known";
    }
    Type elemType = mlir::cast<csl::TypeAttr>(*typeValue).getValue();
    std::optional<uint32_t> elemSizeBytes = csl::getSizeInBytes(elemType);
    if (!elemSizeBytes || *elemSizeBytes < 2 || *elemSizeBytes % 2 != 0) {
      return ComptimeError::semanticError(op, context)
             << "element type has unknown or invalid size for DSD offset";
    }
    int16_t wordScaleFactor = static_cast<int16_t>(*elemSizeBytes / 2);

    IntegerType i16Type = IntegerType::get(ctx, 16, IntegerType::Signed);
    InFlightDiagnostic diag = op->emitRemark();
    FailureOr<TypedAttr> castedOffset =
        csl::evaluateCast(*offsetValue, i16Type, &diag);
    if (failed(castedOffset)) {
      return ComptimeError::comptimeFailure(op, context, resultType)
             << "failed to cast offset to i16" << std::move(diag);
    }
    diag.abandon();

    int16_t offsetIncrement = static_cast<int16_t>(
        mlir::cast<IntegerAttr>(*castedOffset).getValue().getSExtValue());
    offsetIncrement = static_cast<int16_t>(offsetIncrement * wordScaleFactor);

    if (auto mem1dDsd = mlir::dyn_cast<Mem1dDsdAttr>(*dsdValue)) {
      int16_t newOffset = mem1dDsd.getOffset() + offsetIncrement;
      return cast<TypedAttr>(Mem1dDsdAttr::get(
          ctx, mem1dDsd.getBaseAddress(), mem1dDsd.getExtent(),
          mem1dDsd.getStride(), newOffset, mem1dDsd.getWaveletIndexOffset()));
    }

    if (auto mem4dDsd = mlir::dyn_cast<Mem4dDsdAttr>(*dsdValue)) {
      int16_t newOffset = mem4dDsd.getOffset() + offsetIncrement;
      return cast<TypedAttr>(Mem4dDsdAttr::get(
          ctx, mem4dDsd.getBaseAddress(), mem4dDsd.getExtents(),
          mem4dDsd.getStrides(), newOffset, mem4dDsd.getWaveletIndexOffset()));
    }

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "comptime evaluation of @increment_dsd_offset not supported for "
              "this DSD type";
  }
};

//===----------------------------------------------------------------------===//
// DsdModifierDefaultEvaluator - for builtins that can't be comptime-evaluated
// (e.g. @set_fifo_read_length, @set_fifo_write_length)
//===----------------------------------------------------------------------===//

struct DsdModifierDefaultEvaluator
    : public BuiltinEvaluatorDefaults<DsdModifierDefaultEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    return ComptimeError::comptimeFailure(op, context, validation.resultType)
           << "DSD modifier cannot be evaluated at comptime";
  }
};

//===----------------------------------------------------------------------===//
// PromoteToConstGlobalEvaluator - @__internal_promote_to_const_global builtin
//===----------------------------------------------------------------------===//

struct PromoteToConstGlobalEvaluator
    : public BuiltinEvaluatorDefaults<PromoteToConstGlobalEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    MLIRContext *ctx = op->getContext();

    // Validate: exactly one argument, must be comptime_string
    if (builtinCallOp.getArgs().size() != 1) {
      return ComptimeError::semanticError(op, context)
             << "@__internal_promote_to_const_global expects exactly 1 "
                "argument";
    }

    FailureOr<TypedAttr> argValue =
        context.getOrLoadComptimeRValue(builtinCallOp.getArgs()[0]);
    if (failed(argValue)) {
      return ComptimeError::semanticError(op, context)
             << "argument must be a comptime-known string";
    }
    auto strAttr = mlir::dyn_cast<ComptimeStringAttr>(*argValue);
    if (!strAttr) {
      return ComptimeError::semanticError(op, context)
             << "argument must be a comptime_string";
    }

    // Convert string to array of u8 bytes
    StringRef strValue = strAttr.getValue();
    auto u8Type = IntegerType::get(ctx, 8, IntegerType::Unsigned);
    SmallVector<TypedAttr> elements;
    for (char c : strValue)
      elements.push_back(IntegerAttr::get(u8Type, static_cast<uint8_t>(c)));

    int64_t len = strValue.size();
    auto arrayValue = ArrayValueAttr::get(ctx, {len}, u8Type, elements);

    // Find the enclosing top-level op (FuncOp or GlobalOp) for insertion
    Operation *insertionPoint = op->getParentOfType<FuncOp>();
    if (!insertionPoint)
      insertionPoint = op->getParentOfType<GlobalOp>();
    if (!insertionPoint) {
      return ComptimeError::semanticError(op, context)
             << "@__internal_promote_to_const_global must be used inside a "
                "function or global initializer";
    }

    // Generate unique name
    static std::atomic<unsigned> counter{0};
    const unsigned id = counter.fetch_add(1, std::memory_order_relaxed);
    std::string globalName = "__comptime_string_" + std::to_string(id);

    // Create GlobalOp at module scope (before the enclosing top-level op)
    OpBuilder scopeBuilder(insertionPoint);
    auto newGlobalOp = GlobalOp::create(
        scopeBuilder, op->getLoc(), globalName, /*dynamicType=*/Value(),
        /*is_const=*/true, /*is_export=*/false, /*is_extern=*/false,
        /*is_unbound_param=*/false, /*linkname=*/nullptr, /*align=*/nullptr,
        /*inferred_type=*/nullptr, /*inferred_initializer=*/nullptr);

    // Populate the init region with the array constant
    {
      Block &initBlock = newGlobalOp.getInitRegion().emplaceBlock();
      OpBuilder::InsertionGuard guard(scopeBuilder);
      scopeBuilder.setInsertionPointToStart(&initBlock);
      Value constValue = ConstantOp::create(scopeBuilder, op->getLoc(),
                                            cast<TypedAttr>(arrayValue));
      YieldOp::create(scopeBuilder, op->getLoc(), constValue);
    }

    // Evaluate the new global through the normal GlobalOp evaluation pipeline
    auto *globalCtx = context.getGlobalEvalContext();
    ControlFlowOutcomes outcome =
        globalCtx->evaluateGlobalOnDemand(newGlobalOp);
    if (outcome.hasFatalError())
      return outcome.getFatalError();

    // Create GetSymbolOp + AddressOfOp at the builtin call's position
    auto unresolvedType = UnresolvedType::get(ctx);
    OpBuilder callBuilder(builtinCallOp);
    auto getSymbolOp =
        GetSymbolOp::create(callBuilder, op->getLoc(), unresolvedType,
                            callBuilder.getStringAttr(globalName),
                            /*lexical_module_identifier=*/{},
                            /*inferred_type=*/{});
    auto addressOfOp = AddressOfOp::create(
        callBuilder, op->getLoc(), unresolvedType, getSymbolOp.getResult(),
        /*inferred_type=*/nullptr);

    // Replace the builtin call with the address-of result
    builtinCallOp.getResult().replaceAllUsesWith(addressOfOp.getResult());
    builtinCallOp.erase();

    return ComptimeEvalResult::replaced();
  }

  bool requiresComptimeBySemantics(Operation *op,
                                   const ComptimeEvalContext &context) const {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// AwaitEvaluator — validates async context and optional .offload struct
//===----------------------------------------------------------------------===//

struct AwaitEvaluator : public BuiltinEvaluatorDefaults<AwaitEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    // Standard arg gathering + validation.
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    auto &argValues = std::get<SmallVector<TypedAttr>>(argValuesOrError);

    if (auto error = defaultValidation(op, context, argValues))
      return *error;

    Type resultType = defaultInferResultType(op, context, argValues);

    // Check enclosing function is async.
    auto enclosing = op->getParentOfType<FuncOp>();
    if (!enclosing || !enclosing.getIsAsync()) {
      return ComptimeError::semanticError(op, context)
             << "@await can only be used inside an async function or task";
    }

    // Validate optional config struct (second argument).
    if (argValues.size() > 1) {
      auto structAttr = mlir::dyn_cast<csl::StructValueAttr>(argValues[1]);
      if (!structAttr) {
        return ComptimeError::semanticError(op, context)
               << "@await config argument must be a struct";
      }

      // Only .offload is allowed.
      for (size_t i = 0; i < structAttr.getNumFields(); ++i) {
        auto nameResult = structAttr.getFieldName(i);
        if (failed(nameResult))
          continue;
        StringRef name = *nameResult;
        if (name != "offload") {
          return ComptimeError::semanticError(op, context)
                 << "@await: unknown config field '." << name
                 << "'; only '.offload' is supported";
        }
      }

      // Validate .offload type.
      if (TypedAttr offloadField = structAttr.getField("offload")) {
        if (!mlir::isa<BoolAttr>(offloadField)) {
          return ComptimeError::semanticError(op, context)
                 << "@await: '.offload' must be a bool (true or false)";
        }
      }
    }

    return ComptimeError::comptimeFailure(op, context, resultType)
           << "runtime await";
  }
};

//===----------------------------------------------------------------------===//
// RequiresAsyncContextEvaluator — for @frame_address
//===----------------------------------------------------------------------===//

/// Validates that the builtin is called from within an async function or task.
/// Uses the default arg validation and type inference, then checks the
/// enclosing function context.
struct RequiresAsyncContextEvaluator
    : public BuiltinEvaluatorDefaults<RequiresAsyncContextEvaluator> {

  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto builtinCallOp = cast<BuiltinCallOp>(op);
    auto builtinKind = *builtinCallOp.resolveBuiltin();
    StringRef name = csl::stringifyBuiltinKind(builtinKind);

    // Standard arg gathering + validation.
    auto argValuesOrError = gatherArgValues(op, context);
    if (std::holds_alternative<ComptimeError>(argValuesOrError))
      return std::get<ComptimeError>(argValuesOrError);
    auto &argValues = std::get<SmallVector<TypedAttr>>(argValuesOrError);

    if (auto error = defaultValidation(op, context, argValues))
      return *error;

    Type resultType = defaultInferResultType(op, context, argValues);

    // Check enclosing function is async.
    auto enclosing = op->getParentOfType<FuncOp>();
    if (!enclosing || !enclosing.getIsAsync()) {
      return ComptimeError::semanticError(op, context)
             << "@" << name
             << " can only be used inside an async function or task";
    }

    // Runtime-only — return comptime failure with the inferred type.
    return ComptimeError::comptimeFailure(op, context, resultType)
           << "@" << name << " is a runtime-only builtin";
  }
};

//===----------------------------------------------------------------------===//
// Resource Claim/Release Evaluators
//===----------------------------------------------------------------------===//

/// Single evaluator for all @claim_* builtins. Dispatches on the BuiltinKind
/// read from the op to call the right ResourceBitmaps method and wrap the
/// result in the appropriate typed attribute.
struct ClaimResourceEvaluator
    : public BuiltinEvaluatorDefaults<ClaimResourceEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    if (!context.isComptimeContext())
      return defaultEvaluation(op, context, validation.resultType);

    auto *globalCtx = context.getGlobalEvalContext();
    auto arch = globalCtx->getOptions().cslFlags.targetArch;
    auto &bitmaps = globalCtx->getResourceBitmaps();
    auto kind = *cast<BuiltinCallOp>(op).resolveBuiltin();
    auto *ctx = op->getContext();

    switch (kind) {
    case BuiltinKind::ClaimColor: {
      auto id = bitmaps.claimColor(arch);
      if (!id)
        return ComptimeError::semanticError(op, context)
               << "no free colors available";
      return cast<TypedAttr>(csl::ColorAttr::get(ctx, *id));
    }
    case BuiltinKind::ClaimLocalTask: {
      auto id = bitmaps.claimLocalTask(arch);
      if (!id)
        return ComptimeError::semanticError(op, context)
               << "no free local task entries available";
      return cast<TypedAttr>(csl::LocalTaskIdAttr::get(ctx, *id));
    }
    case BuiltinKind::ClaimDataTask: {
      auto id = bitmaps.claimDataTask(arch);
      if (!id)
        return ComptimeError::semanticError(op, context)
               << "no free data task entries available";
      if (arch == csl::TargetArch::WSE2)
        return cast<TypedAttr>(
            csl::DataTaskIdAttr::get(ctx, csl::ColorAttr::get(ctx, *id)));
      return cast<TypedAttr>(
          csl::DataTaskIdAttr::get(ctx, csl::InputQueueAttr::get(ctx, *id)));
    }
    case BuiltinKind::ClaimInputQueue: {
      auto id = bitmaps.claimInputQueue();
      if (!id)
        return ComptimeError::semanticError(op, context)
               << "no free input queues available";
      return cast<TypedAttr>(csl::InputQueueAttr::get(ctx, *id));
    }
    case BuiltinKind::ClaimOutputQueue: {
      auto id = bitmaps.claimOutputQueue();
      if (!id)
        return ComptimeError::semanticError(op, context)
               << "no free output queues available";
      return cast<TypedAttr>(csl::OutputQueueAttr::get(ctx, *id));
    }
    case BuiltinKind::ClaimControlTask: {
      auto id = bitmaps.claimControlTask();
      if (!id)
        return ComptimeError::semanticError(op, context)
               << "no free control task entries available";
      return cast<TypedAttr>(csl::ControlTaskIdAttr::get(ctx, *id));
    }
    default:
      llvm_unreachable("ClaimResourceEvaluator used for non-claim builtin");
    }
  }
};

/// Single evaluator for all @release_* builtins. Rejects at comptime
/// (globally allocated resources cannot be released), keeps for runtime.
struct ReleaseResourceEvaluator
    : public BuiltinEvaluatorDefaults<ReleaseResourceEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    if (context.isComptimeContext())
      return ComptimeError::semanticError(op, context)
             << "globally allocated resources cannot be released at comptime";

    return defaultEvaluation(op, context, validation.resultType);
  }
};

/// @count_free(T): the number of free resources of the kind T names. At
/// comptime, it counts what the comptime claims have left; at runtime, it is
/// lowered to a call into the allocator.
struct CountFreeEvaluator : public BuiltinEvaluatorDefaults<CountFreeEvaluator> {
  ComptimeEvalResult evaluate(Operation *op,
                              ComptimeEvalContext &context) const {
    auto validationOrError = runValidationAndTypeInference(op, context);
    if (std::holds_alternative<ComptimeError>(validationOrError))
      return std::get<ComptimeError>(validationOrError);
    auto &validation = std::get<ValidationResult>(validationOrError);

    Value typeArg = cast<BuiltinCallOp>(op).getArgs()[0];
    FailureOr<TypedAttr> typeValue = context.getOrLoadComptimeRValue(typeArg);
    if (failed(typeValue))
      return ComptimeError::comptimeFailure(op, context, validation.resultType)
                 .withDependency(typeArg)
             << "the resource type is not comptime-known";
    Type idType = cast<csl::TypeAttr>(*typeValue).getValue();

    auto *globalCtx = context.getGlobalEvalContext();
    csl::TargetArch arch = globalCtx->getOptions().cslFlags.targetArch;
    std::optional<uint16_t> count =
        globalCtx->getResourceBitmaps().countFree(arch, idType);
    if (!count)
      return ComptimeError::semanticError(op, context)
             << "@count_free counts the free resources of a kind given by "
                "its ID type: color, local_task_id, data_task_id, "
                "input_queue, output_queue or control_task_id, not '"
             << csl::prettyPrintType(idType) << "'";

    if (!context.isComptimeContext())
      return defaultEvaluation(op, context, validation.resultType);

    return cast<TypedAttr>(IntegerAttr::get(
        IntegerType::get(op->getContext(), 16, IntegerType::Unsigned),
        *count));
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// BuiltinEvaluatorRegistry Implementation
//===----------------------------------------------------------------------===//

namespace mlir::parsed_csl {

BuiltinEvaluatorRegistry::BuiltinEvaluatorRegistry() {
  // Create default evaluator
  defaultEvaluator = std::make_unique<DefaultBuiltinEvaluator>();

  // Register per-builtin evaluators
  evaluators[static_cast<unsigned>(BuiltinKind::TypeOf)] =
      std::make_unique<TypeOfEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::FutureOf)] =
      std::make_unique<FutureOfEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::ImportModule)] =
      std::make_unique<ImportModuleEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::IsSameType)] =
      std::make_unique<IsSameTypeEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::IsArch)] =
      std::make_unique<IsArchEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::IsComptime)] =
      std::make_unique<IsComptimeEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Strlen)] =
      std::make_unique<StrlenEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Strcat)] =
      std::make_unique<StrcatEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetRectangle)] =
      std::make_unique<GetRectangleEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::SetRectangle)] =
      std::make_unique<SetRectangleEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::ComptimeAssert)] =
      std::make_unique<ComptimeAssertEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::ComptimePrint)] =
      std::make_unique<ComptimePrintEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetColor)] =
      std::make_unique<GetColorEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetInputQueue)] =
      std::make_unique<GetInputQueueEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetOutputQueue)] =
      std::make_unique<GetOutputQueueEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetLocalTaskId)] =
      std::make_unique<GetLocalTaskIdEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetDataTaskId)] =
      std::make_unique<GetDataTaskIdEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetControlTaskId)] =
      std::make_unique<GetControlTaskIdEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetUtId)] =
      std::make_unique<GetUtIdEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetDsr)] =
      std::make_unique<GetDsrEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetXdsr)] =
      std::make_unique<GetXdsrEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetSr)] =
      std::make_unique<GetSrEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetDsd)] =
      std::make_unique<GetDsdEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::AllocateFifo)] =
      std::make_unique<AllocateFifoEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::LoadToDsr)] =
      std::make_unique<LoadToDsrEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::LoadToDsrXdsrSr)] =
      std::make_unique<LoadToDsrXdsrSrEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Range)] =
      std::make_unique<RangeEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::RangeStart)] =
      std::make_unique<RangeStartEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::RangeStop)] =
      std::make_unique<RangeStopEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::RangeStep)] =
      std::make_unique<RangeStepEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Zeros)] =
      std::make_unique<ArrayInitEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Constants)] =
      std::make_unique<ArrayInitEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::GetInt)] =
      std::make_unique<GetIntEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Fp16)] =
      std::make_unique<Fp16Evaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::HasField)] =
      std::make_unique<HasFieldEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Bitcast)] =
      std::make_unique<BitcastEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Field)] =
      std::make_unique<FieldEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::SetDsdStride)] =
      std::make_unique<SetDsdStrideEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::SetDsdLength)] =
      std::make_unique<SetDsdLengthEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::SetDsdBaseAddr)] =
      std::make_unique<SetDsdBaseAddrEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::IncrementDsdOffset)] =
      std::make_unique<IncrementDsdOffsetEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::SetFifoReadLength)] =
      std::make_unique<DsdModifierDefaultEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::SetFifoWriteLength)] =
      std::make_unique<DsdModifierDefaultEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::InternalPromoteToConstGlobal)] =
      std::make_unique<PromoteToConstGlobalEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::Await)] =
      std::make_unique<AwaitEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::FrameAddress)] =
      std::make_unique<RequiresAsyncContextEvaluator>();
  // Resource claim/release evaluators
  for (auto kind :
       {BuiltinKind::ClaimColor, BuiltinKind::ClaimLocalTask,
        BuiltinKind::ClaimDataTask, BuiltinKind::ClaimInputQueue,
        BuiltinKind::ClaimOutputQueue, BuiltinKind::ClaimControlTask})
    evaluators[static_cast<unsigned>(kind)] =
        std::make_unique<ClaimResourceEvaluator>();
  for (auto kind :
       {BuiltinKind::ReleaseColor, BuiltinKind::ReleaseLocalTask,
        BuiltinKind::ReleaseDataTask, BuiltinKind::ReleaseInputQueue,
        BuiltinKind::ReleaseOutputQueue, BuiltinKind::ReleaseControlTask})
    evaluators[static_cast<unsigned>(kind)] =
        std::make_unique<ReleaseResourceEvaluator>();
  evaluators[static_cast<unsigned>(BuiltinKind::CountFree)] =
      std::make_unique<CountFreeEvaluator>();
}

BuiltinEvaluatorRegistry::~BuiltinEvaluatorRegistry() = default;

const BuiltinEvaluatorRegistry::Concept &
BuiltinEvaluatorRegistry::get(BuiltinKind kind) const {
  auto it = evaluators.find(static_cast<unsigned>(kind));
  if (it != evaluators.end())
    return *it->second;
  return *defaultEvaluator;
}

} // namespace mlir::parsed_csl
