//===- GlobalEvalContext.cpp - Global Evaluation Context -----------------===//
//
// Implementation of global evaluation context.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeFunctionEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/FunctionEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"
using namespace mlir;
using namespace mlir::parsed_csl;

#define DEBUG_TYPE "comptime-eval"

//===----------------------------------------------------------------------===//
// GlobalEvalContext Implementation
//===----------------------------------------------------------------------===//

std::shared_ptr<llvm::SourceMgr> GlobalEvalContext::getSourceMgr() const {
  return moduleResolver_ ? moduleResolver_->getSourceMgr() : nullptr;
}

GlobalEvalContext::GlobalEvalContext(
    ComptimeFunctionEvaluator *funcEvaluator, csl::CompilationUnitOp unit,
    std::shared_ptr<csl::CslModuleResolver> moduleResolver,
    ComptimeEvalOptions options)
    : ComptimeEvalContext(nullptr, funcEvaluator), options_(std::move(options)),
      moduleResolver_(std::move(moduleResolver)) {
  // Load bitmap state from the unit if present (e.g., layout-level claims
  // propagated to PE programs). Otherwise initializes with arch defaults.
  resourceBitmaps_ =
      csl::ResourceBitmaps::loadFromUnit(unit, options_.cslFlags.targetArch);
  successful = true;
  ControlFlowOutcomes outcome = evaluateScopeBody(unit.getRegion());
  if (outcome.hasFatalError()) {
    reportComptimeEvalFailure(unit->getContext(), outcome.getFatalError());
    if (!options_.lspMode) {
      successful = false;
      return;
    }
  }

  // Instantiate all exported functions.
  // NOTE: We cannot instantiate all non-generic functions because there exists
  // functions that are valid when comptime-called but not valid at runtime.
  // Tasks are not auto-instantiated - they must be explicitly bound/used.
  // GenericFuncOps are never erased during instantiation, so iteration
  // remains valid.
  walkTopLevelDecls<GenericFuncOp>(unit, [&](GenericFuncOp func) {
    if (!successful)
      return;
    if (func.getIsExport()) {
      auto result = getOrInstantiateFunc(func, {}, {});
      if (std::holds_alternative<ComptimeError>(result)) {
        auto &err = std::get<ComptimeError>(result);
        if (options_.cslFlags.traceErrors)
          err.addCallStackEntry(func.getLoc(), "in exported function '" +
                                                   func.getName() + "'");
        reportComptimeEvalFailure(unit->getContext(), err);
        if (!options_.lspMode)
          successful = false;
      }
    }
  });

  // In LSP mode, eagerly evaluate open files so users get diagnostics and
  // hover info even for non-exported, non-called functions and globals.
  if (options_.shouldEagerlyEvaluate)
    eagerlyEvaluateOpenFiles(unit);
}

void GlobalEvalContext::eagerlyEvaluateOpenFiles(csl::CompilationUnitOp unit) {
  assert(options_.shouldEagerlyEvaluate &&
         "should only be called when predicate is set");

  // Helper: check if an operation's source file matches the predicate.
  auto isInOpenFile = [&](Operation *op) -> bool {
    mlir::Location loc = op->getLoc();
    if (auto flcRange = mlir::dyn_cast<FileLineColRange>(loc))
      return options_.shouldEagerlyEvaluate(flcRange.getFilename().getValue());
    if (auto flcLoc = mlir::dyn_cast<FileLineColLoc>(loc))
      return options_.shouldEagerlyEvaluate(flcLoc.getFilename().getValue());
    return false;
  };

  // Eagerly evaluate globals in open files that haven't been evaluated yet.
  walkTopLevelDecls<GlobalOp>(unit, [&](GlobalOp global) {
    if (!isInOpenFile(global.getOperation()))
      return;
    if (isOpAlive(global.getOperation()))
      return;
    ControlFlowOutcomes outcome = evaluateGlobalOnDemand(global);
    if (outcome.hasFatalError())
      reportComptimeEvalFailure(unit->getContext(), outcome.getFatalError());
  });

  // Eagerly instantiate non-generic functions in open files.
  walkTopLevelDecls<GenericFuncOp>(unit, [&](GenericFuncOp func) {
    if (!isInOpenFile(func.getOperation()))
      return;
    LLVM_DEBUG(llvm::dbgs()
               << "eagerlyEvaluate: '" << func.getSymName()
               << "' isGeneric=" << func.isActuallyGeneric() << "\n");
    if (func.isActuallyGeneric())
      return;
    if (lookupInstantiatedFunc(func, {}, {})) {
      LLVM_DEBUG(llvm::dbgs() << "  -> already instantiated\n");
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "  -> instantiating\n");
    auto result = getOrInstantiateFunc(func, {}, {});
    if (std::holds_alternative<ComptimeError>(result))
      reportComptimeEvalFailure(unit->getContext(),
                                std::get<ComptimeError>(result));
  });
}

ControlFlowOutcomes GlobalEvalContext::evaluateScopeBody(Region &body) {
  Block &bodyBlock = body.front();

  // In LSP mode, we continue past errors to analyze as much as possible,
  // but track the first error to return to callers (e.g. @import_module).
  std::optional<ComptimeError> firstError;

  // Iterate over global decls. For functions, we only annotate the inferred
  // function type for now, as this is needed for comptime functions calls.
  // The function itself is evaluated later.
  for (Operation &op : bodyBlock.getOperations()) {
    if (mlir::isa<parsed_csl::FuncOp>(op) ||
        mlir::isa<parsed_csl::GenericFuncOp>(op)) {
      // Functions are evaluated during their instantiation
      continue;
    }

    // Skip ops already evaluated on demand (via forward reference resolution)
    if (isOpAlive(&op))
      continue;

    // Only eagerly evaluate exported globals and imported modules.
    // Non-exported globals are evaluated on demand via evaluateGlobalOnDemand
    // (triggered by GetSymbolOp). Unreferenced globals remain unevaluated
    // (matches cslc behavior).
    if (auto globalOp = mlir::dyn_cast<parsed_csl::GlobalOp>(op)) {
      bool hasModuleImport = false;
      globalOp.getInitRegion().walk([&](BuiltinCallOp call) {
        if (call.resolveBuiltin() == csl::BuiltinKind::ImportModule) {
          hasModuleImport = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (globalOp.getIsExport() || hasModuleImport) {
        ControlFlowOutcomes outcome = evaluateGlobalOnDemand(globalOp);
        if (outcome.hasFatalError()) {
          if (options_.lspMode) {
            if (!firstError)
              firstError = outcome.getFatalError();
            reportComptimeEvalFailure(globalOp->getContext(),
                                      outcome.getFatalError());
          } else {
            return outcome;
          }
        }
      }
      continue;
    }

    // TODO: Should we instantiate exported functions here too?

    // Evaluate other operations
    ComptimeEvalResult result = this->evaluateOperation(&op);

    // If evaluation produced a fatal error, propagate it
    if (result.isError()) {
      if (options_.lspMode) {
        if (!firstError)
          firstError = result.getError();
        reportComptimeEvalFailure(op.getContext(), result.getError());
        continue;
      }
      return ControlFlowOutcomes::fatal(std::move(result.getError()));
    }
  }

  if (firstError)
    return ControlFlowOutcomes::fatal(std::move(*firstError));

  return ControlFlowOutcomes::fallthrough();
}

ControlFlowOutcomes GlobalEvalContext::evaluateGlobalOnDemand(GlobalOp target) {
  // IMPORTANT: Check cycle BEFORE isOpAlive because evaluateOperation marks
  // ops alive eagerly before their model finishes evaluating.
  if (currentlyEvaluatingGlobals_.contains(target.getOperation())) {
    // ImportModuleEvaluator publishes its stable ModuleRefAttr before
    // evaluating the imported body. A callback analyzed by that body may
    // refer back to the module without depending on completion of the
    // initializer.
    if (auto initializer = target.getInferredInitializer();
        target.getInferredType() && initializer &&
        mlir::isa<ModuleRefAttr>(*initializer))
      return ControlFlowOutcomes::fallthrough();
    return ControlFlowOutcomes::fatal(
        ComptimeError::semanticError(target.getOperation(), *this)
        << "circular dependency in global initialization of '"
        << target.getSymName() << "'");
  }

  // Already evaluated?
  if (isOpAlive(target.getOperation()))
    return ControlFlowOutcomes::fallthrough();

  // RAII guard for cycle tracking
  currentlyEvaluatingGlobals_.insert(target.getOperation());
  llvm::scope_exit guard(
      [&] { currentlyEvaluatingGlobals_.erase(target.getOperation()); });

  // A global can be demanded before the surrounding scope walk reaches the
  // SSA operations that construct its explicit type. Evaluate that type's
  // dependency graph in definition order; evaluating only the final type op
  // leaves references such as `var p: [*]T` unresolved when T is a global.
  if (Value dynType = target.getDynamicType()) {
    std::function<std::optional<ComptimeError>(Operation *)> evaluateDeps =
        [&](Operation *dependency) -> std::optional<ComptimeError> {
      if (!dependency || isOpAlive(dependency))
        return std::nullopt;
      for (Value operand : dependency->getOperands()) {
        if (auto error = evaluateDeps(operand.getDefiningOp()))
          return error;
      }
      ComptimeEvalResult result = evaluateOperation(dependency);
      if (result.isError())
        return std::move(result.getError());
      return std::nullopt;
    };
    if (auto error = evaluateDeps(dynType.getDefiningOp())) {
      return ControlFlowOutcomes::fatal(std::move(*error));
    }
  }

  // Evaluate the GlobalOp itself — its comptime model evaluates the init
  // region, which may trigger recursive evaluateGlobalOnDemand calls via
  // GetSymbolOpComptimeModel for any referenced globals.
  ComptimeEvalResult result = this->evaluateOperation(target.getOperation());
  if (result.isError())
    return ControlFlowOutcomes::fatal(std::move(result.getError()));

  return ControlFlowOutcomes::fallthrough();
}

ControlFlowOutcomes GlobalEvalContext::analyzeFunctionBody(FuncOp func) {
  // Get function body
  Region &body = func.getBody();
  if (body.empty())
    return ControlFlowOutcomes::fallthrough();

  // Create function evaluation context
  auto ctx =
      std::make_unique<FunctionEvalContext>(this, funcEvaluator_, func,
                                            /*isComptimeEvaluated=*/false);

  // Evaluate the function body (optional evaluation for non-comptime functions)
  ControlFlowOutcomes outcome = ctx->evaluateRegion(body, false, false);

  // Store the context (keep it alive)
  functionContexts[func] = std::move(ctx);

  // Insert implicit terminator at end of function body if needed.
  // Skip if fatal error occurred during analysis.
  if (!outcome.hasFatalError()) {
    Block &block = body.front();
    bool alreadyTerminated =
        !block.empty() &&
        mlir::isa<parsed_csl::ReturnOp, parsed_csl::UnreachableOp>(
            block.back());

    if (!alreadyTerminated) {
      OpBuilder builder(func->getContext());
      builder.setInsertionPointToEnd(&block);
      auto *funcCtx = functionContexts[func].get();

      // Use the end of the function's location for implicit terminators,
      // not the full function range — otherwise the implicit return/unreachable
      // competes with the FuncOp itself in LSP position lookups.
      mlir::Location endLoc = func.getLoc();
      if (auto range = mlir::dyn_cast<FileLineColRange>(func.getLoc()))
        endLoc = FileLineColRange::get(range.getFilename(), range.getEndLine(),
                                       range.getEndColumn());

      if (outcome.mayFallthrough) {
        if (mlir::isa<csl::VoidType>(func.getBodyReturnType())) {
          auto retOp = parsed_csl::ReturnOp::create(builder, endLoc);
          funcCtx->markOpAlive(retOp);
        } else {
          func.emitError(
              "non-void function does not return a value on all paths");
          successful = false;
          return outcome;
        }
      } else {
        auto unreachableOp = parsed_csl::UnreachableOp::create(builder, endLoc);
        funcCtx->markOpAlive(unreachableOp);
      }
    }
  }

  return outcome;
}

FunctionEvalContext *GlobalEvalContext::getFunctionContext(FuncOp func) {
  auto it = functionContexts.find(func);
  if (it != functionContexts.end())
    return it->second.get();
  return nullptr;
}

const FunctionEvalContext *
GlobalEvalContext::getFunctionContext(FuncOp func) const {
  auto it = functionContexts.find(func);
  if (it != functionContexts.end())
    return it->second.get();
  return nullptr;
}

bool GlobalEvalContext::isComptimeContextImpl() const {
  return inComptimeBlock;
}

ControlFlowOutcomes GlobalEvalContext::evaluateRegion(Region &region,
                                                      bool isRuntimeConditional,
                                                      bool requiresComptime) {
  if (isRuntimeConditional) {
    return ControlFlowOutcomes::fatal(
        ComptimeError::semanticError(region.getParentOp(), *this)
        << "runtime conditional code is not allowed outside of "
           "functions");
  }

  // Track whether we are in a comptime-required block
  bool previousInComptimeBlock = inComptimeBlock;
  inComptimeBlock |= requiresComptime;

  ControlFlowOutcomes result = ComptimeEvalContext::evaluateRegion(region);

  inComptimeBlock = previousInComptimeBlock;
  return result;
}

ControlFlowOutcomes GlobalEvalContext::evaluateModule(ModuleOp module) {
  // Mark the ModuleOp itself as alive so it's not erased during dead code
  // elimination
  aliveOps_.insert(module.getOperation());

  // Evaluate the module's body into this same flat context
  // Modules are imported in const initializers, which are in comptime blocks.
  // However, they are not evaluated as comptime-only.
  assert(inComptimeBlock &&
         "imported modules must be evaluated in comptime context");
  inComptimeBlock = false; // Modules are not comptime blocks

  // Snapshot error count to detect errors that occurred during module
  // evaluation but were swallowed by LSP-mode error tolerance in
  // evaluateRegion.
  unsigned errorsBefore = errorCount_;

  logger().indent(2);
  ControlFlowOutcomes outcome = evaluateScopeBody(module.getRegion());
  inComptimeBlock = true;
  logger().unindent(2);

  // If evaluateScopeBody reported errors in LSP mode (swallowed by
  // evaluateRegion), surface them as a fatal error so the @import_module
  // caller can warn about the module having errors.
  if (!outcome.hasFatalError() && errorCount_ > errorsBefore) {
    return ControlFlowOutcomes::fatal(
        ComptimeError::semanticError(module.getOperation(), *this)
        << "errors during semantic analysis of module");
  }

  return outcome;
}

FuncOp GlobalEvalContext::lookupInstantiatedFunc(
    GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgValues,
    ArrayRef<Type> anytypeArgTypes) const {
  InstantiationKey key{generic, SmallVector<TypedAttr>(comptimeArgValues),
                       SmallVector<Type>(anytypeArgTypes)};
  auto it = instantiationCache.find(key);
  if (it != instantiationCache.end())
    return it->second;
  return nullptr;
}

std::variant<ComptimeError, FunctionType>
GlobalEvalContext::evaluateFuncType(GenericFuncOp generic,
                                    ArrayRef<TypedAttr> comptimeArgValues,
                                    ArrayRef<Type> anytypeArgTypes) {
  if (generic.getComptimeParams().size() != comptimeArgValues.size()) {
    return ComptimeError::semanticError(generic, *this)
           << "expected " << generic.getComptimeParams().size()
           << " comptime argument(s), but got " << comptimeArgValues.size();
  }
  if (generic.getAnytypeParams().size() != anytypeArgTypes.size()) {
    return ComptimeError::semanticError(generic, *this)
           << "expected " << generic.getAnytypeParams().size()
           << " anytype argument(s), but got " << anytypeArgTypes.size();
  }

  // Evaluate type generator region to get concrete function type
  auto funcTypeResult = funcEvaluator_->evaluateTypeRegion(
      generic, comptimeArgValues, anytypeArgTypes, *this);
  if (std::holds_alternative<ComptimeError>(funcTypeResult)) {
    return std::get<ComptimeError>(funcTypeResult);
  }
  return std::get<FunctionType>(funcTypeResult);
}

std::variant<ComptimeError, FuncOp>
GlobalEvalContext::getOrInstantiateFunc(GenericFuncOp generic,
                                        ArrayRef<TypedAttr> comptimeArgValues,
                                        ArrayRef<Type> anytypeArgTypes) {
  LLVM_DEBUG(llvm::dbgs() << "getOrInstantiateFunc: '" << generic.getSymName()
                          << "' comptime=" << comptimeArgValues.size()
                          << " anytype=" << anytypeArgTypes.size() << "\n");
  if (generic.getComptimeParams().size() != comptimeArgValues.size()) {
    return ComptimeError::semanticError(generic, *this)
           << "expected " << generic.getComptimeParams().size()
           << " comptime argument(s), but got " << comptimeArgValues.size();
  }
  if (generic.getAnytypeParams().size() != anytypeArgTypes.size()) {
    return ComptimeError::semanticError(generic, *this)
           << "expected " << generic.getAnytypeParams().size()
           << " anytype argument(s), but got " << anytypeArgTypes.size();
  }
  // Check cache first
  if (FuncOp cachedFunc =
          lookupInstantiatedFunc(generic, comptimeArgValues, anytypeArgTypes)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  -> cached as '" << cachedFunc.getSymName() << "'\n");
    return cachedFunc;
  }

  // Need to instantiate: evaluate type generator region to get concrete types
  auto funcTypeResult = funcEvaluator_->evaluateTypeRegion(
      generic, comptimeArgValues, anytypeArgTypes, *this);
  if (std::holds_alternative<ComptimeError>(funcTypeResult)) {
    LLVM_DEBUG(llvm::dbgs() << "  -> type region evaluation failed\n");
    return std::get<ComptimeError>(funcTypeResult);
  }
  FunctionType genericFuncType = std::get<FunctionType>(funcTypeResult);

  // Create function type of instantiated function without comptime params.
  // Use genericFuncType.getNumInputs() instead of numParams because extern
  // functions have empty body (numParams=0) but have params in their type.
  SmallVector<Type> instParamTypes;
  for (size_t i = 0; i < genericFuncType.getNumInputs(); ++i) {
    if (!llvm::is_contained(generic.getComptimeParams(),
                            static_cast<int32_t>(i))) {
      instParamTypes.push_back(genericFuncType.getInput(i));
    }
  }
  auto instFuncType = FunctionType::get(generic.getContext(), instParamTypes,
                                        genericFuncType.getResults());

  // Validate task parameter types (must be integer or float types)
  if (generic.getIsTask()) {
    for (size_t i = 0; i < instParamTypes.size(); ++i) {
      Type paramType = instParamTypes[i];
      bool isValidTaskParam =
          mlir::isa<IntegerType>(paramType) || mlir::isa<FloatType>(paramType);
      if (!isValidTaskParam) {
        return ComptimeError::semanticError(generic, *this)
               << "task parameter " << i << " has invalid type '"
               << csl::prettyPrintType(paramType)
               << "'; task parameters must be integer or float types";
      }
    }
  }

  LLVM_DEBUG({
    logger().indent(2);
    logger().startLine() << "++++++++++++++++++++++++++++++++++++++++++++\n";
    logger().startLine() << "Instantiating function '" << generic.getName()
                         << "' of generic type '" << genericFuncType
                         << "' to instantiated type '" << instFuncType << "'\n";
  });

  // Generate mangled name
  std::string mangledName = generic.getName().str();
  mangledName += "_inst" + std::to_string(nextInstantiationId++);

  // For async fns (not tasks), wrap the result type in FutureType and store
  // the original result type separately as async_result_type. Tasks are never
  // called, so they don't produce a future — they keep their original type.
  mlir::TypeAttr asyncResultTypeAttr;
  if (generic.getIsAsync() && !generic.getIsTask()) {
    Type origResultTy = instFuncType.getNumResults() > 0
                            ? instFuncType.getResult(0)
                            : csl::VoidType::get(generic.getContext());
    asyncResultTypeAttr = mlir::TypeAttr::get(origResultTy);
    std::optional<uint32_t> moduleId;
    if (auto moduleOp = generic->getParentOfType<parsed_csl::ModuleOp>())
      moduleId = moduleOp.getModuleIdentifier();
    auto futureTy = csl::FutureType::get(
        generic.getContext(), moduleId,
        FlatSymbolRefAttr::get(generic.getContext(), mangledName),
        origResultTy);
    instFuncType = FunctionType::get(generic.getContext(),
                                     instFuncType.getInputs(), {futureTy});
  }

  // Evaluate linkname region if present
  StringAttr linknameAttr;
  Region &linknameRegion = generic.getLinknameRegion();
  if (!linknameRegion.empty()) {
    ControlFlowOutcomes linknameOutcome =
        this->evaluateRegion(linknameRegion, /*isRuntimeConditional=*/false,
                             /*requiresComptime=*/true);
    if (linknameOutcome.hasFatalError())
      return linknameOutcome.getFatalError();
    if (linknameOutcome.mayYield.empty())
      return ComptimeError::semanticError(generic, *this)
             << "linkname expression did not yield a value";
    auto yieldedValue = linknameOutcome.mayYield[0].getValue();
    if (!yieldedValue)
      return ComptimeError::semanticError(generic, *this)
             << "linkname expression must be comptime-known";
    auto strAttr = mlir::dyn_cast<csl::ComptimeStringAttr>(*yieldedValue);
    if (!strAttr)
      return ComptimeError::semanticError(generic, *this)
             << "linkname must evaluate to a comptime_string";
    linknameAttr = StringAttr::get(generic->getContext(), strAttr.getValue());
  }

  // Create instantiated FuncOp with attributes copied from GenericFuncOp
  OpBuilder builder(generic->getContext());
  builder.setInsertionPointAfter(generic);

  auto funcOp = FuncOp::create(
      builder, generic.getLoc(), mangledName, instFuncType, generic.getName(),
      generic.getIsTask(), generic.getIsInline(), generic.getIsNoinline(),
      generic.getIsExport(), generic.getIsExtern(), generic.getIsAsync(),
      generic.getStackSizeAttr(), linknameAttr, generic.getParamNamesAttr(),
      generic.getDocStringAttr(), asyncResultTypeAttr,
      generic.getMaxRecursionDepthAttr());

  // Clone body region, remapping comptime params (only if function has a body)
  Region &genericBody = generic.getBody();
  Region &funcBody = funcOp.getBody();

  // Only clone body if generic has a body (declarations have empty region)
  if (!genericBody.empty()) {
    // Build block argument mapping: skip comptime params
    IRMapping mapping;
    Block &genericBodyBlock = genericBody.front();
    Block *funcBodyBlock = builder.createBlock(&funcBody);

    builder.setInsertionPointToEnd(funcBodyBlock);
    size_t comptimeIdx = 0;
    for (size_t i = 0; i < genericBodyBlock.getNumArguments(); ++i) {
      Value genericArg = genericBodyBlock.getArgument(i);
      if (llvm::is_contained(generic.getComptimeParams(),
                             static_cast<int32_t>(i))) {
        // Comptime param - inline casted constant
        TypedAttr comptimeValue = comptimeArgValues[comptimeIdx++];
        InFlightDiagnostic diag = generic.emitRemark();
        FailureOr<TypedAttr> castedValue = csl::evaluateImplicitCast(
            comptimeValue, genericFuncType.getInput(i), &diag);
        if (failed(castedValue)) {
          return ComptimeError::semanticError(generic, *this)
                 << "failed to cast comptime argument to parameter type"
                 << std::move(diag);
        }
        diag.abandon();
        Operation *inlinedConst = generic->getDialect()->materializeConstant(
            builder, *castedValue, comptimeValue.getType(), generic.getLoc());
        mapping.map(genericArg, inlinedConst->getResult(0));
      } else {
        // Runtime param - add block argument, preserving the original location
        Value funcArg = funcBodyBlock->addArgument(
            builder.getType<UnresolvedType>(), genericArg.getLoc());
        mapping.map(genericArg, funcArg);
      }
    }

    // Clone operations from generic body to func body
    for (Operation &op : genericBodyBlock.getOperations()) {
      builder.clone(op, mapping);
    }
  }
  // For function declarations, funcBody remains empty

  // Cache the instantiated function
  InstantiationKey key{generic, SmallVector<TypedAttr>(comptimeArgValues),
                       SmallVector<Type>(anytypeArgTypes)};
  instantiationCache[key] = funcOp;

  // Analyze the new function body (only if it has one)
  if (!funcOp.getBody().empty()) {
    ControlFlowOutcomes bodyOutcome = analyzeFunctionBody(funcOp);
    if (bodyOutcome.hasFatalError()) {
      LLVM_DEBUG({
        llvm::dbgs() << "  -> body analysis failed for '" << funcOp.getSymName()
                     << "'\n";
        logger().unindent(2);
      });
      return bodyOutcome.getFatalError();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  -> instantiated '" << generic.getSymName() << "' as '"
                 << funcOp.getSymName() << "': " << funcOp.getFunctionType()
                 << "\n";
    logger().unindent(2);
  });

  return funcOp;
}

void GlobalEvalContext::recordBuiltinCall(csl::BuiltinKind builtin,
                                          ArrayRef<TypedAttr> args,
                                          Type resultType, Operation *op) {
  LLVM_DEBUG({
    logger().startLine() << "Recording builtin call to @"
                         << csl::stringifyBuiltinKind(builtin) << "\n";
  });
  bool fromLayout = isEvaluatingLayoutBlock();
  recordedBuiltinCalls.push_back(
      RecordedBuiltinCall{builtin, SmallVector<TypedAttr>(args), resultType,
                          op->getLoc(), fromLayout});
}

FailureOr<Value>
RecordedBuiltinCall::rematerializeCall(OpBuilder &builder) const {
  // Create BuiltinCallOp
  ParsedCSLDialect *dialect =
      builder.getContext()->getLoadedDialect<ParsedCSLDialect>();

  SmallVector<Value> rematerializedArgs;
  for (TypedAttr constAttr : argValues) {
    Operation *constOp = dialect->materializeConstant(
        builder, constAttr, constAttr.getType(), location);
    rematerializedArgs.push_back(constOp->getResult(0));
  }

  return parsed_csl::BuiltinCallOp::create(
             builder, location, csl::stringifyBuiltinKind(builtin),
             rematerializedArgs, mlir::TypeAttr::get(resultType))
      .getResult();
}
