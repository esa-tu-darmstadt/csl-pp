//===- ComptimeFunctionEvaluator.cpp - Comptime Function Evaluation ------===//
//
// Implementation of comptime function call evaluator.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeFunctionEvaluator.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/FunctionEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include <atomic>

using namespace mlir;
using namespace mlir::parsed_csl;

std::atomic<int64_t> ComptimeFunctionEvaluator::tempCounter_{0};

std::variant<ComptimeError, TypedAttr>
ComptimeFunctionEvaluator::evaluateCall(FuncOp func,
                                        ArrayRef<TypedAttr> argValues,
                                        ComptimeEvalContext *parentContext) {
  // Check maximum recursion depth
  if (callStack_.size() >= maxRecursionDepth) {
    return ComptimeError::semanticError(func, *parentContext)
           << "maximum comptime evaluation recursion depth ("
           << maxRecursionDepth << ") exceeded";
  }

  // Get function body
  Region &body = func.getBody();
  if (body.empty()) {
    return ComptimeError::semanticError(func, *parentContext)
           << "cannot evaluate function with no body at comptime";
  }

  Block &entryBlock = body.front();
  auto blockArgs = entryBlock.getArguments();
  FunctionType funcType = func.getFunctionType();

  // Check parameter count matches
  if (blockArgs.size() != argValues.size()) {
    return ComptimeError::semanticError(func, *parentContext)
           << "comptime call argument count mismatch: expected "
           << blockArgs.size() << " arguments, got " << argValues.size();
  }

  // Create function evaluation context (comptime-required)
  // Pass parentContext as the parent and this as the function evaluator
  FunctionEvalContext funcContext(parentContext, this, func,
                                  /*isComptimeEvaluated=*/true);

  // Cast arguments and map to parameters
  for (auto [param, argValue, paramType] :
       llvm::zip(blockArgs, argValues, funcType.getInputs())) {
    // Perform implicit cast if needed
    TypedAttr castedArgValue = argValue;
    if (castedArgValue.getType() != paramType) {
      InFlightDiagnostic diag = func->emitError();
      FailureOr<TypedAttr> maybeCastedArgValue =
          csl::evaluateImplicitCast(argValue, paramType, &diag);
      if (failed(maybeCastedArgValue)) {
        return ComptimeError::semanticError(func, *parentContext)
               << "failed to cast argument to parameter type";
      }
      diag.abandon();
      castedArgValue = *maybeCastedArgValue;
    }

    // Bind casted arg to parameter
    funcContext.bindParameter(param, castedArgValue);
  }

  // Evaluate the function body region
  ControlFlowOutcomes outcome =
      funcContext.evaluateRegion(body, /*isRuntimeConditional=*/false,
                                 /*requiresComptime=*/true);

  // If evaluation failed, report error
  if (outcome.hasFatalError()) {
    auto err = outcome.getFatalError().fatalize();
    auto *gctx = parentContext->getGlobalEvalContext();
    if (!gctx || !gctx->getOptions().cslFlags.traceErrors)
      err << func->emitRemark("comptime call to this function failed");
    return err;
  }

  // Extract the return value from the outcomes.
  // In comptime context, exactly one return path should be active.
  TypedAttr returnValue = nullptr;
  if (!outcome.mayReturn.empty()) {
    assert(outcome.mayReturn.size() == 1 &&
           "comptime function call should have exactly one return path");
    auto val = outcome.mayReturn[0].getValue();
    if (val)
      returnValue = *val;
  }
  if (!returnValue) {
    returnValue = csl::VoidAttr::get(func->getContext());
  }

  // Cast return value to declared return type if needed.
  if (funcType.getNumResults() == 1 &&
      returnValue.getType() != funcType.getResult(0)) {
    InFlightDiagnostic diag = func->emitError();
    FailureOr<TypedAttr> castedReturnValue =
        csl::evaluateImplicitCast(returnValue, funcType.getResult(0), &diag);
    if (failed(castedReturnValue)) {
      return ComptimeError::semanticError(func, *parentContext)
             << "failed to cast return value to declared return type"
             << std::move(diag);
    }
    diag.abandon();
    returnValue = *castedReturnValue;
  }

  return returnValue;
}

bool ComptimeFunctionEvaluator::isRecursive(
    GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgs,
    ArrayRef<TypedAttr> runtimeArgs) const {
  CallFrame frame{generic, SmallVector<TypedAttr>(comptimeArgs),
                  SmallVector<TypedAttr>(runtimeArgs)};
  return llvm::is_contained(callStack_, frame);
}

std::variant<ComptimeError, FuncOp>
ComptimeFunctionEvaluator::createTempInstantiation(
    GenericFuncOp generic, FunctionType funcType,
    ArrayRef<TypedAttr> comptimeArgValues, ArrayRef<Type> anytypeArgTypes,
    ComptimeEvalContext &context) {
  assert(!generic.getIsAsync() &&
         "async functions cannot be evaluated at comptime");

  // Generate unique temp name
  size_t uniqueId = tempCounter_.fetch_add(1);
  std::string tempName =
      generic.getName().str() + "_comptimeeval" + std::to_string(uniqueId);

  // Create the instantiated function type (without comptime params)
  SmallVector<Type> instParamTypes;
  for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
    if (!llvm::is_contained(generic.getComptimeParams(),
                            static_cast<int32_t>(i)))
      instParamTypes.push_back(funcType.getInput(i));
  }
  auto instFuncType = FunctionType::get(generic.getContext(), instParamTypes,
                                        funcType.getResults());

  // Evaluate linkname region if present
  StringAttr linknameAttr;
  Region &linknameRegion = generic.getLinknameRegion();
  if (!linknameRegion.empty()) {
    ControlFlowOutcomes linknameOutcome =
        context.evaluateRegion(linknameRegion, /*isRuntimeConditional=*/false,
                               /*requiresComptime=*/true);
    if (linknameOutcome.hasFatalError())
      return linknameOutcome.getFatalError();
    if (linknameOutcome.mayYield.empty())
      return ComptimeError::semanticError(generic, context)
             << "linkname expression did not yield a value";
    auto yieldedValue = linknameOutcome.mayYield[0].getValue();
    if (!yieldedValue)
      return ComptimeError::semanticError(generic, context)
             << "linkname expression must be comptime-known";
    auto strAttr = mlir::dyn_cast<csl::ComptimeStringAttr>(*yieldedValue);
    if (!strAttr)
      return ComptimeError::semanticError(generic, context)
             << "linkname must evaluate to a comptime_string";
    linknameAttr = StringAttr::get(generic->getContext(), strAttr.getValue());
  }

  // Create FuncOp
  OpBuilder builder(generic->getContext());
  builder.setInsertionPointAfter(generic);
  auto funcOp = FuncOp::create(
      builder, generic.getLoc(), tempName, instFuncType, generic.getName(),
      generic.getIsTask(), generic.getIsInline(), generic.getIsNoinline(),
      /*isExport=*/false, generic.getIsExtern(), generic.getIsAsync(),
      generic.getStackSizeAttr(), linknameAttr, generic.getParamNamesAttr(),
      generic.getDocStringAttr(),
      /*async_result_type=*/{}, generic.getMaxRecursionDepthAttr());

  // Clone body region, remapping comptime params
  Region &genericBody = generic.getBody();
  Region &funcBody = funcOp.getBody();

  if (!genericBody.empty()) {
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
            comptimeValue, funcType.getInput(i), &diag);
        if (failed(castedValue)) {
          funcOp->erase();
          return ComptimeError::semanticError(generic, context)
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

  return funcOp;
}

std::variant<ComptimeError, TypedAttr>
ComptimeFunctionEvaluator::evaluateComptimeCallViaTemp(
    GenericFuncOp generic, FunctionType funcType,
    ArrayRef<TypedAttr> comptimeArgValues, ArrayRef<Type> anytypeArgTypes,
    ArrayRef<TypedAttr> runtimeArgValues, ComptimeEvalContext *parentContext) {
  // Check recursion using GenericFuncOp
  if (isRecursive(generic, comptimeArgValues, runtimeArgValues)) {
    return ComptimeError::semanticError(generic, *parentContext)
           << "recursive call detected with same arguments during comptime "
              "evaluation";
  }

  // Check max depth
  if (callStack_.size() >= maxRecursionDepth) {
    return ComptimeError::semanticError(generic, *parentContext)
           << "maximum comptime evaluation recursion depth ("
           << maxRecursionDepth << ") exceeded";
  }

  // Push call frame
  callStack_.push_back(CallFrame{generic,
                                 SmallVector<TypedAttr>(comptimeArgValues),
                                 SmallVector<TypedAttr>(runtimeArgValues)});

  // Create temp instantiation
  auto tempResult = createTempInstantiation(
      generic, funcType, comptimeArgValues, anytypeArgTypes, *parentContext);
  if (std::holds_alternative<ComptimeError>(tempResult)) {
    callStack_.pop_back();
    return std::get<ComptimeError>(tempResult);
  }
  FuncOp tempFunc = std::get<FuncOp>(tempResult);

  // Evaluate the temp function
  auto result = evaluateCall(tempFunc, runtimeArgValues, parentContext);

  // Clean up: erase the temp function
  tempFunc->erase();

  // Pop call frame
  callStack_.pop_back();

  return result;
}

std::variant<ComptimeError, FunctionType>
ComptimeFunctionEvaluator::evaluateTypeRegion(
    GenericFuncOp generic, ArrayRef<TypedAttr> comptimeArgValues,
    ArrayRef<Type> anytypeArgTypes, ComptimeEvalContext &context) {
  size_t numParams = generic.getNumParams();

  Region &typeRegion = generic.getTypeRegion();
  if (typeRegion.empty()) {
    return ComptimeError::semanticError(generic, context)
           << "generic function has no type region";
  }

  // Bind comptime and anytype block arguments of type generator region
  size_t comptimeIdx = 0;
  size_t anytypeIdx = 0;
  size_t typeRegionParamIdx = 0;
  for (int32_t paramIdx = 0; paramIdx < (int32_t)numParams; ++paramIdx) {
    bool isComptimeParam =
        llvm::is_contained(generic.getComptimeParams(), paramIdx);
    bool isAnytypeParam =
        llvm::is_contained(generic.getAnytypeParams(), paramIdx);
    if (!isComptimeParam && !isAnytypeParam)
      continue;

    BlockArgument arg = typeRegion.getArgument(typeRegionParamIdx++);
    if (isComptimeParam) {
      // Comptime param: bind provided value
      context.setComptimeValue(arg, comptimeArgValues[comptimeIdx++]);
    } else if (isAnytypeParam) {
      // Anytype param: bind provided type
      context.setInferredTypeDirectly(arg, anytypeArgTypes[anytypeIdx++]);
    }
  }

  ControlFlowOutcomes typeOutcome =
      context.evaluateRegion(typeRegion, /*isRuntimeConditional=*/false,
                             /*requiresComptime=*/true);

  if (typeOutcome.hasFatalError()) {
    return typeOutcome.getFatalError();
  }

  // In comptime context, exactly one yield should be produced
  if (typeOutcome.mayYield.empty()) {
    return ComptimeError::semanticError(generic, context)
           << "type region did not produce a value";
  }

  assert(typeOutcome.mayYield.size() == 1 &&
         "type region should yield exactly once");
  auto yieldedValue = typeOutcome.mayYield[0].getValue();
  if (!yieldedValue) {
    return ComptimeError::semanticError(generic, context)
           << "type region did not produce a comptime-known value";
  }

  auto typeAttr = mlir::dyn_cast<csl::TypeAttr>(*yieldedValue);
  if (!typeAttr) {
    return ComptimeError::semanticError(generic, context)
           << "type region did not produce a type value";
  }

  auto funcType = mlir::dyn_cast<FunctionType>(typeAttr.getValue());
  if (!funcType) {
    return ComptimeError::semanticError(generic, context)
           << "type region did not produce a function type";
  }

  // For non-extern functions, validate that the computed function type
  // matches the body's parameter count. Extern functions don't have a body
  // so numParams is 0 while funcType has the actual parameter count.
  if (!generic.getIsExtern() && funcType.getNumInputs() != numParams) {
    return ComptimeError::semanticError(generic, context)
           << "function cannot be instantiated with " << numParams
           << " parameters (requires " << funcType.getNumInputs() << ")";
  }

  // For export or extern functions, validate export compatibility
  if (generic.getIsExport() || generic.getIsExtern()) {
    if (!csl::isExportCompatibleFunctionType(funcType)) {
      return ComptimeError::semanticError(generic, context)
             << "function type '" << csl::prettyPrintType(funcType)
             << "' is not export compatible";
    }
  }

  return funcType;
}
