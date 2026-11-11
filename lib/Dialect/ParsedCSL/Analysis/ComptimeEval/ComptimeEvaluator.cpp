//===- ComptimeEvaluator.cpp - Compile-Time Evaluation -------------------===//
//
// Implementation of top-level comptime analysis.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/FunctionEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::parsed_csl;

#define DEBUG_TYPE "comptime-eval"

ComptimeEvaluator::ComptimeEvaluator(
    csl::CompilationUnitOp unit,
    std::shared_ptr<csl::CslModuleResolver> moduleResolver,
    ComptimeEvalOptions options)
    : funcEvaluator(),
      globalContext(&funcEvaluator, unit, std::move(moduleResolver),
                    std::move(options)),
      successful(succeeded(globalContext)) {}

std::variant<ComptimeError, FunctionType>
ComptimeEvaluator::evaluateFunctionType(GenericFuncOp function,
                                        ArrayRef<TypedAttr> comptimeArguments,
                                        ArrayRef<Type> anytypeArgumentTypes) {
  return globalContext.evaluateFuncType(function, comptimeArguments,
                                        anytypeArgumentTypes);
}

std::variant<ComptimeError, FuncOp>
ComptimeEvaluator::instantiateFunction(GenericFuncOp function,
                                       ArrayRef<TypedAttr> comptimeArguments,
                                       ArrayRef<Type> anytypeArgumentTypes) {
  return globalContext.getOrInstantiateFunc(function, comptimeArguments,
                                            anytypeArgumentTypes);
}

FailureOr<bool> ComptimeEvaluator::isComptimeOnlyFunction(FuncOp func) const {
  return globalContext.isComptimeOnlyFunction(func);
}

std::variant<ComptimeError, TypedAttr>
ComptimeEvaluator::evaluateFunction(GenericFuncOp func,
                                    ArrayRef<TypedAttr> arguments) {
  if (arguments.size() != static_cast<size_t>(func.getNumParams())) {
    return ComptimeError::semanticError(func, globalContext)
           << "expected " << func.getNumParams() << " argument(s), but got "
           << arguments.size();
  }

  SmallVector<TypedAttr> comptimeArguments;
  SmallVector<Type> anytypeArgumentTypes;
  SmallVector<TypedAttr> runtimeArguments;
  for (auto [index, argument] : llvm::enumerate(arguments)) {
    int32_t parameterIndex = static_cast<int32_t>(index);
    if (llvm::is_contained(func.getComptimeParams(), parameterIndex))
      comptimeArguments.push_back(argument);
    else
      runtimeArguments.push_back(argument);

    if (llvm::is_contained(func.getAnytypeParams(), parameterIndex))
      anytypeArgumentTypes.push_back(argument.getType());
  }

  auto functionType = globalContext.evaluateFuncType(func, comptimeArguments,
                                                     anytypeArgumentTypes);
  if (std::holds_alternative<ComptimeError>(functionType))
    return std::get<ComptimeError>(functionType);

  return globalContext.evaluateComptimeCallViaTemp(
      func, std::get<FunctionType>(functionType), comptimeArguments,
      anytypeArgumentTypes, runtimeArguments);
}

//===----------------------------------------------------------------------===//
// Context Routing
//===----------------------------------------------------------------------===//

const ComptimeEvalContext &
ComptimeEvaluator::getContextForValue(Value v) const {
  // Find which function this value belongs to
  Operation *definingOp = v.getDefiningOp();
  FuncOp parentFunc = nullptr;

  if (!definingOp) {
    // Could be a block argument - check parent op
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (mlir::isa<FuncOp>(parentOp))
        parentFunc = mlir::cast<FuncOp>(parentOp);
      else
        parentFunc = parentOp->getParentOfType<FuncOp>();
    }
  } else {
    parentFunc = definingOp->getParentOfType<FuncOp>();
  }

  // If in a function, return its context
  if (parentFunc) {
    if (auto *funcCtx = globalContext.getFunctionContext(parentFunc))
      return *funcCtx;
  }

  // Default to global context
  return globalContext;
}

const ComptimeEvalContext *
ComptimeEvaluator::getContextForOp(Operation *op) const {
  // Find which function this operation belongs to
  FuncOp parentFunc = op->getParentOfType<FuncOp>();

  // If in a function, return its context (or nullptr if none exists)
  if (parentFunc)
    return globalContext.getFunctionContext(parentFunc);

  // At global scope, return global context
  return &globalContext;
}

bool ComptimeEvaluator::isOpAlive(Operation *op) const {
  // FuncOps with contexts are alive (they were analyzed)
  if (auto funcOp = mlir::dyn_cast<FuncOp>(op))
    return globalContext.getFunctionContext(funcOp) != nullptr;

  const ComptimeEvalContext *ctx = getContextForOp(op);
  // If no context (e.g., comptime-only function), consider op alive
  // (it will be erased along with the function, not as dead code)
  if (!ctx)
    return true;
  return ctx->isOpAlive(op);
}

void ComptimeEvaluator::markOpAlive(Operation *op) {
  ComptimeEvalContext *ctx =
      const_cast<ComptimeEvalContext *>(getContextForOp(op));
  if (ctx)
    ctx->markOpAlive(op);
}

std::optional<ComptimeEvalContext::InstantiationResult>
ComptimeEvaluator::getInstantiatedCallee(CallOp callOp) const {
  const ComptimeEvalContext *ctx = getContextForOp(callOp);
  if (!ctx)
    return std::nullopt;
  return ctx->getInstantiatedCallee(callOp);
}

//===----------------------------------------------------------------------===//
// Query Interface Implementation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ComptimeEvaluator::getComptimeRValue(Value v) const {
  return getContextForValue(v).getComptimeRValue(v);
}

FailureOr<LValueDescriptor>
ComptimeEvaluator::getComptimeLValue(Value v) const {
  return getContextForValue(v).getComptimeLValue(v);
}

FailureOr<TypedAttr> ComptimeEvaluator::getOrLoadComptimeRValue(Value v) const {
  return getContextForValue(v).getOrLoadComptimeRValue(v);
}

FailureOr<Type> ComptimeEvaluator::getInferredType(Value v) const {
  return getContextForValue(v).getInferredType(v);
}

bool ComptimeEvaluator::isComptimeKnown(Value v,
                                        bool voidAlwaysComptimeKnown) const {
  return getContextForValue(v).isComptimeKnown(v, voidAlwaysComptimeKnown);
}

const FunctionEvalContext *
ComptimeEvaluator::getFunctionContext(FuncOp func) const {
  return globalContext.getFunctionContext(func);
}

//===----------------------------------------------------------------------===//
// Mutation Interface Implementation
//===----------------------------------------------------------------------===//

void ComptimeEvaluator::setInferredType(Value v, Type type) {
  const_cast<ComptimeEvalContext &>(getContextForValue(v))
      .setInferredTypeDirectly(v, type);
}

//===----------------------------------------------------------------------===//
// Materialization Interface Implementation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
ComptimeEvaluator::getMaterializedValue(OpOperand &operand) const {
  Operation *op = operand.getOwner();
  const ComptimeEvalContext *context = getContextForOp(op);
  if (!context)
    return failure();

  Value value = operand.get();
  FailureOr<Type> inferredType = getInferredType(value);
  if (failed(inferredType))
    return failure();

  auto lValueType = mlir::dyn_cast_if_present<csl::LValueType>(*inferredType);
  if (!lValueType)
    return context->getOrLoadComptimeRValue(value);

  FailureOr<LValueDescriptor> lvalueDescr = context->getComptimeLValue(value);
  if (failed(lvalueDescr))
    return failure();

  // A const, loadable lvalue is replaced by its rvalue unless the op expects
  // an lvalue (e.g., AddressOfOp).
  if (FailureOr<TypedAttr> rvalue = context->loadLValue(*lvalueDescr);
      succeeded(rvalue) && lValueType.getIsConst() &&
      !operandExpectsLValue(op, operand.getOperandNumber(),
                            [&](Value v) { return getInferredType(v); }))
    return *rvalue;

  // Otherwise the lvalue is kept if it can be represented as an attribute
  // (e.g., a global variable reference).
  if (std::optional<TypedAttr> attr = lvalueDescr->toAttr())
    return *attr;
  return failure();
}

bool ComptimeEvaluator::isErasedAsComptimeStatement(Operation *op) const {
  if (auto callOp = mlir::dyn_cast<CallOp>(op))
    return isComptimeKnown(callOp, /*voidAlwaysComptimeKnown=*/false);
  if (auto assignOp = mlir::dyn_cast<AssignOp>(op)) {
    // Storing the single value of a void-like type changes nothing, e.g. the
    // void field of a runtime tuple. The stored value, e.g. a call, remains.
    FailureOr<Type> lvalueType = getInferredType(assignOp.getLvalue());
    if (succeeded(lvalueType)) {
      if (auto lvalue = mlir::dyn_cast<csl::LValueType>(*lvalueType);
          lvalue && csl::isVoidLikeType(lvalue.getValueType()))
        return true;
    }
    return isComptimeKnown(assignOp.getLvalue(),
                           /*voidAlwaysComptimeKnown=*/false);
  }
  return false;
}

namespace {

/// Decides which evaluated ops remain in the runtime program after
/// materialization. An op does not remain if it is dead, nested in code that
/// only exists at comptime, erased by materialization, or folded away: an op
/// with a comptime-known result is dead once materialization replaced its
/// uses by constants, unless a remaining op still uses the result itself
/// (e.g., the address of a comptime-known const variable is taken).
class RuntimeReachability {
public:
  explicit RuntimeReachability(const ComptimeEvaluator &evaluator)
      : evaluator(evaluator) {}

  bool reachesRuntime(Operation *op) {
    auto [it, inserted] = cache.try_emplace(op, true);
    if (!inserted)
      return it->second;
    bool result = compute(op);
    cache[op] = result;
    return result;
  }

private:
  bool compute(Operation *op) {
    if (mlir::isa<csl::CompilationUnitOp>(op))
      return true;
    // Bodies of generic functions are templates; comptime and layout blocks
    // are replaced by the recorded builtin calls; comptime expressions fold to
    // their value; affine expressions of tensor accesses are not evaluated.
    if (mlir::isa<GenericFuncOp, ComptimeBlockOp, LayoutBlockOp, ComptimeExprOp,
                  TensorAccessOp>(op))
      return false;

    Operation *parent = op->getParentOp();
    if (!parent || !reachesRuntime(parent))
      return false;
    // The init and linkname regions of globals are cleared after evaluation.
    if (mlir::isa<GlobalOp>(parent))
      return false;
    if (!evaluator.isOpAlive(op))
      return false;

    // Globals holding functions are comptime-only and erased.
    if (auto globalOp = mlir::dyn_cast<GlobalOp>(op)) {
      std::optional<Type> inferredType = globalOp.getInferredType();
      return inferredType &&
             !mlir::isa<csl::GenericFunctionType, FunctionType>(
                 mlir::cast<csl::LValueType>(*inferredType).getValueType());
    }

    if (evaluator.isErasedAsComptimeStatement(op))
      return false;

    if (op->getNumRegions() == 0 && op->getNumResults() == 1 &&
        evaluator.isComptimeKnown(op->getResult(0))) {
      return llvm::any_of(op->getResult(0).getUses(), [&](OpOperand &use) {
        return reachesRuntime(use.getOwner()) &&
               failed(evaluator.getMaterializedValue(use));
      });
    }
    return true;
  }

  const ComptimeEvaluator &evaluator;
  DenseMap<Operation *, bool> cache;
};

/// Check if a function signature contains comptime-only types. Such functions
/// can only be called at comptime and are never part of the runtime program.
/// A void result is valid in a runtime signature.
bool hasComptimeOnlySignature(FunctionType funcType) {
  auto isComptimeOnly = [](Type type) {
    return !mlir::isa<csl::VoidType>(type) && csl::isComptimeOnlyType(type);
  };
  return llvm::any_of(funcType.getInputs(), isComptimeOnly) ||
         llvm::any_of(funcType.getResults(), isComptimeOnly);
}

} // namespace

LogicalResult
ComptimeEvaluator::instantiateRuntimeReferences(csl::CompilationUnitOp unit) {
  const ComptimeEvalOptions &options = globalContext.getOptions();
  // GenericFuncOps whose instantiation was already attempted. A failed
  // instantiation is not retried.
  DenseSet<Operation *> attempted;

  while (true) {
    SmallVector<std::pair<GenericFuncOp, Location>> pending;
    auto collectReferences = [&](Attribute value, Location loc) {
      value.walk([&](csl::FuncRefAttr funcRef) {
        auto funcType = mlir::dyn_cast<FunctionType>(funcRef.getType());
        if (!funcType || hasComptimeOnlySignature(funcType))
          return;
        auto lookup = lookupFuncByRefAttr(funcRef, unit);
        if (failed(lookup))
          return;
        auto *generic = std::get_if<GenericFuncOp>(&*lookup);
        if (!generic || generic->isActuallyGeneric() ||
            globalContext.lookupInstantiatedFunc(*generic, {}, {}))
          return;
        if (attempted.insert(generic->getOperation()).second)
          pending.emplace_back(*generic, loc);
      });
    };

    RuntimeReachability reachability(*this);
    unit->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!reachability.reachesRuntime(op))
        return WalkResult::skip();
      for (OpOperand &operand : op->getOpOperands()) {
        if (FailureOr<TypedAttr> value = getMaterializedValue(operand);
            succeeded(value))
          collectReferences(*value, op->getLoc());
      }
      if (auto globalOp = mlir::dyn_cast<GlobalOp>(op)) {
        if (std::optional<TypedAttr> initializer =
                globalOp.getInferredInitializer())
          collectReferences(*initializer, op->getLoc());
      }
      return WalkResult::advance();
    });
    for (const RecordedBuiltinCall &call :
         globalContext.getRecordedBuiltinCalls()) {
      for (TypedAttr arg : call.argValues)
        collectReferences(arg, call.location);
    }

    if (pending.empty())
      return success();

    bool succeededAll = true;
    for (auto [generic, loc] : pending) {
      LLVM_DEBUG(llvm::dbgs() << "instantiateRuntimeReferences: '"
                              << generic.getSymName() << "' referenced at "
                              << loc << "\n");
      auto result = globalContext.getOrInstantiateFunc(generic, {}, {});
      if (!std::holds_alternative<ComptimeError>(result))
        continue;
      auto &err = std::get<ComptimeError>(result);
      if (options.cslFlags.traceErrors)
        err.addCallStackEntry(loc, "in reference to '" + generic.getName() +
                                       "'");
      globalContext.reportComptimeEvalFailure(unit->getContext(), err);
      if (!options.lspMode)
        succeededAll = false;
    }
    if (!succeededAll)
      return failure();
  }
}
