//===- ComptimeEvalPass.cpp - Comptime Evaluation Pass -------------------===//
//
// Pass that evaluates and folds compile-time operations.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/GlobalEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "comptime-eval-pass"

namespace mlir {
namespace parsed_csl {

#define GEN_PASS_DEF_COMPTIMEEVALPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

namespace {

/// Populate inferred type attributes on operations and ForOp block arguments
LogicalResult populateInferredTypes(csl::CompilationUnitOp unit,
                                    ComptimeEvaluator &evaluator) {
  // Populate inferred type attributes on operations
  unit.walk([&](Operation *op) {
    if (auto inferredTypeOp = dyn_cast<InferredTypeInterface>(op)) {
      if (op->getNumResults() == 1) {
        FailureOr<Type> maybeType = evaluator.getInferredType(op->getResult(0));
        if (succeeded(maybeType)) {
          inferredTypeOp.setInferredType(*maybeType);
        }
      }
    }
  });

  // Populate inferred type attributes on ForOp block arguments
  if (unit.walk([&](parsed_csl::ForOp forOp) {
            // Element type
            if (!forOp.getBody().empty()) {
              BlockArgument elemArg = forOp.getElemArg();
              FailureOr<Type> elemType = evaluator.getInferredType(elemArg);
              if (failed(elemType)) {
                forOp.emitError() << "failed to infer element type for ForOp";
                return WalkResult::interrupt();
              }
              forOp.setInferredElemType(*elemType);

              // Index type (if present)
              if (forOp.hasIndex()) {
                BlockArgument indexArg = forOp.getIndexArg();
                FailureOr<Type> indexType = evaluator.getInferredType(indexArg);
                if (failed(indexType)) {
                  forOp.emitError() << "failed to infer index type for ForOp";
                  return WalkResult::interrupt();
                }
                forOp.setInferredIndexType(*indexType);
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  return success();
}

/// Populate constant_indices attribute on IndexAccessOp when indices are
/// comptime-known integer constants. This enables downstream passes to
/// handle struct/tuple index access (which requires compile-time constant
/// indices). Indices are cast to u16 for range validation.
void populateConstantIndices(csl::CompilationUnitOp unit,
                             ComptimeEvaluator &evaluator) {
  auto u16Type = IntegerType::get(unit.getContext(), 16, IntegerType::Unsigned);

  unit.walk([&](parsed_csl::IndexAccessOp op) {
    // Skip if already has constant_indices
    if (op.getConstantIndicesAttr())
      return;

    // Skip ops inside GenericFuncOps - they will be erased
    if (op->getParentOfType<parsed_csl::GenericFuncOp>())
      return;

    // Get context for this op
    const ComptimeEvalContext *context = evaluator.getContextForOp(op);
    if (!context)
      return;

    // Try to get comptime-known values for all indices
    SmallVector<int64_t> constantIndices;
    for (Value index : op.getIndices()) {
      FailureOr<TypedAttr> comptimeValue =
          context->getOrLoadComptimeRValue(index);
      if (failed(comptimeValue))
        return; // Not all indices are comptime-known

      // Cast to u16 to validate range
      FailureOr<TypedAttr> castedValue =
          csl::evaluateImplicitCast(*comptimeValue, u16Type);
      if (failed(castedValue))
        return; // Index doesn't fit in u16

      // Extract the integer value
      auto intAttr = mlir::dyn_cast<IntegerAttr>(*castedValue);
      if (!intAttr)
        return;

      constantIndices.push_back(intAttr.getValue().getZExtValue());
    }

    // All indices are comptime-known - set the attribute
    op.setConstantIndicesAttr(
        DenseI64ArrayAttr::get(op.getContext(), constantIndices));
  });
}

/// Populate case_values attribute on SwitchOp. The case values are resolved
/// from the case operands and cast to the input type. This is needed for
/// lowering to TypedCSL.
LogicalResult populateSwitchCaseValues(csl::CompilationUnitOp unit,
                                       ComptimeEvaluator &evaluator) {
  LogicalResult result = success();

  unit.walk([&](parsed_csl::SwitchOp op) {
    // Skip if already has case_values
    if (op.getCaseValues())
      return;

    // Skip ops inside GenericFuncOps - they will be erased
    if (op->getParentOfType<parsed_csl::GenericFuncOp>())
      return;

    // Get context for this op
    const ComptimeEvalContext *context = evaluator.getContextForOp(op);
    if (!context) {
      op.emitError() << "no comptime evaluation context for switch";
      result = failure();
      return;
    }

    // Get input type for casting case values
    FailureOr<Type> inputType = context->getInferredRValueType(op.getInput());
    if (failed(inputType)) {
      op.emitError() << "failed to infer switch input type";
      result = failure();
      return;
    }

    // Resolve all case values
    auto caseOperands = op.getCaseOperands();
    auto segments = op.getCaseOperandSegments();

    SmallVector<Attribute> caseValuesAttrs;
    unsigned offset = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
      SmallVector<Attribute> thisCaseValues;
      for (int32_t j = 0; j < segments[i]; ++j) {
        Value caseOperand = caseOperands[offset + j];

        FailureOr<TypedAttr> comptimeValue =
            context->getOrLoadComptimeRValue(caseOperand);
        if (failed(comptimeValue)) {
          op.emitError() << "case value is not comptime-known";
          result = failure();
          return;
        }

        // Cast to input type
        FailureOr<TypedAttr> castedValue =
            csl::evaluateCast(*comptimeValue, *inputType);
        if (failed(castedValue)) {
          op.emitError() << "case value cannot be cast to input type";
          result = failure();
          return;
        }

        thisCaseValues.push_back(*castedValue);
      }
      caseValuesAttrs.push_back(
          ArrayAttr::get(op.getContext(), thisCaseValues));
      offset += segments[i];
    }

    // Set the case_values attribute
    op.setCaseValuesAttr(ArrayAttr::get(op.getContext(), caseValuesAttrs));
  });

  return result;
}

/// Replace comptime-known operands with constant operations. Requires that
/// inferred types are already populated on all values!
LogicalResult replaceComptimeKnownValues(csl::CompilationUnitOp unit,
                                         ComptimeEvaluator &evaluator) {
  OpBuilder builder(unit.getContext());
  auto *dialect = unit.getContext()->getLoadedDialect<ParsedCSLDialect>();

  // Walk IR and replace comptime-known values with constants
  // This handles both rvalues and lvalues that can be represented as attrs
  unit.walk([&](Operation *op) {
    // Skip ops inside GenericFuncOps - they will be erased
    if (op->getParentOfType<parsed_csl::GenericFuncOp>())
      return;

    // Skip dead ops
    if (!evaluator.isOpAlive(op))
      return;

    builder.setInsertionPoint(op);

    for (OpOperand &operand : op->getOpOperands()) {
      FailureOr<TypedAttr> comptimeKnownValue =
          evaluator.getMaterializedValue(operand);
      if (failed(comptimeKnownValue))
        continue;
      Operation *constOp = dialect->materializeConstant(
          builder, *comptimeKnownValue, comptimeKnownValue->getType(),
          op->getLoc());
      operand.set(constOp->getResult(0));
      evaluator.markOpAlive(constOp);
    }
  });

  return success();
}

} // namespace

LogicalResult materializeComptimeEvaluation(csl::CompilationUnitOp unit,
                                            ComptimeEvaluator &evaluator) {
  // Instantiate the functions referenced from the runtime program. This
  // analyzes their bodies and must therefore complete before anything is
  // materialized.
  if (failed(evaluator.instantiateRuntimeReferences(unit)))
    return failure();

  // Serialize resource bitmaps after evaluation so fiber lowering and custom
  // builtin lowering can consume the claimed state.
  evaluator.getGlobalContext().getResourceBitmaps().storeToUnit(unit);

  // CallOp callee replacement and comptime arg stripping is now done inline
  // during analysis (in CallOpComptimeModel::evaluate, runtime path).
  // GenericFuncOp erasure and FuncOp renaming is done by
  // FinalizeInstantiationPass (runs after this pass).

  // Remove dead code (ops that were never evaluated due to early exits)
  // Must happen before type inference since dead ops have no inferred
  // types. Post-order + reverse iterator ensures users are visited before
  // producers within blocks, allowing safe erasure without dangling uses.
  unit.walk<WalkOrder::PostOrder, mlir::ReverseIterator>([&](Operation *op) {
    if (mlir::isa<csl::CompilationUnitOp>(op))
      return;

    // Affine expression in tensor access ops is not analyzed during
    // comptime but should survive dead code elimination.
    if (op->getParentOfType<TensorAccessOp>())
      return;

    // The region of a function can be empty, indicating a declaration.
    if (mlir::isa<FuncOp>(op)) {
      return;
    }

    // Erase blocks that have become empty. Empty modules keep their (empty)
    // block, as needed by the symbol table verifier.
    for (Region &region : op->getRegions()) {
      for (Block &block : llvm::make_early_inc_range(region.getBlocks())) {
        if (block.empty() && !isa<parsed_csl::ModuleOp>(op)) {
          LDBG() << "Erasing empty block in region of op: " << *op << "\n";
          block.erase();
        }
      }
    }
    if (!evaluator.isOpAlive(op)) {
      LDBG() << "Erasing dead operation: " << *op << "\n";
      op->erase();
    }
  });

  if (failed(populateInferredTypes(unit, evaluator))) {
    return failure();
  }

  // Populate constant_indices on IndexAccessOp for struct/tuple access
  populateConstantIndices(unit, evaluator);

  // Populate case_values on SwitchOp for lowering
  if (failed(populateSwitchCaseValues(unit, evaluator))) {
    return failure();
  }

  if (failed(replaceComptimeKnownValues(unit, evaluator))) {
    return failure();
  }

  {
    // Remove comptime and layout blocks, then insert new blocks for recorded
    // builtin calls grouped by their origin. Must run after dead code
    // elimination above because rematerialized builtin calls would otherwise
    // be detected as dead.
    unit.walk([&](parsed_csl::ComptimeBlockOp op) { op->erase(); });
    unit.walk([&](parsed_csl::LayoutBlockOp op) { op->erase(); });

    ArrayRef<RecordedBuiltinCall> recordedBuiltins =
        evaluator.getGlobalContext().getRecordedBuiltinCalls();

    if (!recordedBuiltins.empty()) {
      // Group builtin calls by their origin block kind
      SmallVector<const RecordedBuiltinCall *> comptimeCalls;
      SmallVector<const RecordedBuiltinCall *> layoutCalls;

      for (const RecordedBuiltinCall &call : recordedBuiltins) {
        if (call.fromLayoutBlock) {
          layoutCalls.push_back(&call);
        } else {
          comptimeCalls.push_back(&call);
        }
      }

      OpBuilder builder(unit.getBody(), unit.getBody()->end());

      // Emit comptime block if we have comptime calls
      if (!comptimeCalls.empty()) {
        auto comptimeBlockOp =
            parsed_csl::ComptimeBlockOp::create(builder, unit.getLoc());
        Block &block = comptimeBlockOp.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&block);
        for (const RecordedBuiltinCall *recordedCall : comptimeCalls) {
          if (failed(recordedCall->rematerializeCall(builder))) {
            return failure();
          }
        }
        parsed_csl::YieldOp::create(builder, unit.getLoc(), Value());
      }

      // Emit layout block if we have layout calls
      if (!layoutCalls.empty()) {
        builder.setInsertionPointToEnd(unit.getBody());
        auto layoutBlockOp =
            parsed_csl::LayoutBlockOp::create(builder, unit.getLoc());
        Block &block = layoutBlockOp.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&block);
        for (const RecordedBuiltinCall *recordedCall : layoutCalls) {
          if (failed(recordedCall->rematerializeCall(builder))) {
            return failure();
          }
        }
        parsed_csl::YieldOp::create(builder, unit.getLoc(), Value());
      }
    }
  }

  // Remove comptime calls and stores to comptime variables
  unit.walk([&](Operation *op) {
    if (evaluator.isErasedAsComptimeStatement(op))
      op->erase();
  });

  // Clean up GlobalOps after comptime evaluation:
  // 1. Clear init regions — the comptime value is stored in
  //    inferred_initializer, so the region is dead code.
  // 2. Remove globals referencing generic functions.
  // Skip unevaluated globals — they were never referenced and will be
  // erased as dead code later.
  LogicalResult globalsResult = success();
  unit.walk([&](parsed_csl::GlobalOp globalOp) {
    if (!evaluator.isOpAlive(globalOp))
      return;
    std::optional<Type> inferredType = globalOp.getInferredType();
    assert(inferredType &&
           "GlobalOp should have inferred type after comptime evaluation");
    assert(isa<csl::LValueType>(*inferredType) &&
           "GlobalOp inferred type should be an LValueType");
    csl::LValueType lvalueType = cast<csl::LValueType>(*inferredType);

    if (isa<csl::GenericFunctionType, FunctionType>(
            lvalueType.getValueType())) {
      if (!globalOp.getIsConst()) {
        globalOp.emitError() << "global with comptime-only type must be const";
        globalsResult = failure();
        return;
      }
      LDBG() << "Erasing comptime-only global: " << globalOp.getName() << "\n";
      globalOp.erase();
      return;
    }

    // Clear init region — comptime value is in inferred_initializer attr
    if (globalOp.hasInitializer()) {
      globalOp.getInitRegion().front().clear();
      globalOp.getInitRegion().dropAllReferences();
      globalOp.getInitRegion().getBlocks().clear();
    }
  });
  if (failed(globalsResult))
    return failure();

  // Simplify regions to clean up dead code
  {
    IRRewriter rewriter(unit);
    (void)mlir::simplifyRegions(rewriter, unit->getRegions());
  }

  return success();
}

namespace {

struct ComptimeEvalPass : public impl::ComptimeEvalPassBase<ComptimeEvalPass> {
  using Base = impl::ComptimeEvalPassBase<ComptimeEvalPass>;
  using Base::Base;

  std::shared_ptr<csl::CslModuleResolver> moduleResolver;

  ComptimeEvalPass() = default;

  ComptimeEvalPass(std::shared_ptr<csl::CslModuleResolver> resolver,
                   const ComptimeEvalPassOptions &options)
      : Base(options), moduleResolver(std::move(resolver)) {}

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    ComptimeEvalOptions evaluatorOptions(this->cslFlags);
    ComptimeEvaluator evaluator(unit, moduleResolver, evaluatorOptions);
    if (failed(evaluator)) {
      signalPassFailure();
      return;
    }

    if (failed(materializeComptimeEvaluation(unit, evaluator)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
createComptimeEvalPass(std::shared_ptr<csl::CslModuleResolver> resolver,
                       const ComptimeEvalPassOptions &options) {
  return std::make_unique<ComptimeEvalPass>(std::move(resolver), options);
}

} // namespace parsed_csl
} // namespace mlir
