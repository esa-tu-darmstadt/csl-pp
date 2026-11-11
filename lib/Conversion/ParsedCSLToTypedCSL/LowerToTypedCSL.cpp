//===- LowerToTypedCSL.cpp - Lower ParsedCSL to TypedCSL -------*- C++ -*-===//
//
// This file implements the pass for converting ParsedCSL dialect to TypedCSL
// dialect, materializing all inferred types and inserting explicit casts.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Conversion/Passes.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "lower-to-typed-csl"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_CONVERTPARSEDCSLTOTYPEDCSLPASS
#include "csl-mlir/Conversion/Passes.h.inc"

} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class ParsedCSLTypeConverter : public TypeConverter {
public:
  ParsedCSLTypeConverter() {
    // Identity conversion for concrete types (CSL types, FunctionType, etc.)
    addConversion([](Type type) -> Type { return type; });

    // Convert parsed_csl.unresolved to inferred concrete type
    addConversion([&](Value value) -> Type {
      auto maybeType = parsed_csl::getInferredType(value);
      if (failed(maybeType)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "No inferred type for value: " << value << "\n");
        return nullptr;
      }
      return *maybeType;
    });
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Compute the number of breaking control regions between a terminator op and
/// its target. Walks up from the op through parent regions counting levels.
///
/// For break/continue: target is the loop/labeled_block matching the label
/// (or innermost breakable if no label).
/// For return/unreachable: target is the enclosing FuncOp.
///
/// Works with both ParsedCSL and TypedCSL ops during conversion, using
/// LabeledOpInterface from the CSL base dialect.
static int computeNumBreakingRegions(Operation *op,
                                     std::optional<StringRef> targetLabel) {
  int n = 0;
  Operation *parent = op->getParentOp();
  while (parent) {
    // Each parent that owns regions contributes one level per region the op
    // is nested in (typically 1).
    n++;

    // Check if this parent is the target.
    if (auto labeled = mlir::dyn_cast<csl::LabeledOpInterface>(parent)) {
      if (!targetLabel) {
        // No label: target is the innermost breakable construct.
        return n;
      }
      auto parentLabel = labeled.getOptionalLabel();
      if (parentLabel && *parentLabel == *targetLabel)
        return n;
    }

    parent = parent->getParentOp();
  }

  // Should not reach here if the IR is well-formed.
  llvm_unreachable("could not find target for breaking control flow op");
}

/// Compute N for return/unreachable — count regions up to the enclosing FuncOp.
/// Works with both parsed_csl::FuncOp and typed_csl::FuncOp.
static int computeNumBreakingRegionsToFunc(Operation *op) {
  int n = 0;
  Operation *parent = op->getParentOp();
  while (parent) {
    n++;
    if (mlir::isa<parsed_csl::FuncOp>(parent) ||
        mlir::isa<typed_csl::FuncOp>(parent))
      return n;
    parent = parent->getParentOp();
  }
  llvm_unreachable("return/unreachable not inside a function");
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Base class for ParsedCSL to TypedCSL conversion patterns
template <typename OpTy>
class ParsedCSLOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  ParsedCSLOpConversionPattern(const ParsedCSLTypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit),
        typeConverter(typeConverter) {}

protected:
  const ParsedCSLTypeConverter &typeConverter;
};

/// Convert parsed_csl.constant and parsed_csl.builtin_type to
/// typed_csl.constant. Now that ParsedCSL functions are IsolatedFromAbove, this
/// is a simple 1:1 conversion.
template <typename OpTy>
struct ConstantOrBuiltinTypeOpConversion
    : public ParsedCSLOpConversionPattern<OpTy> {
  using Base = ParsedCSLOpConversionPattern<OpTy>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Simple 1:1 conversion - no inlining needed since ParsedCSL funcs are
    // now IsolatedFromAbove
    rewriter.replaceOpWithNewOp<typed_csl::ConstantOp>(op, adaptor.getValue());
    return success();
  }
};

/// Convert parsed_csl.binary to typed_csl.binary
struct BinaryOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::BinaryOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::BinaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Create typed binary operation
    auto binaryOp = typed_csl::BinaryOp::create(
        rewriter, loc, *resultType, adaptor.getKind(), adaptor.getLhs(),
        adaptor.getRhs());

    rewriter.replaceOp(op, binaryOp);
    return success();
  }
};

/// Convert parsed_csl.unary to typed_csl.unary
struct UnaryOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::UnaryOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::UnaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Create typed unary operation
    auto unaryOp = typed_csl::UnaryOp::create(
        rewriter, loc, *resultType, adaptor.getKind(), adaptor.getOperand());

    rewriter.replaceOp(op, unaryOp);
    return success();
  }
};

/// Convert parsed_csl.cast to typed_csl.cast
struct CastOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::CastOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get target type from the inferred result type
    FailureOr<Type> targetType = typeConverter.convertType(op.getResult());
    if (failed(targetType))
      return rewriter.notifyMatchFailure(
          op, "failed to get inferred target type for cast");

    // Create typed cast operation
    rewriter.replaceOpWithNewOp<typed_csl::CastOp>(op, *targetType,
                                                   adaptor.getSource());
    return success();
  }
};

/// Convert parsed_csl.var to typed_csl.var (local variables only).
/// Globals now use parsed_csl.global -> typed_csl.global via
/// GlobalOpConversion. Cast insertion for initializer is handled by
/// NormalizeCastsPass.
struct VariableOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::VariableOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::VariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred lvalue type
    FailureOr<Type> lvalueType = typeConverter.convertType(op.getResult());
    if (failed(lvalueType))
      return rewriter.notifyMatchFailure(
          op, "failed to get inferred lvalue type for variable");

    auto typedLValueType = llvm::cast<typed_csl::LValueType>(*lvalueType);

    // Simple 1:1 conversion to typed_csl::VariableOp
    rewriter.replaceOpWithNewOp<typed_csl::VariableOp>(
        op, typedLValueType, op.getNameAttr(), adaptor.getInitializer(),
        op.getIsConstAttr());
    return success();
  }
};

/// Convert parsed_csl.global to typed_csl.global (symbol-based, no SSA result)
struct GlobalOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::GlobalOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the inferred type (set during comptime eval)
    std::optional<Type> inferredType = op.getInferredType();
    if (!inferredType)
      return rewriter.notifyMatchFailure(op, "missing inferred type on global");

    inferredType = typeConverter.convertType(*inferredType);

    auto lvalueType = mlir::dyn_cast<csl::LValueType>(*inferredType);
    if (!lvalueType)
      return rewriter.notifyMatchFailure(op, "inferred type is not LValueType");

    // Convert to typed_csl LValueType
    Type valueType = lvalueType.getValueType();
    auto typedLValueType =
        typed_csl::LValueType::get(valueType, lvalueType.getIsConst());

    TypedAttr initValue =
        op.getInferredInitializer() ? op.getInferredInitializerAttr() : nullptr;

    // Create typed_csl::GlobalOp (simple 1:1 mapping)
    typed_csl::GlobalOp::create(
        rewriter, op.getLoc(), TypeAttr::get(typedLValueType.getValueType()),
        op.getSymNameAttr(), initValue, op.getIsConstAttr(),
        op.getIsExportAttr(), op.getIsExternAttr(), op.getLinknameAttr(),
        op.getAlignAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.constant with GlobalRefAttr to a typed_csl.constant with
/// the TypedCSL lvalue type.
/// This handles global references that were materialized during comptime eval
struct GlobalRefConstantOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ConstantOp> {
  GlobalRefConstantOpConversion(const ParsedCSLTypeConverter &typeConverter,
                                MLIRContext *context)
      : ParsedCSLOpConversionPattern(typeConverter, context, 2) {}

  LogicalResult
  matchAndRewrite(parsed_csl::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto globalRef = mlir::dyn_cast<csl::GlobalRefAttr>(adaptor.getValue());
    if (!globalRef)
      return failure(); // Not a GlobalRefAttr, let other patterns handle it

    // Module ID should be absent after inlining
    if (globalRef.getModuleIdentifier())
      return rewriter.notifyMatchFailure(
          op, "GlobalRefAttr with module_id should have been resolved by "
              "InlineModulesPass");

    // Convert to typed_csl LValueType
    csl::LValueType lvalueType = globalRef.getType();
    auto typedLValueType = typed_csl::LValueType::get(lvalueType.getValueType(),
                                                      lvalueType.getIsConst());

    rewriter.replaceOpWithNewOp<typed_csl::ConstantOp>(
        op, csl::GlobalRefAttr::get(op->getContext(), std::nullopt,
                                    globalRef.getSymbolName(),
                                    typedLValueType));
    return success();
  }
};

/// Convert parsed_csl.constant with FuncRefAttr to a typed_csl.constant with
/// the TypedCSL function type.
/// This handles function references that were materialized during comptime eval
struct FuncRefConstantOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ConstantOp> {
  FuncRefConstantOpConversion(const ParsedCSLTypeConverter &typeConverter,
                              MLIRContext *context)
      : ParsedCSLOpConversionPattern(typeConverter, context, 2) {}

  LogicalResult
  matchAndRewrite(parsed_csl::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcRef = mlir::dyn_cast<csl::FuncRefAttr>(adaptor.getValue());
    if (!funcRef)
      return failure(); // Not a FuncRefAttr, let other patterns handle it

    // Module ID should be absent after inlining
    if (funcRef.getModuleIdentifier())
      return rewriter.notifyMatchFailure(
          op, "FuncRefAttr with module_id should have been resolved by "
              "InlineModulesPass");

    // At this stage all FuncRefAttr should have concrete FunctionType
    if (funcRef.isGeneric()) {
      return rewriter.notifyMatchFailure(
          op, "Generic function reference should have been resolved by "
              "ComptimeEvalPass");
    }
    FunctionType funcType = funcRef.getFunctionType();
    // Remove explicit void result type (TypedCSL uses empty results for void)
    if (funcType.getNumResults() == 1 &&
        mlir::isa<csl::VoidType>(funcType.getResult(0))) {
      funcType = FunctionType::get(op->getContext(), funcType.getInputs(), {});
    }

    rewriter.replaceOpWithNewOp<typed_csl::ConstantOp>(
        op, csl::FuncRefAttr::get(op->getContext(), std::nullopt,
                                  funcRef.getSymbolName(), funcType));
    return success();
  }
};

/// Convert parsed_csl.load to typed_csl.load
struct LoadOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::LoadOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Simply replace with typed load
    rewriter.replaceOpWithNewOp<typed_csl::LoadOp>(op, adaptor.getLvalue());
    return success();
  }
};

/// Convert parsed_csl.assign to typed_csl.assign
struct AssignOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::AssignOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::AssignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create typed assign
    rewriter.replaceOpWithNewOp<typed_csl::AssignOp>(op, adaptor.getLvalue(),
                                                     adaptor.getValue());
    return success();
  }
};

/// Convert parsed_csl.inline_asm to typed_csl.inline_asm
struct InlineAsmOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::InlineAsmOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::InlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // All string values were resolved during comptime eval and stored
    // as inferred attributes on the op.
    auto asmStr = op.getInferredAsmStringAttr();
    if (!asmStr)
      return rewriter.notifyMatchFailure(op, "missing inferred_asm_string");

    auto outputConstraints = op.getInferredOutputConstraintsAttr();
    if (!outputConstraints)
      outputConstraints = rewriter.getArrayAttr({});

    auto inputConstraints = op.getInferredInputConstraintsAttr();
    if (!inputConstraints)
      inputConstraints = rewriter.getArrayAttr({});

    auto clobbers = op.getInferredClobbersAttr();
    if (!clobbers)
      clobbers = rewriter.getArrayAttr({});

    typed_csl::InlineAsmOp::create(
        rewriter, op.getLoc(), adaptor.getOutputs(), adaptor.getInputs(),
        asmStr, op.getOutputNamesAttr(), outputConstraints,
        op.getInputNamesAttr(), inputConstraints, clobbers, op.getIsVolatile());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.address_of to typed_csl.address_of
struct AddressOfOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::AddressOfOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::AddressOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result type (should be !csl.ptr<T>)
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Create typed address_of operation
    rewriter.replaceOpWithNewOp<typed_csl::AddressOfOp>(op, *resultType,
                                                        adaptor.getOperand());
    return success();
  }
};

/// Convert parsed_csl.dereference to typed_csl.dereference
struct DereferenceOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::DereferenceOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::DereferenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result type (should be !csl.lvalue<T>)
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Create typed dereference operation
    rewriter.replaceOpWithNewOp<typed_csl::DereferenceOp>(op, *resultType,
                                                          adaptor.getOperand());
    return success();
  }
};

/// Convert parsed_csl.select to typed_csl.if (expression or statement form)
struct SelectOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::SelectOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Select without else should have been folded away by comptime eval
    // (since it requires comptime-known true condition)
    if (!op.hasElseBranch()) {
      return rewriter.notifyMatchFailure(
          op, "select without else requires comptime known true condition and "
              "should have been canonicalized away");
    }

    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Void select → statement form (no result)
    Type result = mlir::isa<csl::VoidType>(*resultType) ? Type() : *resultType;
    if (!result && !op.getResult().use_empty())
      llvm_unreachable("void select result should have no uses");

    auto ifOp =
        typed_csl::IfOp::create(rewriter, loc, result, adaptor.getCondition());
    rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
                                ifOp.getElseRegion().end());

    if (result)
      rewriter.replaceOp(op, ifOp.getResult());
    else
      rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.short_circuit to typed_csl.short_circuit
struct ShortCircuitOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ShortCircuitOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ShortCircuitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create typed short-circuit operation with region
    auto shortCircuitOp = typed_csl::ShortCircuitOp::create(
        rewriter, loc, op.getKind(), adaptor.getLhs());

    // Move the rhs region
    rewriter.inlineRegionBefore(op.getRhsRegion(),
                                shortCircuitOp.getRhsRegion(),
                                shortCircuitOp.getRhsRegion().end());

    rewriter.replaceOp(op, shortCircuitOp);
    return success();
  }
};

/// Convert parsed_csl.switch to typed_csl.switch
struct SwitchOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::SwitchOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get resolved case values (set during comptime evaluation)
    auto caseValuesAttr = op.getCaseValues();
    if (!caseValuesAttr) {
      return rewriter.notifyMatchFailure(
          op, "switch case values not resolved during comptime evaluation");
    }

    // Determine if this is an expression (has result) or statement
    std::optional<Type> resultType;
    FailureOr<Type> inferredType = typeConverter.convertType(op.getResult());
    if (succeeded(inferredType) && !mlir::isa<csl::VoidType>(*inferredType)) {
      resultType = *inferredType;
    }

    // Create typed switch operation
    // The create function takes the number of case regions as the last
    // parameter
    unsigned numCaseRegions = caseValuesAttr->size();
    Type result = resultType.value_or(Type());
    auto switchOp =
        typed_csl::SwitchOp::create(rewriter, loc, result, adaptor.getInput(),
                                    *caseValuesAttr, numCaseRegions);

    // Move all case regions (remember: else is stored first in MLIR, but
    // getCaseRegions() returns only the case regions)
    auto srcCaseRegions = op.getCaseRegions();
    auto dstCaseRegions = switchOp.getCaseRegions();
    for (size_t i = 0; i < srcCaseRegions.size(); ++i) {
      rewriter.inlineRegionBefore(srcCaseRegions[i], dstCaseRegions[i],
                                  dstCaseRegions[i].end());
    }

    // Move else region if present
    if (op.hasElseBranch()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), switchOp.getElseRegion(),
                                  switchOp.getElseRegion().end());
    }

    if (resultType) {
      rewriter.replaceOp(op, switchOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// Convert parsed_csl.if to typed_csl.if
struct IfOpConversion : public ParsedCSLOpConversionPattern<parsed_csl::IfOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create typed if operation (statement form, no result)
    auto newIf = typed_csl::IfOp::create(rewriter, loc, /*resultType=*/Type(),
                                         adaptor.getCondition());

    // Convert then region
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());

    // Convert else region if present
    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                  newIf.getElseRegion().end());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.for to typed_csl.for
struct ForOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ForOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get block argument types BEFORE inlining the region (while op still owns
    // the body)
    SmallVector<Type> argTypes;
    if (!op.getBody().empty()) {
      Block &oldBodyBlock = op.getBody().front();
      for (BlockArgument arg : oldBodyBlock.getArguments()) {
        FailureOr<Type> argType = typeConverter.convertType(arg);
        if (failed(argType))
          return rewriter.notifyMatchFailure(
              op, "failed to convert block argument type");
        argTypes.push_back(*argType);
      }
    }

    // Create typed for operation
    auto newFor = typed_csl::ForOp::create(
        rewriter, loc, adaptor.getIterable(), op.getElemNameAttr(),
        op.getIndexNameAttr(), op.getLabelAttr(), op.getIsInlineAttr());

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), newFor.getBody(),
                                newFor.getBody().end());

    // Convert block argument types in the body using pre-computed types
    if (!newFor.getBody().empty()) {
      Block &bodyBlock = newFor.getBody().front();
      for (unsigned i = 0; i < argTypes.size(); ++i) {
        bodyBlock.getArgument(i).setType(argTypes[i]);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.while to typed_csl.while
struct WhileOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::WhileOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create typed while operation with empty regions
    auto newWhile =
        typed_csl::WhileOp::create(rewriter, loc, op.getLabelAttr());

    // Move the condition region
    rewriter.inlineRegionBefore(op.getCondRegion(), newWhile.getCondRegion(),
                                newWhile.getCondRegion().end());

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), newWhile.getBody(),
                                newWhile.getBody().end());

    // Move the continuation region if present
    if (!op.getContinuation().empty()) {
      rewriter.inlineRegionBefore(op.getContinuation(),
                                  newWhile.getContinuation(),
                                  newWhile.getContinuation().end());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.labeled_block to typed_csl.labeled_block
struct LabeledBlockOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::LabeledBlockOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::LabeledBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "failed to get inferred result type for labeled block");

    // Create typed labeled block operation — omit result for void blocks
    bool isVoid = mlir::isa<csl::VoidType>(*resultType);
    typed_csl::LabeledBlockOp newOp;
    if (isVoid) {
      newOp = typed_csl::LabeledBlockOp::create(rewriter, op.getLoc(),
                                                TypeRange{}, op.getLabelAttr());
    } else {
      newOp = typed_csl::LabeledBlockOp::create(rewriter, op.getLoc(),
                                                *resultType, op.getLabelAttr());
    }

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());

    if (isVoid) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }
};

/// Convert parsed_csl.break to typed_csl.break
struct BreakOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::BreakOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::BreakOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int n = computeNumBreakingRegions(op, op.getTargetLabel());
    rewriter.replaceOpWithNewOp<typed_csl::BreakOp>(
        op, n, op.getTargetLabelAttr(), adaptor.getValue());
    return success();
  }
};

/// Convert parsed_csl.continue to typed_csl.continue
struct ContinueOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ContinueOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ContinueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int n = computeNumBreakingRegions(op, op.getTargetLabel());
    rewriter.replaceOpWithNewOp<typed_csl::ContinueOp>(op, n,
                                                       op.getTargetLabelAttr());
    return success();
  }
};

/// Convert parsed_csl.comptime to typed_csl.comptime
struct ComptimeBlockOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ComptimeBlockOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ComptimeBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create typed comptime operation
    auto newComptime = typed_csl::ComptimeOp::create(rewriter, loc);

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), newComptime.getBody(),
                                newComptime.getBody().end());

    // Ensure the body has a terminator (may be missing if the comptime
    // block was not reconstructed by ComptimeEvalPass).
    Block &body = newComptime.getBody().front();
    if (body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&body);
      typed_csl::YieldOp::create(rewriter, loc, Value());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.layout to typed_csl.layout
struct LayoutBlockOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::LayoutBlockOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::LayoutBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create typed layout operation
    auto newLayout = typed_csl::LayoutOp::create(rewriter, loc);

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), newLayout.getBody(),
                                newLayout.getBody().end());

    // Ensure the body has a terminator (may be missing if the layout
    // block was not reconstructed by ComptimeEvalPass).
    Block &body = newLayout.getBody().front();
    if (body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&body);
      typed_csl::YieldOp::create(rewriter, loc, Value());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.return to typed_csl.return
/// Cast insertion is handled by NormalizeCastsPass
struct ReturnOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ReturnOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int n = computeNumBreakingRegionsToFunc(op);
    if (adaptor.getOperands().empty()) {
      rewriter.replaceOpWithNewOp<typed_csl::ReturnOp>(op, n);
    } else {
      rewriter.replaceOpWithNewOp<typed_csl::ReturnOp>(
          op, n, adaptor.getOperands()[0]);
    }
    return success();
  }
};

/// Convert parsed_csl.unreachable to typed_csl.unreachable
struct UnreachableOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::UnreachableOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::UnreachableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int n = computeNumBreakingRegionsToFunc(op);
    rewriter.replaceOpWithNewOp<typed_csl::UnreachableOp>(op, n);
    return success();
  }
};

/// Convert parsed_csl.suspend to typed_csl.suspend (non-terminator).
/// LowerSuspendingSCFToCFPass will later replace with typed_csl.suspend_br.
struct SuspendOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::SuspendOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::SuspendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    typed_csl::SuspendOp::create(rewriter, op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.await to typed_csl.await (non-terminator).
/// LowerSuspendingSCFToCFPass will later replace with typed_csl.await_br.
struct AwaitOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::AwaitOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::AwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Remove explicit void result type.
    Type awaitResultType = *resultType;
    if (mlir::isa<csl::VoidType>(awaitResultType))
      awaitResultType = {};

    if (awaitResultType) {
      auto awaitOp = typed_csl::AwaitOp::create(
          rewriter, op.getLoc(), awaitResultType, adaptor.getFrameHandle());
      rewriter.replaceOp(op, awaitOp.getResult());
    } else {
      assert(op->use_empty() && "Void await must not have any use");
      typed_csl::AwaitOp::create(rewriter, op.getLoc(), TypeRange{},
                                 adaptor.getFrameHandle());
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// Convert parsed_csl.yield to typed_csl.yield
struct YieldOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::YieldOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In WhileOp's condition region, produce CondYieldOp instead of YieldOp.
    // The parent may already be lowered to typed_csl::WhileOp.
    if (auto whileOp = mlir::dyn_cast<parsed_csl::WhileOp>(op->getParentOp())) {
      if (op->getParentRegion() == &whileOp.getCondRegion()) {
        rewriter.replaceOpWithNewOp<typed_csl::CondYieldOp>(op,
                                                            adaptor.getValue());
        return success();
      }
    } else if (auto whileOp =
                   mlir::dyn_cast<typed_csl::WhileOp>(op->getParentOp())) {
      if (op->getParentRegion() == &whileOp.getCondRegion()) {
        rewriter.replaceOpWithNewOp<typed_csl::CondYieldOp>(op,
                                                            adaptor.getValue());
        return success();
      }
    }
    // Strip void values — void YieldOp serves as fallthrough terminator.
    Value value = adaptor.getValue();
    if (value && mlir::isa<csl::VoidType>(value.getType()))
      value = {};
    rewriter.replaceOpWithNewOp<typed_csl::YieldOp>(op, value);
    return success();
  }
};

/// Convert parsed_csl.call to typed_csl.call or typed_csl.indirect_call
/// Cast insertion for arguments is handled by NormalizeCastsPass
struct CallOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::CallOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Remove explicit void result type
    Type resultType = typeConverter.convertType(op.getResult());
    if (mlir::isa<csl::VoidType>(resultType))
      resultType = {};

    Value callee = adaptor.getCallee();
    ValueRange args = adaptor.getArgs();

    // Check if this is a direct call (callee is a constant FuncRefAttr or
    // FuncAddrAttr)
    std::optional<FlatSymbolRefAttr> calleeSymbol;

    if (auto constantOp = callee.getDefiningOp<typed_csl::ConstantOp>()) {
      if (auto funcRef =
              mlir::dyn_cast<csl::FuncRefAttr>(constantOp.getValue()))
        calleeSymbol = funcRef.getSymbolName();
      else if (auto funcAddr =
                   mlir::dyn_cast<csl::FuncAddrAttr>(constantOp.getValue()))
        calleeSymbol = funcAddr.getFuncRef().getSymbolName();
    }

    Operation *newCall;
    if (calleeSymbol) {
      // Direct call
      newCall =
          typed_csl::CallOp::create(rewriter, loc, resultType, *calleeSymbol,
                                    args, /*arg_attrs=*/{}, /*res_attrs=*/{});
    } else {
      // Indirect call
      newCall = typed_csl::IndirectCallOp::create(rewriter, loc, resultType,
                                                  callee, args);
    }

    if (resultType) {
      rewriter.replaceOp(op, newCall->getResult(0));
    } else {
      assert(op->use_empty() && "Void function call must not have any use");
      rewriter.eraseOp(op);
    }

    return success();
  }
};

/// Convert parsed_csl.builtin_call to typed_csl.builtin_call
/// Cast insertion for arguments is handled by NormalizeCastsPass
struct BuiltinCallOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::BuiltinCallOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::BuiltinCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the inferred result type
    Type resultType = typeConverter.convertType(op.getResult());
    if (mlir::isa<csl::VoidType>(resultType))
      resultType = {};

    auto builtinKind = op.resolveBuiltin();
    if (!builtinKind) {
      return op.emitError("unknown builtin function: @") << op.getBuiltinName();
    }

    // @field(base, "name") lowers to field_access or tuple_index_access
    // instead of builtin_call, because it can produce lvalue results.
    if (*builtinKind == csl::BuiltinKind::Field) {
      assert(adaptor.getArgs().size() == 2 &&
             "@field requires exactly 2 arguments");

      Value base = adaptor.getArgs()[0];

      // Extract field name from the second argument (ComptimeStringAttr).
      Value fieldNameVal = adaptor.getArgs()[1];
      auto fieldNameConst = fieldNameVal.getDefiningOp<typed_csl::ConstantOp>();
      if (!fieldNameConst)
        return op.emitError("@field requires a comptime string field name");
      auto fieldNameAttr =
          mlir::dyn_cast<csl::ComptimeStringAttr>(fieldNameConst.getValue());
      if (!fieldNameAttr)
        return op.emitError("@field requires a comptime string field name");

      StringRef fieldName = fieldNameAttr.getValue();

      // Determine base value type (unwrap lvalue if present).
      Type baseType = base.getType();
      Type baseValueType = baseType;
      if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(baseType))
        baseValueType = lvalueType.getValueType();

      auto structType = mlir::dyn_cast<csl::StructType>(baseValueType);
      if (!structType)
        return op.emitError("@field base must be a struct type");

      // If it's a tuple and the field name is a numeric index, use
      // TupleIndexAccessOp.
      unsigned index;
      if (structType.isTuple() && !fieldName.getAsInteger(10, index)) {
        rewriter.replaceOpWithNewOp<typed_csl::TupleIndexAccessOp>(
            op, resultType, base, static_cast<int64_t>(index));
      } else {
        rewriter.replaceOpWithNewOp<typed_csl::FieldAccessOp>(
            op, resultType, base, rewriter.getStringAttr(fieldName));
      }
      return success();
    }

    // @await(future) or @await(future, .{ .offload = true }) → typed_csl.await
    if (*builtinKind == csl::BuiltinKind::Await) {
      Value future = adaptor.getArgs()[0];

      // Extract offload flag from optional config struct.
      UnitAttr offloadAttr;
      if (adaptor.getArgs().size() == 2) {
        Value configVal = adaptor.getArgs()[1];
        if (auto constOp = configVal.getDefiningOp<typed_csl::ConstantOp>()) {
          if (auto structAttr =
                  mlir::dyn_cast<csl::StructValueAttr>(constOp.getValue())) {
            if (TypedAttr field = structAttr.getField("offload")) {
              if (auto boolVal = mlir::dyn_cast<BoolAttr>(field)) {
                if (boolVal.getValue())
                  offloadAttr = rewriter.getUnitAttr();
              }
            }
          }
        }
      }

      // Strip VoidType from result (no VoidType in TypedCSL).
      Type awaitResultType = resultType;
      if (awaitResultType && mlir::isa<csl::VoidType>(awaitResultType))
        awaitResultType = {};

      if (awaitResultType) {
        auto awaitOp =
            typed_csl::AwaitOp::create(rewriter, loc, awaitResultType, future);
        if (offloadAttr)
          awaitOp.setOffloadAttr(offloadAttr);
        rewriter.replaceOp(op, awaitOp.getResult());
      } else {
        auto awaitOp =
            typed_csl::AwaitOp::create(rewriter, loc, TypeRange{}, future);
        if (offloadAttr)
          awaitOp.setOffloadAttr(offloadAttr);
        rewriter.eraseOp(op);
      }
      return success();
    }

    // @await_parallel(future1, future2, ...) → typed_csl.await_parallel
    // The result is the tuple of the callee results. If all callees return
    // void, it has a single value, which comptime evaluation substituted for
    // every use, so the op gets no result.
    if (*builtinKind == csl::BuiltinKind::AwaitParallel) {
      if (!resultType || csl::isVoidLikeType(resultType)) {
        typed_csl::AwaitParallelOp::create(rewriter, loc, Type(),
                                           adaptor.getArgs());
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOpWithNewOp<typed_csl::AwaitParallelOp>(
            op, resultType, adaptor.getArgs());
      }
      return success();
    }

    // @frame_address() → typed_csl.frame_address
    if (*builtinKind == csl::BuiltinKind::FrameAddress) {
      auto frameAddrOp =
          typed_csl::FrameAddressOp::create(rewriter, loc, resultType);
      rewriter.replaceOp(op, frameAddrOp.getResult());
      return success();
    }

    auto newBuiltinCall = typed_csl::BuiltinCallOp::create(
        rewriter, loc, resultType, *builtinKind, adaptor.getArgs());

    assert(resultType ||
           (op->use_empty() && "Void builtin must not have any use"));
    rewriter.replaceOp(op, newBuiltinCall.getResult());

    return success();
  }
};

/// Convert parsed_csl.func to typed_csl.func
struct FuncOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::FuncOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Remove explicit void result type, but preserve CoroFrameType for async
    FunctionType fnType = op.getFunctionType();
    if (fnType.getNumResults() > 0 &&
        mlir::isa<csl::VoidType>(fnType.getResult(0)))
      fnType = fnType.getWithoutArgsAndResults({}, BitVector{1}.set(0));

    // Strip VoidType from async_result_type (TypedCSL uses nullopt for void).
    auto asyncResultTypeAttr = op.getAsyncResultTypeAttr();
    if (asyncResultTypeAttr &&
        mlir::isa<csl::VoidType>(asyncResultTypeAttr.getValue()))
      asyncResultTypeAttr = {};

    // Create new typed function with attributes copied from ParsedCSL FuncOp
    auto newFunc = typed_csl::FuncOp::create(
        rewriter, loc, op.getName(), fnType, op.getIsTask(), op.getIsInline(),
        op.getIsNoinline(), op.getIsExport(), op.getIsExtern(), op.getIsAsync(),
        op.getStackSizeAttr(), op.getLinknameAttr(), /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr, asyncResultTypeAttr,
        op.getMaxRecursionDepthAttr());

    // Move the body region
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.getBody().end());

    // Convert block argument types
    Region &newFuncBody = newFunc.getBody();
    if (!newFuncBody.empty()) {
      TypeConverter::SignatureConversion signatureConversion(
          fnType.getNumInputs());
      for (unsigned i = 0; i < fnType.getNumInputs(); ++i) {
        auto lvalueType =
            csl::LValueType::get(fnType.getInput(i), /*isConst=*/true);
        signatureConversion.addInputs(i, lvalueType);
      }
      if (failed(rewriter.convertRegionTypes(&newFuncBody, typeConverter,
                                             &signatureConversion))) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert function body types");
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert parsed_csl.construct_tuple to typed_csl.construct_tuple
struct ConstructTupleOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ConstructTupleOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ConstructTupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Verify it's a struct type
    if (!mlir::isa<csl::StructType>(*resultType))
      return rewriter.notifyMatchFailure(
          op, "inferred result type is not a struct type");

    // Create typed construct_tuple operation
    rewriter.replaceOpWithNewOp<typed_csl::ConstructTupleOp>(
        op, *resultType, adaptor.getFieldValues());

    return success();
  }
};

/// Convert parsed_csl.construct_struct to typed_csl.construct_struct
struct ConstructStructOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ConstructStructOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ConstructStructOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Verify it's a struct type
    if (!mlir::isa<csl::StructType>(*resultType))
      return rewriter.notifyMatchFailure(
          op, "inferred result type is not a struct type");

    // Create typed construct_struct operation
    rewriter.replaceOpWithNewOp<typed_csl::ConstructStructOp>(
        op, *resultType, op.getFieldNamesAttr(), adaptor.getFieldValues());

    return success();
  }
};

/// Convert parsed_csl.construct_array to typed_csl.construct_array
struct ConstructArrayOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::ConstructArrayOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::ConstructArrayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Verify it's an array type
    if (!mlir::isa<csl::ArrayType>(*resultType))
      return rewriter.notifyMatchFailure(
          op, "inferred result type is not an array type");

    // Create typed construct_array operation
    rewriter.replaceOpWithNewOp<typed_csl::ConstructArrayOp>(
        op, *resultType, adaptor.getElements());

    return success();
  }
};

/// Convert parsed_csl.field_access to typed_csl.field_access
struct FieldAccessOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::FieldAccessOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::FieldAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result lvalue type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "failed to get inferred result lvalue type");

    // Create typed field_access operation
    rewriter.replaceOpWithNewOp<typed_csl::FieldAccessOp>(
        op, *resultType, adaptor.getBase(), op.getFieldNameAttr());

    return success();
  }
};

/// Convert parsed_csl.index_access to typed_csl ops
/// - For struct/tuple: typed_csl.tuple_index_access (single comptime index)
/// - For array: typed_csl.array_index_access (runtime indices)
/// Both ops support lvalue and rvalue base operands.
struct IndexAccessOpConversion
    : public ParsedCSLOpConversionPattern<parsed_csl::IndexAccessOp> {
  using ParsedCSLOpConversionPattern::ParsedCSLOpConversionPattern;

  LogicalResult
  matchAndRewrite(parsed_csl::IndexAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get inferred result type
    FailureOr<Type> resultType = typeConverter.convertType(op.getResult());
    if (failed(resultType))
      return rewriter.notifyMatchFailure(op,
                                         "failed to get inferred result type");

    // Get the base type from the converted base operand
    Value convertedBase = adaptor.getBase();
    Type baseType = convertedBase.getType();

    // Extract the underlying struct/array type (may be lvalue or rvalue)
    Type baseValueType;
    if (auto baseLvalueType = mlir::dyn_cast<csl::LValueType>(baseType)) {
      baseValueType = baseLvalueType.getValueType();
    } else {
      baseValueType = baseType;
    }

    // Determine which TypedCSL op to use based on base value type
    if (mlir::isa<csl::StructType>(baseValueType)) {
      // Struct/tuple access: requires comptime constant index
      auto constantIndicesAttr = op.getConstantIndicesAttr();
      if (!constantIndicesAttr || constantIndicesAttr.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "struct/tuple index_access requires exactly one compile-time "
                "constant index");

      int64_t index = constantIndicesAttr[0];
      rewriter.replaceOpWithNewOp<typed_csl::TupleIndexAccessOp>(
          op, *resultType, convertedBase, index);
    } else if (mlir::isa<csl::ArrayType>(baseValueType)) {
      // Array access: indices can be runtime values
      rewriter.replaceOpWithNewOp<typed_csl::ArrayIndexAccessOp>(
          op, *resultType, convertedBase, adaptor.getIndices());
    } else if (auto ptrType = mlir::dyn_cast<csl::PointerType>(baseValueType)) {
      // Many-item pointer access: single index, runtime allowed
      if (!ptrType.getIsManyPtr())
        return rewriter.notifyMatchFailure(
            op, "cannot index into single-item pointer");
      if (adaptor.getIndices().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "pointer indexing requires exactly one index");
      rewriter.replaceOpWithNewOp<typed_csl::PtrIndexAccessOp>(
          op, *resultType, convertedBase, adaptor.getIndices()[0]);
    } else {
      return rewriter.notifyMatchFailure(
          op, "index_access base must be struct/tuple, array, or many-item "
              "pointer type");
    }

    return success();
  }
};

/// Erase comptime-only operations (already processed by ComptimeEvalPass)
template <typename OpTy>
struct EraseComptimeOpPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Void Fields
//===----------------------------------------------------------------------===//

/// The struct type that a tuple or field access reads from.
static csl::StructType getAccessedStructType(Value base) {
  Type baseType = base.getType();
  if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(baseType))
    baseType = lvalueType.getValueType();
  return mlir::cast<csl::StructType>(baseType);
}

/// Erase an access to a void field. Comptime evaluation replaced every read
/// of the field by its value and erased every store to it, so only loads
/// without uses can remain.
static LogicalResult eraseVoidFieldAccess(Operation *access,
                                          IRRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(access->getUsers()))
    if (isOpTriviallyDead(user))
      rewriter.eraseOp(user);
  if (!access->use_empty())
    return access->emitError()
           << "internal error: a void field is still used at runtime";
  rewriter.eraseOp(access);
  return success();
}

/// Remove the void fields from the ops that build and access runtime
/// structs: CSL does not allow void fields in runtime values, and they take
/// no memory. The struct types and constants are rewritten afterwards by
/// csl::addVoidFieldStripping.
static LogicalResult removeVoidFields(csl::CompilationUnitOp unit) {
  IRRewriter rewriter(unit.getContext());
  SmallVector<Operation *> ops;
  unit.walk([&](Operation *op) {
    if (mlir::isa<typed_csl::ConstructTupleOp, typed_csl::ConstructStructOp,
                  typed_csl::TupleIndexAccessOp, typed_csl::FieldAccessOp>(op))
      ops.push_back(op);
  });

  for (Operation *op : ops) {
    if (auto access = mlir::dyn_cast<typed_csl::TupleIndexAccessOp>(op)) {
      ArrayRef<Type> fieldTypes =
          getAccessedStructType(access.getBase()).getFieldTypes();
      int64_t index = access.getIndex();
      if (mlir::isa<csl::VoidType>(fieldTypes[index])) {
        if (failed(eraseVoidFieldAccess(access, rewriter)))
          return failure();
        continue;
      }
      int64_t numVoidBefore =
          llvm::count_if(fieldTypes.take_front(index), llvm::IsaPred<csl::VoidType>);
      if (numVoidBefore != 0)
        access.setIndex(index - numVoidBefore);
      continue;
    }

    if (auto access = mlir::dyn_cast<typed_csl::FieldAccessOp>(op)) {
      csl::StructType structType = getAccessedStructType(access.getStructLvalue());
      if (mlir::isa<csl::VoidType>(
              structType.getFieldType(access.getFieldName())) &&
          failed(eraseVoidFieldAccess(access, rewriter)))
        return failure();
      continue;
    }

    auto structType = mlir::cast<csl::StructType>(op->getResult(0).getType());
    if (!llvm::any_of(structType.getFieldTypes(), llvm::IsaPred<csl::VoidType>))
      continue;
    SmallVector<Value> fieldValues;
    SmallVector<Attribute> fieldNames;
    for (auto [index, fieldType] : llvm::enumerate(structType.getFieldTypes())) {
      if (mlir::isa<csl::VoidType>(fieldType))
        continue;
      fieldValues.push_back(op->getOperand(index));
      if (!structType.isTuple())
        fieldNames.push_back(structType.getFieldNames()[index]);
    }
    Type strippedType = csl::stripVoidFields(structType);
    rewriter.setInsertionPoint(op);
    Value replacement;
    if (mlir::cast<csl::StructType>(strippedType).isTuple())
      replacement = typed_csl::ConstructTupleOp::create(
          rewriter, op->getLoc(), strippedType, fieldValues);
    else
      replacement = typed_csl::ConstructStructOp::create(
          rewriter, op->getLoc(), strippedType,
          rewriter.getArrayAttr(fieldNames), fieldValues);
    rewriter.replaceOp(op, replacement);
  }

  // The void values that built the removed fields are no longer used.
  unit.walk([&](typed_csl::ConstantOp constant) {
    if (mlir::isa<csl::VoidAttr>(constant.getValue()) &&
        isOpTriviallyDead(constant))
      rewriter.eraseOp(constant);
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Lower To TypedCSL Pass
//===----------------------------------------------------------------------===//

struct LowerToTypedCSLPass
    : public impl::ConvertParsedCSLToTypedCSLPassBase<LowerToTypedCSLPass> {

  void runOnOperation() override {
    auto unit = getOperation();
    MLIRContext *context = &getContext();

    // Set up type converter
    ParsedCSLTypeConverter typeConverter;

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<typed_csl::TypedCSLDialect>();
    target.addIllegalDialect<parsed_csl::ParsedCSLDialect>();
    target.addLegalOp<csl::CompilationUnitOp>();

    // Set up conversion patterns
    RewritePatternSet patterns(context);

    // Add patterns for ConstantOp with GlobalRefAttr/FuncRefAttr first
    // (they have higher benefit so they're tried before the generic pattern)
    patterns.add<GlobalRefConstantOpConversion, FuncRefConstantOpConversion>(
        typeConverter, context);

    // Add all other conversion patterns
    patterns
        .add<ConstantOrBuiltinTypeOpConversion<parsed_csl::ConstantOp>,
             ConstantOrBuiltinTypeOpConversion<parsed_csl::BuiltinTypeOp>,
             BinaryOpConversion, UnaryOpConversion, CastOpConversion,
             VariableOpConversion, GlobalOpConversion, LoadOpConversion,
             AssignOpConversion, InlineAsmOpConversion, AddressOfOpConversion,
             DereferenceOpConversion, IfOpConversion, ForOpConversion,
             WhileOpConversion, LabeledBlockOpConversion, BreakOpConversion,
             ContinueOpConversion, ComptimeBlockOpConversion,
             LayoutBlockOpConversion, CallOpConversion, BuiltinCallOpConversion,
             FuncOpConversion, ReturnOpConversion, UnreachableOpConversion,
             SuspendOpConversion, AwaitOpConversion, YieldOpConversion,
             SelectOpConversion, ShortCircuitOpConversion, SwitchOpConversion,
             ConstructTupleOpConversion, ConstructStructOpConversion,
             ConstructArrayOpConversion, FieldAccessOpConversion,
             IndexAccessOpConversion>(typeConverter, context);

    // Add patterns to erase comptime-only operations
    // patterns.add<EraseComptimeOpPattern<parsed_csl::BuiltinTypeOp>>(context);

    // Apply conversion
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyFullConversion(unit, target, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }

    if (failed(removeVoidFields(unit))) {
      signalPassFailure();
      return;
    }

    // Strip VoidType results from FunctionTypes throughout the IR.
    // In parsed_csl, void functions have an explicit VoidType result; in
    // typed_csl they have zero results. This must be done as a post-pass
    // because FunctionTypes appear both as SSA value types and nested
    // inside attributes (FuncRefAttr, FuncAddrAttr, etc.).
    AttrTypeReplacer voidStripper;
    voidStripper.addReplacement(
        [](FunctionType funcType) -> std::optional<Type> {
          if (funcType.getNumResults() == 1 &&
              mlir::isa<csl::VoidType>(funcType.getResult(0)))
            return FunctionType::get(funcType.getContext(),
                                     funcType.getInputs(), {});
          return std::nullopt;
        });
    // Strip VoidType from FutureType's result_type parameter.
    voidStripper.addReplacement(
        [](csl::FutureType futureType) -> std::optional<Type> {
          if (futureType.getResultType() &&
              mlir::isa<csl::VoidType>(futureType.getResultType()))
            return csl::FutureType::get(
                futureType.getContext(), futureType.getModuleIdentifier(),
                futureType.getSymbolName(), /*result_type=*/Type());
          return std::nullopt;
        });
    // Strip the void fields from struct types and struct values.
    csl::addVoidFieldStripping(voidStripper);
    voidStripper.recursivelyReplaceElementsIn(unit, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/false,
                                              /*replaceTypes=*/true);
  }
};

} // namespace
