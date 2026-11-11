//===- TensorAccessAnalysis.cpp - Tensor Access Expression Analysis ------===//
//
// Implementation of tensor access analysis utilities for DSD construction.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/TensorAccessAnalysis.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-access-analysis"

using namespace mlir;
using namespace mlir::parsed_csl;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Helper: Build affine expression from a value (recursive)
//===----------------------------------------------------------------------===//

namespace {

/// Recursively build an AffineExpr from a value by walking the def-use chain.
/// Returns failure if the expression is non-affine.
FailureOr<AffineExpr>
buildAffineExprFromValue(Value value, Block &bodyBlock,
                         llvm::DenseMap<Value, AffineExpr> &cache,
                         ComptimeEvalContext *evalContext) {
  // Check cache first
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  MLIRContext *ctx = value.getContext();

  // Case 2: Block argument (induction variable)
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == &bodyBlock) {
      AffineExpr expr = getAffineDimExpr(blockArg.getArgNumber(), ctx);
      cache[value] = expr;
      return expr;
    }
    // Block arg from elsewhere - not supported
    LLVM_DEBUG(llvm::dbgs() << "Block arg from outside TAE body\n");
    return failure();
  }

  // Must be an OpResult - get the defining operation
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    LLVM_DEBUG(llvm::dbgs() << "Value has no defining op\n");
    return failure();
  }

  // Case: LocOp, the location wrapper used in LSP mode, or ReadOp, the read
  // of an induction variable. Pass through to the wrapped value so that
  // induction variable block args are correctly recognized as affine dims
  // rather than constants.
  if (mlir::isa<LocOp, ReadOp>(defOp)) {
    auto exprResult = buildAffineExprFromValue(defOp->getOperand(0), bodyBlock,
                                               cache, evalContext);
    if (succeeded(exprResult))
      cache[value] = *exprResult;
    return exprResult;
  }

  // Case 3: Binary operation
  if (auto binOp = mlir::dyn_cast<BinaryOp>(defOp)) {
    auto lhsResult =
        buildAffineExprFromValue(binOp.getLhs(), bodyBlock, cache, evalContext);
    if (failed(lhsResult))
      return failure();

    auto rhsResult =
        buildAffineExprFromValue(binOp.getRhs(), bodyBlock, cache, evalContext);
    if (failed(rhsResult))
      return failure();

    AffineExpr lhs = *lhsResult;
    AffineExpr rhs = *rhsResult;
    AffineExpr result;

    switch (binOp.getKind()) {
    case BinaryOpKind::Add:
      result = lhs + rhs;
      break;
    case BinaryOpKind::Sub:
      result = lhs - rhs;
      break;
    case BinaryOpKind::Mul:
      // For affine, at least one operand must be constant
      if (!lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant()) {
        LLVM_DEBUG(llvm::dbgs() << "Non-affine multiplication\n");
        return failure();
      }
      result = lhs * rhs;
      break;
    default:
      LLVM_DEBUG(llvm::dbgs() << "Non-affine binary op: "
                              << static_cast<int>(binOp.getKind()) << "\n");
      return failure();
    }

    cache[value] = result;
    return result;
  }

  // Case 3: Constant operation
  if (evalContext) {
    // Query the eval context for constant values
    FailureOr<TypedAttr> constValueResult =
        evalContext->getOrLoadComptimeRValue(value);
    if (succeeded(constValueResult)) {
      TypedAttr constValue = *constValueResult;
      InFlightDiagnostic diag;
      FailureOr<TypedAttr> castedValue = csl::evaluateImplicitCast(
          constValue, ComptimeIntType::get(ctx), &diag);
      if (failed(castedValue)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Non-integer constant: " << constValue << "\n");
        diag.report();
        return failure();
      }
      auto comptimeInt = mlir::cast<ComptimeIntAttr>(*castedValue);
      int64_t intVal = comptimeInt.getValue().getSExtValue();
      AffineExpr expr = getAffineConstantExpr(intVal, ctx);
      cache[value] = expr;
      return expr;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Value is not a comptime known constant: " << value << "\n");
  } else {
    // Assume that comptime known values have been folded to ConstantOps by the
    // comptime-eval pass before
    if (auto constOp = mlir::dyn_cast<ConstantOp>(defOp)) {
      TypedAttr constValue = constOp.getValue();

      // Try to get an integer value
      Type comptimeIntType = ComptimeIntType::get(ctx);
      auto castResult = csl::evaluateCast(constValue, comptimeIntType);
      if (failed(castResult)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Non-integer constant: " << constValue << "\n");
        return failure();
      }

      auto comptimeInt = mlir::cast<ComptimeIntAttr>(*castResult);
      int64_t intVal = comptimeInt.getValue().getSExtValue();
      AffineExpr expr = getAffineConstantExpr(intVal, ctx);
      cache[value] = expr;
      return expr;
    }
  }

  // Unknown operation
  LLVM_DEBUG(llvm::dbgs() << "Unknown op in TAE body: " << *defOp << "\n");
  return failure();
}

/// Calculate CSL strides from deltas using the reset-accumulate formula.
/// Strides are computed innermost-first and stored in that order.
void computeCSLStridesFromDeltas(ArrayRef<int64_t> deltas,
                                 ArrayRef<uint16_t> extents,
                                 SmallVectorImpl<int16_t> &outStrides) {
  unsigned numDims = deltas.size();
  outStrides.clear();
  outStrides.resize(numDims, 0);

  // Process from outermost (first) to innermost (last).
  // The CSL hardware stride for dimension n is the address delta when all
  // inner dimensions reset (wrap from max back to 0) and dimension n steps:
  //   stride[n] = delta[n] - sum_{m>n} delta[m] * (extent[m] - 1)
  // This uses the raw deltas (not the previously computed hardware strides)
  // because the inner dimensions contribute their full traversal range.
  for (int i = 0; i < static_cast<int>(numDims); ++i) {
    int64_t stride = deltas[i];
    for (unsigned j = i + 1; j < numDims; ++j) {
      stride -= deltas[j] * (static_cast<int64_t>(extents[j]) - 1);
    }
    outStrides[i] = static_cast<int16_t>(stride);
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// buildAffineIndices - Public API
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<AffineExpr>>
mlir::parsed_csl::buildAffineIndices(TensorAccessOp op,
                                     ComptimeEvalContext *evalContext) {
  Block &body = op.getBody().front();
  auto yieldOp = mlir::cast<TensorAccessYieldOp>(body.getTerminator());

  // Cache for memoization
  llvm::DenseMap<Value, AffineExpr> cache;

  // Build affine expression for each yielded index
  SmallVector<AffineExpr> result;
  result.reserve(yieldOp.getIndices().size());

  for (Value idx : yieldOp.getIndices()) {
    auto exprResult = buildAffineExprFromValue(idx, body, cache, evalContext);
    if (failed(exprResult))
      return failure();
    result.push_back(*exprResult);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Built affine indices for TAE with "
                 << op.getNumDimensions() << " dimensions:\n";
    for (auto [i, expr] : llvm::enumerate(result)) {
      llvm::dbgs() << "  index[" << i << "]: " << expr << "\n";
    }
  });

  return result;
}

//===----------------------------------------------------------------------===//
// calculateCSLStrides - Public API
//===----------------------------------------------------------------------===//

LogicalResult mlir::parsed_csl::calculateCSLStrides(
    ArrayRef<AffineExpr> indexExprs, ArrayRef<uint16_t> extents,
    ArrayRef<int64_t> arrayShape, SmallVectorImpl<int16_t> &outStrides,
    int64_t &outOffset) {
  unsigned numDims = extents.size();
  unsigned numArrayDims = indexExprs.size();

  if (indexExprs.empty())
    return failure();

  MLIRContext *ctx = indexExprs.front().getContext();

  // Compute row-major array strides from shape
  // For shape [s0, s1, s2], strides are [s1*s2, s2, 1]
  // For many-pointer (empty shape), treat as 1D with stride 1
  SmallVector<int64_t> arrayStrides;
  if (!arrayShape.empty()) {
    arrayStrides = computeStrides(arrayShape);
  } else {
    arrayStrides.assign(numArrayDims, 1);
  }

  // Extract coefficients from affine expressions by evaluating at specific
  // points. For affine expr a*d0 + b*d1 + c, coeff[i] = expr(di=1, others=0) -
  // expr(all=0)
  SmallVector<int64_t> deltas(numDims, 0);
  int64_t constantOffset = 0;

  for (unsigned arrayDim = 0; arrayDim < numArrayDims; ++arrayDim) {
    AffineExpr indexExpr = indexExprs[arrayDim];
    int64_t arrayStride = arrayStrides[arrayDim];

    // Evaluate at all zeros to get the constant base
    SmallVector<AffineExpr> zeroExprs(numDims, getAffineConstantExpr(0, ctx));
    AffineExpr baseExpr = indexExpr.replaceDims(zeroExprs);

    // Get coefficient for each induction variable
    for (unsigned d = 0; d < numDims; ++d) {
      SmallVector<AffineExpr> oneAtD(numDims, getAffineConstantExpr(0, ctx));
      oneAtD[d] = getAffineConstantExpr(1, ctx);
      AffineExpr exprAtOne = indexExpr.replaceDims(oneAtD);

      // coeff = expr(di=1) - expr(all=0)
      AffineExpr coeffExpr = exprAtOne - baseExpr;
      if (auto constExpr = mlir::dyn_cast<AffineConstantExpr>(coeffExpr)) {
        deltas[d] += constExpr.getValue() * arrayStride;
      }
    }

    // Accumulate constant term
    if (auto constBase = mlir::dyn_cast<AffineConstantExpr>(baseExpr)) {
      constantOffset += constBase.getValue() * arrayStride;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Computed deltas: [";
    for (size_t i = 0; i < deltas.size(); ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << deltas[i];
    }
    llvm::dbgs() << "], constantOffset: " << constantOffset << "\n";
  });

  // Apply CSL reset-accumulate stride formula
  computeCSLStridesFromDeltas(deltas, extents, outStrides);
  outOffset = constantOffset;

  LLVM_DEBUG({
    llvm::dbgs() << "CSL strides: [";
    for (size_t i = 0; i < outStrides.size(); ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << outStrides[i];
    }
    llvm::dbgs() << "], offset: " << outOffset << "\n";
  });

  return success();
}

//===----------------------------------------------------------------------===//
// analyzeTensorAccess - Main entry point
//===----------------------------------------------------------------------===//

FailureOr<TensorAccessInfo> mlir::parsed_csl::analyzeTensorAccess(
    TensorAccessOp op, ArrayRef<uint16_t> extents, ArrayRef<int64_t> arrayShape,
    Type elementType, ComptimeEvalContext *evalContext) {
  TensorAccessInfo info;
  info.extents.assign(extents.begin(), extents.end());

  // 1. Build affine expressions from the body
  auto affineResult = buildAffineIndices(op, evalContext);
  if (failed(affineResult))
    return failure();

  // 2. Calculate CSL strides and raw element-unit offset (outermost-first)
  SmallVector<int16_t> stridesOutermostFirst;
  int64_t elementOffset = 0;
  if (failed(calculateCSLStrides(*affineResult, extents, arrayShape,
                                 stridesOutermostFirst, elementOffset)))
    return failure();

  // 3. Reverse to innermost-first order for CSL convention
  info.strides.assign(stridesOutermostFirst.rbegin(),
                      stridesOutermostFirst.rend());

  // 4. Scale offset from element units to 2-byte hardware units
  if (elementOffset != 0) {
    auto elemSize = csl::getSizeInBytes(elementType);
    if (!elemSize)
      return op.emitError(
          "element type has unknown size for DSD offset calculation");
    int64_t byteOffset = elementOffset * static_cast<int64_t>(*elemSize);
    if (byteOffset % 2 != 0)
      return op.emitError(
          "tensor_access offset is not aligned to 2-byte boundary");
    info.offset = static_cast<int16_t>(byteOffset / 2);
  }

  return info;
}
