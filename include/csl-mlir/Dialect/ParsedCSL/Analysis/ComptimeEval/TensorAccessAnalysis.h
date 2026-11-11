//===- TensorAccessAnalysis.h - Tensor Access Expression Analysis --------===//
//
// Utilities for analyzing TensorAccessOp expressions and calculating CSL-style
// strides for DSD construction.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_PARSEDCSL_ANALYSIS_TENSORACCESSANALYSIS_H
#define CSL_MLIR_DIALECT_PARSEDCSL_ANALYSIS_TENSORACCESSANALYSIS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace parsed_csl {

class TensorAccessOp;
class ComptimeEvalContext;

//===----------------------------------------------------------------------===//
// TensorAccessInfo
//===----------------------------------------------------------------------===//

/// Result of analyzing a TensorAccessOp for DSD construction.
/// Contains the computed extents, strides, and offset.
struct TensorAccessInfo {
  /// Extent per induction variable.
  SmallVector<uint16_t> extents;

  /// CSL-style strides per induction variable (innermost-first order).
  SmallVector<int16_t> strides;

  /// Linear offset from base address, in 2-byte (16-bit word) units.
  int16_t offset = 0;
};

//===----------------------------------------------------------------------===//
// Analysis Functions
//===----------------------------------------------------------------------===//

/// Build AffineExpr for each array index from TensorAccessOp body.
///
/// The TensorAccessOp body contains operations that compute indices into the
/// target array. This function extracts the affine relationship between the
/// induction variables and the array indices by walking the def-use chain
/// backwards from the yield operands.
///
/// Example: For `|i,j|{10,20} -> array[2*i + j, i - j]`, this would return
/// AffineExprs representing `2*d0 + d1` and `d0 - d1`.
///
/// Returns failure if the index expressions are non-affine (CSL requires
/// affine expressions for tensor access).
///
/// @param op  The TensorAccessOp to analyze
FailureOr<SmallVector<AffineExpr>>
buildAffineIndices(TensorAccessOp op,
                   ComptimeEvalContext *evalContext = nullptr);

/// Calculate CSL-style strides from affine index expressions.
///
/// CSL uses a "reset and accumulate" stride model that differs from standard
/// row-major strides. When incrementing dimension `i`, inner dimensions
/// `j > i` reset from their max to 0, so the effective stride is:
///
///   stride[i] = delta[i] - sum(delta[j] * (extent[j]-1)) for j > i
///
/// Example: `|i,j|{5,5} -> A[2*i + j]` has:
///   - stride[j] = 1 (innermost dimension)
///   - stride[i] = 2 - 1*(5-1) = -2 (not just 2!)
///
/// Output strides are in innermost-first order to match CSL convention.
/// The outOffset is in element units; callers are responsible for converting
/// to hardware units using the element type size.
///
/// @param indexExprs     Affine expressions for each array index dimension
/// @param extents        Extent (iteration count) per induction variable
/// @param arrayShape     Shape of the target array (empty for many-pointer)
/// @param outStrides     Output: computed CSL-style strides (innermost-first)
/// @param outOffset      Output: linear offset in element units
/// @return success() if strides were computed, failure() for non-affine cases
LogicalResult calculateCSLStrides(ArrayRef<AffineExpr> indexExprs,
                                  ArrayRef<uint16_t> extents,
                                  ArrayRef<int64_t> arrayShape,
                                  SmallVectorImpl<int16_t> &outStrides,
                                  int64_t &outOffset);

/// Analyze a TensorAccessOp and extract all information needed for DSD
/// construction.
///
/// This is the main entry point for tensor access analysis. It combines
/// buildAffineIndices() and calculateCSLStrides() to produce a complete
/// TensorAccessInfo with the offset scaled to hardware 2-byte units using
/// the element type size. Returns failure if the offset would be unaligned
/// (not a multiple of 2 bytes).
///
/// @param op          The TensorAccessOp to analyze
/// @param extents     Extent values for each induction variable
/// @param arrayShape  Shape of the underlying array (empty for many-pointer)
/// @param elementType Element type of the array/pointer being accessed
/// @param evalContext Optional comptime evaluation context
/// @return TensorAccessInfo on success, failure() if analysis fails
FailureOr<TensorAccessInfo>
analyzeTensorAccess(TensorAccessOp op, ArrayRef<uint16_t> extents,
                    ArrayRef<int64_t> arrayShape, Type elementType,
                    ComptimeEvalContext *evalContext = nullptr);

} // namespace parsed_csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_PARSEDCSL_ANALYSIS_TENSORACCESSANALYSIS_H
