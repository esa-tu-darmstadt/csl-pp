//===- LowerTensorAccessPass.cpp - Lower Runtime TAEs -----------*- C++ -*-===//
//
// This file implements a pass that lowers runtime TensorAccessOps to
// ConstructStructOps. Comptime TAEs are handled by ComptimeEvalPass - this
// pass handles the remaining runtime cases.
//
// The pass must run after ComptimeEvalPass and before NormalizeCastsPass.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/TensorAccessAnalysis.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-tensor-access"

using namespace mlir;
using namespace mlir::parsed_csl;
using namespace mlir::csl;

namespace mlir {
namespace parsed_csl {

#define GEN_PASS_DEF_LOWERTENSORACCESSPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

} // namespace parsed_csl
} // namespace mlir

namespace {

/// Get the array shape from the base type.
/// Returns empty vector for many-pointers (unknown shape at compile time).
static SmallVector<int64_t> getArrayShapeFromType(Type baseType) {
  // Strip lvalue wrapper if present
  if (auto lvalueType = mlir::dyn_cast<LValueType>(baseType))
    baseType = lvalueType.getValueType();

  // Extract shape from array type
  if (auto arrayType = mlir::dyn_cast<ArrayType>(baseType))
    return SmallVector<int64_t>(arrayType.getShape());

  // Extract shape from pointer-to-array (e.g. *[4, 5, 10]f32)
  if (auto ptrType = mlir::dyn_cast<PointerType>(baseType))
    if (auto arrayType = mlir::dyn_cast<ArrayType>(ptrType.getPointeeType()))
      return SmallVector<int64_t>(arrayType.getShape());

  // Many-pointer has unknown shape
  return {};
}

/// Lower a single TensorAccessOp to a ConstructStructOp.
static LogicalResult lowerTensorAccessOp(TensorAccessOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *ctx = op->getContext();
  unsigned numDims = op.getNumDimensions();

  LLVM_DEBUG(llvm::dbgs() << "Lowering TensorAccessOp: " << op << "\n");

  // 1. Get extent values - these are comptime-known per validation
  SmallVector<uint16_t> extentValues;
  Type comptimeIntType = ComptimeIntType::get(ctx);

  for (Value extent : op.getExtents()) {
    auto constOp = extent.getDefiningOp<ConstantOp>();
    if (!constOp)
      return op.emitError("tensor_access extent must be comptime-known");

    auto castResult = csl::evaluateCast(constOp.getValue(), comptimeIntType);
    if (failed(castResult))
      return op.emitError("tensor_access extent must be an integer");

    auto comptimeInt = mlir::cast<ComptimeIntAttr>(*castResult);
    extentValues.push_back(
        static_cast<uint16_t>(comptimeInt.getValue().getZExtValue()));
  }

  // 2. Get array shape and element type from base type
  FailureOr<Type> baseTypeResult = getInferredType(op.getBase());
  SmallVector<int64_t> arrayShape;
  Type elementType;
  if (succeeded(baseTypeResult)) {
    arrayShape = getArrayShapeFromType(*baseTypeResult);
    Type baseValType = *baseTypeResult;
    if (auto lvalueType = mlir::dyn_cast<LValueType>(baseValType))
      baseValType = lvalueType.getValueType();
    if (auto arrayType = mlir::dyn_cast<ArrayType>(baseValType)) {
      elementType = arrayType.getElementType();
    } else if (auto pointerType = mlir::dyn_cast<PointerType>(baseValType)) {
      Type pointeeType = pointerType.getPointeeType();
      if (auto arrType = mlir::dyn_cast<ArrayType>(pointeeType))
        elementType = arrType.getElementType();
      else
        elementType = pointeeType;
    }
  }

  // 3. Analyze tensor access to get strides and offset
  auto analysisResult =
      analyzeTensorAccess(op, extentValues, arrayShape, elementType);
  if (failed(analysisResult)) {
    return op.emitError("non-affine index expression in tensor_access");
  }
  TensorAccessInfo &info = *analysisResult;

  LLVM_DEBUG({
    llvm::dbgs() << "  Computed strides: [";
    for (size_t i = 0; i < info.strides.size(); ++i) {
      if (i > 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << info.strides[i];
    }
    llvm::dbgs() << "], offset: " << info.offset << "\n";
  });

  // Get the UnresolvedType for result types
  Type unresolvedType = UnresolvedType::get(ctx);

  // 5. Create the struct fields with proper inferred types

  // base_address: either AddressOf(base) for arrays, or use base directly for
  // pointers Strip lvalue to get the underlying type
  Type baseValueType = *baseTypeResult;
  if (auto lvalueType = mlir::dyn_cast<LValueType>(baseValueType))
    baseValueType = lvalueType.getValueType();

  Value baseAddr;
  Type ptrType;
  if (auto arrayTy = mlir::dyn_cast<ArrayType>(baseValueType)) {
    // Base is array - take address to get pointer to array
    ptrType = PointerType::get(arrayTy, /*isConst=*/false, /*isArray=*/false);
    baseAddr = AddressOfOp::create(builder, loc, unresolvedType, op.getBase(),
                                   mlir::TypeAttr::get(ptrType));
  } else if (mlir::isa<PointerType>(baseValueType)) {
    // Base is already a many-pointer: a pointer variable is loaded here,
    // where the tensor access is evaluated.
    ptrType = baseValueType;
    baseAddr = op.getBase();
    if (mlir::isa<LValueType>(*baseTypeResult))
      baseAddr = LoadOp::create(builder, loc, unresolvedType, baseAddr,
                                mlir::TypeAttr::get(ptrType));
  } else {
    return op.emitError("tensor_access base must be array or many-pointer");
  }

  // offset: i16 constant
  Type i16Type = IntegerType::get(ctx, 16, IntegerType::Signed);
  Value offsetVal =
      ConstantOp::create(builder, loc, IntegerAttr::get(i16Type, info.offset));

  // stride: i8 (1D) or tuple (multi-D)
  Value strideVal;
  Type i8Type = IntegerType::get(ctx, 8, IntegerType::Signed);
  if (numDims == 1) {
    strideVal = ConstantOp::create(builder, loc,
                                   IntegerAttr::get(i8Type, info.strides[0]));
  } else {
    // Multi-dimensional: create a tuple of strides (anonymous struct)
    // Note: strides are already in innermost-first order from
    // analyzeTensorAccess
    SmallVector<Value> strideValues;
    SmallVector<Type> strideTypes;
    for (int16_t s : info.strides) {
      strideValues.push_back(
          ConstantOp::create(builder, loc, IntegerAttr::get(i16Type, s)));
      strideTypes.push_back(i16Type);
    }
    Type strideTupleType = StructType::get(ctx, strideTypes);
    strideVal =
        ConstructTupleOp::create(builder, loc, unresolvedType, strideValues,
                                 mlir::TypeAttr::get(strideTupleType));
  }

  // extent: u16 (1D) or tuple (multi-D)
  Value extentVal;
  Type u16Type = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  if (numDims == 1) {
    // Use the original extent operand directly
    extentVal = op.getExtents()[0];
  } else {
    // Multi-dimensional: create a tuple of extents (anonymous struct)
    SmallVector<Value> extentVals(op.getExtents().begin(),
                                  op.getExtents().end());
    SmallVector<Type> extentTypes(numDims, u16Type);
    Type extentTupleType = StructType::get(ctx, extentTypes);
    extentVal =
        ConstructTupleOp::create(builder, loc, unresolvedType, extentVals,
                                 mlir::TypeAttr::get(extentTupleType));
  }

  // 6. Create the ConstructStructOp with all fields
  SmallVector<Attribute> fieldNameAttrs;
  fieldNameAttrs.push_back(builder.getStringAttr("base_address"));
  fieldNameAttrs.push_back(builder.getStringAttr("offset"));
  fieldNameAttrs.push_back(builder.getStringAttr("stride"));
  fieldNameAttrs.push_back(builder.getStringAttr("extent"));
  ArrayAttr fieldNamesAttr = builder.getArrayAttr(fieldNameAttrs);

  SmallVector<Value> fieldValues = {baseAddr, offsetVal, strideVal, extentVal};

  auto structOp =
      ConstructStructOp::create(builder, loc, unresolvedType, fieldNamesAttr,
                                /*struct_type=*/mlir::Value(), fieldValues,
                                /*inferred_type=*/op.getInferredTypeAttr());

  // Replace all uses and erase the original op
  op.replaceAllUsesWith(structOp.getResult());
  op.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerTensorAccessPass
    : public parsed_csl::impl::LowerTensorAccessPassBase<
          LowerTensorAccessPass> {

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();

    // Collect all TensorAccessOps first to avoid iterator invalidation
    SmallVector<TensorAccessOp> opsToLower;

    unit->walk([&](TensorAccessOp op) {
      opsToLower.push_back(op);
      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "Found " << opsToLower.size()
                            << " runtime TensorAccessOps to lower\n");

    // Lower each runtime TensorAccessOp
    for (TensorAccessOp op : opsToLower) {
      if (failed(lowerTensorAccessOp(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace parsed_csl {

std::unique_ptr<Pass> createLowerTensorAccessPass() {
  return std::make_unique<LowerTensorAccessPass>();
}

} // namespace parsed_csl
} // namespace mlir
