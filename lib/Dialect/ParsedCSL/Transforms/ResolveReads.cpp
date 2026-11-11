//===- ResolveReads.cpp - Resolve reads to loads ----------------*- C++ -*-===//
//
// The parser emits a ReadOp wherever the source uses an expression as a
// value, at the position where the source reads it. Once comptime evaluation
// has inferred the types, this pass resolves each read: a read of a location
// becomes a LoadOp at the same position, and a read of anything else is
// replaced by the value it reads.
//
// The base of an index access is the one place where a load depends on the
// type: an array variable is indexed in place, but a pointer variable is
// loaded and the pointer indexed. That load happens at the access, after the
// indices are evaluated.
//
// Afterwards, every location that is used as a value is loaded explicitly.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::parsed_csl;

namespace mlir {
namespace parsed_csl {

#define GEN_PASS_DEF_RESOLVEREADSPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"

} // namespace parsed_csl
} // namespace mlir

namespace {

/// Replace `readOp` by a load if it reads a location, or by the value it
/// reads otherwise.
LogicalResult resolveRead(ReadOp readOp, OpBuilder &builder) {
  Value source = readOp.getSource();
  FailureOr<Type> sourceType = getInferredType(source);
  if (failed(sourceType))
    return readOp.emitError()
           << "type of the read value is unknown (internal error)";

  auto locationType = mlir::dyn_cast<csl::LValueType>(*sourceType);
  if (!locationType) {
    readOp.getResult().replaceAllUsesWith(source);
    readOp.erase();
    return success();
  }

  builder.setInsertionPoint(readOp);
  auto loadOp = LoadOp::create(
      builder, readOp.getLoc(), UnresolvedType::get(readOp.getContext()),
      source, mlir::TypeAttr::get(locationType.getValueType()));
  readOp.getResult().replaceAllUsesWith(loadOp.getResult());
  readOp.erase();
  return success();
}

/// Load the base of `indexOp` if it is a pointer variable.
void loadPointerBase(IndexAccessOp indexOp, OpBuilder &builder) {
  Value base = indexOp.getBase();
  FailureOr<Type> baseType = getInferredType(base);
  if (failed(baseType))
    return;
  auto locationType = mlir::dyn_cast<csl::LValueType>(*baseType);
  if (!locationType ||
      !mlir::isa<csl::PointerType>(locationType.getValueType()))
    return;
  builder.setInsertionPoint(indexOp);
  auto loadOp = LoadOp::create(
      builder, indexOp.getLoc(), UnresolvedType::get(indexOp.getContext()),
      base, mlir::TypeAttr::get(locationType.getValueType()));
  indexOp.getBaseMutable().assign(loadOp.getResult());
}

struct ResolveReadsPass
    : public parsed_csl::impl::ResolveReadsPassBase<ResolveReadsPass> {

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    OpBuilder builder(&getContext());

    SmallVector<Operation *> ops;
    unit.walk([&](Operation *op) { ops.push_back(op); });
    for (Operation *op : ops) {
      if (auto readOp = mlir::dyn_cast<ReadOp>(op)) {
        if (failed(resolveRead(readOp, builder)))
          return signalPassFailure();
      } else if (auto indexOp = mlir::dyn_cast<IndexAccessOp>(op)) {
        loadPointerBase(indexOp, builder);
      }
    }

    // Every location that is used as a value is loaded now.
    WalkResult result = unit.walk([](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        FailureOr<Type> type = getInferredType(operand.get());
        if (failed(type) || !mlir::isa<csl::LValueType>(*type) ||
            isUnreadOperand(op, operand.getOperandNumber()))
          continue;
        op->emitError() << "operand " << operand.getOperandNumber()
                        << " is a location that is used as a value without "
                           "a load (internal error)";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace parsed_csl {

std::unique_ptr<Pass> createResolveReadsPass() {
  return std::make_unique<ResolveReadsPass>();
}

} // namespace parsed_csl
} // namespace mlir
