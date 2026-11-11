//===- CSLOps.cpp - CSL Operations -------------------------------------===//
//
// Implementation of CSL operations.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::csl;

void CompilationUnitOp::build(OpBuilder &builder, OperationState &state,
                              std::optional<StringRef> name) {
  state.addRegion();
  if (name)
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(*name));
}

//===----------------------------------------------------------------------===//
// TableGen Operation Definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "csl-mlir/Dialect/CSL/IR/CSLOps.cpp.inc"
