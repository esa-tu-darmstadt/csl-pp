#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/IR/MLIRContext.h"

/// Verify that installed dialect headers and their generated definitions link.
int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::csl::CSLDialect>();
  context.getOrLoadDialect<mlir::parsed_csl::ParsedCSLDialect>();
  context.getOrLoadDialect<mlir::typed_csl::TypedCSLDialect>();
  return 0;
}
