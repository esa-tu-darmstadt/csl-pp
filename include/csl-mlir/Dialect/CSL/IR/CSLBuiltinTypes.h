//===- CSLBuiltinTypes.h - Builtin Type Registry ----------------*- C++ -*-===//
//
// Single source of truth for CSL builtin type names (i32, f16, bool, etc.).
// Used by the parser, pretty printer, and LSP completion.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINTYPES_H
#define CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;

namespace csl {

/// Look up the CSL name for a builtin type (e.g., si32 → "i32").
/// Returns failure for non-builtin types (arrays, structs, pointers, etc.).
mlir::FailureOr<llvm::StringRef> getBuiltinTypeName(mlir::Type type);

/// Look up the MLIR Type for a CSL builtin type name (e.g., "i32" → si32).
/// Returns failure if the name is not a known builtin type.
mlir::FailureOr<mlir::Type> getBuiltinType(llvm::StringRef name,
                                           mlir::MLIRContext *ctx);

/// Check if \p name is a known builtin type name.
bool isBuiltinTypeName(llvm::StringRef name);

/// Collect all builtin type names into \p names.
void getAllBuiltinTypeNames(llvm::SmallVectorImpl<llvm::StringRef> &names);

} // namespace csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINTYPES_H
