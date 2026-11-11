
//===- TypedCSLTypes.td - TypedCSL Type Definitions -------*- tablegen -*-===//
//
// Type definitions for the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_CSL_TYPES
#define TYPED_CSL_TYPES

include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLBase.td"
include "csl-mlir/Dialect/CSL/IR/CSLTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Note: All TypedCSL types have been moved to the base CSL dialect
// Operations should use the CSL_ prefixed types directly (e.g., CSL_IntType)
//===----------------------------------------------------------------------===//

class TypedCSL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TypedCSL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// LValue type constraint for TypedCSL (wraps concrete CSL types)
// Note: We don't need to explicitly check against UnresolvedType here.
// TypedCSL IR should only contain concrete types from the CSL dialect.
def TypedCSL_LValueType : Type<
  CPred<"::llvm::isa<::mlir::csl::LValueType>($_self)">,
  "lvalue type", "::mlir::csl::LValueType">;

// Type constraint for mutable lvalues (used by AssignOp)
def TypedCSL_MutableLValueType : Type<
  CPred<"::llvm::isa<::mlir::csl::LValueType>($_self) && "
        "!::llvm::cast<::mlir::csl::LValueType>($_self).getIsConst()">,
  "mutable lvalue type">;

#endif // TYPED_CSL_TYPES
