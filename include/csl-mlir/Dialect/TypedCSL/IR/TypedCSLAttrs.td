//===- TypedCSLAttrs.td - TypedCSL Attribute Definitions --*- tablegen -*-===//
//
// Attribute definitions for the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_CSL_ATTRS
#define TYPED_CSL_ATTRS

include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLBase.td"
include "csl-mlir/Dialect/CSL/IR/CSLAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Note: All TypedCSL attributes have been moved to the base CSL dialect
// Operations should use the CSL_ prefixed attributes directly
// (e.g., CSL_IntAttr, CSL_BoolAttr, CSL_FloatKind, etc.)
//===----------------------------------------------------------------------===//

class TypedCSL_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TypedCSL_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Note: All TypedCSL attributes have been moved to the base CSL dialect
// Operations should use the CSL_ prefixed attributes directly
// (e.g., CSL_ComptimeIntAttr, CSL_TypeAttr, CSL_BoolAttr, etc.)
//===----------------------------------------------------------------------===//

#endif // TYPED_CSL_ATTRS
