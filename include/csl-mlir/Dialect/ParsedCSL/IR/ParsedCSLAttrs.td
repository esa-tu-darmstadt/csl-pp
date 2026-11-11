//===- ParsedCSLAttrs.td - ParsedCSL Attribute Definitions -*- tablegen -*-===//
//
// Attribute definitions for the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_ATTRS
#define PARSED_CSL_ATTRS

include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLBase.td"
include "csl-mlir/Dialect/CSL/IR/CSLAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// ParsedCSL Attribute Base Class
//===----------------------------------------------------------------------===//

class ParsedCSL_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ParsedCSL_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def ParsedCSL_ModuleRefAttr : ParsedCSL_Attr<"ModuleRef", "module_ref", [TypedAttrInterface]> {
  let summary = "Reference to an imported module";
  let description = [{
    Represents a reference to an imported module with a given identifier.

    Example:
    ```mlir
    #csl.module_ref<"my_module">
    ```
  }];

  let parameters = (ins "uint64_t":$module_identifier);
  let assemblyFormat = "`<` $module_identifier `>`";

  let extraClassDeclaration = [{
    csl::ImportedModuleType getType() {
      return csl::ImportedModuleType::get(getContext());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Note: All ParsedCSL attributes have been moved to the base CSL dialect
// Operations should use the CSL_ prefixed attributes directly
// (e.g., CSL_ComptimeIntAttr, CSL_TypeAttr, CSL_BoolAttr, etc.)
//===----------------------------------------------------------------------===//

#endif // PARSED_CSL_ATTRS
