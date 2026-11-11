//===- ParsedCSLTypes.td - ParsedCSL Type Definitions -----*- tablegen -*-===//
//
// Type definitions for the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_TYPES
#define PARSED_CSL_TYPES

include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLBase.td"
include "csl-mlir/Dialect/CSL/IR/CSLTypes.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// ParsedCSL Type Definitions
//===----------------------------------------------------------------------===//

class ParsedCSL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ParsedCSL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Unresolved Type
//===----------------------------------------------------------------------===//

def ParsedCSL_UnresolvedType : ParsedCSL_Type<"Unresolved", "unresolved">,
    BuildableType<"::mlir::parsed_csl::UnresolvedType::get($_builder.getContext())"> {
  let summary = "Unresolved type placeholder";
  let description = [{
    Represents a value whose actual type is not yet resolved. This is THE type
    used for all value SSA results in ParsedCSL before type resolution.

    The actual type is tracked separately by the ComptimeEvaluator and inferred
    through the TypeInferenceOpInterface. This allows type information to flow
    through the analysis without modifying the IR's static types.

    After conversion to TypedCSL, all unresolved types are replaced with
    concrete types (i32, f32, ptr<i32>, etc.).

    Example:
    ```mlir
    %x = parsed_csl.constant 42 : ui32 // result type is !parsed_csl.unresolved
    %y = parsed_csl.binary add, %x, %x // result type is !parsed_csl.unresolved
    ```

    In the example above, both %x and %y have static type !parsed_csl.unresolved,
    but the analysis tracks that they actually represent i32 values.
  }];
}


//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//


#endif // PARSED_CSL_TYPES
