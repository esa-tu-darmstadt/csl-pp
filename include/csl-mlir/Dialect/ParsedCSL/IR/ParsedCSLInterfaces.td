//===- ParsedCSLInterfaces.td - ParsedCSL Interfaces ------*- tablegen -*-===//
//
// Interface definitions for the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_INTERFACES
#define PARSED_CSL_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// InferredTypeInterface
//===----------------------------------------------------------------------===//

def InferredTypeInterface : OpInterface<"InferredTypeInterface"> {
  let cppNamespace = "::mlir::parsed_csl";

  let description = [{
    Interface for operations that can store their inferred result type.

    This interface provides support for caching inferred types as persistent
    IR attributes. Types are computed during compile-time analysis and stored
    to survive IR transformations.

    Operations implementing this interface should have the following ODS argument:

      - `inferred_type` (optional)
        * A TypeAttr that holds the inferred result type of the operation.
        * This is populated by the ComptimeEvalPass after type inference.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the inferred type if available.

      Returns std::nullopt if no inferred type has been computed or stored yet.
    }],
    "::std::optional<::mlir::Type>", "getInferredType", (ins)
    >,

    InterfaceMethod<[{
      Set the inferred type.

      This method should be called by analysis passes after computing the
      inferred type of the operation. Pass std::nullopt to clear the type.
    }],
    "void", "setInferredType", (ins "::std::optional<::mlir::Type>":$attrValue)
    >,

    InterfaceMethod<[{
      Clear the inferred type.

      This removes the stored inferred type, if present.
    }],
    "void", "clearInferredType", (ins),
    /*methodBody=*/[{}], /*defaultImpl=*/[{
      $_op.setInferredType(::std::nullopt);
    }]>
  ];
}

#endif // PARSED_CSL_INTERFACES