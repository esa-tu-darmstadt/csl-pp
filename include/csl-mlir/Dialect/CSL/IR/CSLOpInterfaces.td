//===- CSLOpInterfaces.td - CSL Op Interfaces -----------*- tablegen -*-===//
//
// Op interface definitions for the CSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_OP_INTERFACES
#define CSL_OP_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LabeledOpInterface
//===----------------------------------------------------------------------===//

def CSL_LabeledOpInterface : OpInterface<"LabeledOpInterface"> {
  let cppNamespace = "::mlir::csl";

  let description = [{
    Interface for operations that carry an optional label (for, while,
    labeled_block). Used by break/continue to locate their target across
    dialect boundaries (e.g., during lowering from ParsedCSL to TypedCSL,
    the break may already be lowered while the enclosing loop is not).
  }];

  let methods = [
    InterfaceMethod<
      "Return the label of this operation, or std::nullopt if unlabeled.",
      "std::optional<llvm::StringRef>", "getOptionalLabel", (ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        return $_op.getLabel();
      }]
    >
  ];
}

#endif // CSL_OP_INTERFACES
