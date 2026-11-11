//===- CSLTypeInterfaces.td - CSL Type Interfaces ---------*- tablegen -*-===//
//
// Type interface definitions for the CSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_TYPE_INTERFACES
#define CSL_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TypeInfoInterface
//===----------------------------------------------------------------------===//

def CSL_TypeInfoInterface : TypeInterface<"TypeInfoInterface"> {
  let cppNamespace = "::mlir::csl";

  let description = [{
    Interface providing type-level information for CSL types.

    This interface allows CSL-defined types to expose information used by
    analyses and lowering passes:
    - The size of the type in bytes (for offset and stride calculations in
      DSD construction).
    - Whether the type can exist as a runtime value on the hardware.

    Note: MLIR builtin types (IntegerType, FloatType) used in CSL are handled
    directly by the free functions `csl::getSizeInBytes` and
    `csl::isRuntimeType` in CSL.h; they are not required to implement this
    interface.
  }];

  let methods = [
    InterfaceMethod<[{
      Returns the size of this type in bytes, or std::nullopt if the size is
      not statically known or not applicable.

      Not applicable cases include: DSD types, DSR types, hardware types with
      opaque layout, lvalue references, and void.

      For composite types (arrays, structs), this recursively sums the element
      sizes assuming packed layout (no alignment padding).
    }],
    "::std::optional<uint32_t>", "getSizeInBytes", (ins),
    /*methodBody=*/[{}],
    /*defaultImpl=*/[{
      return ::std::nullopt;
    }]>,

    InterfaceMethod<[{
      Returns whether this type is a runtime type — a type that can exist as
      a value on the hardware at runtime.

      Returns false for comptime-only types (comptime_int, comptime_float,
      comptime_string, type, etc.) and for meta-types (lvalue, void).

      Returns true for scalar numeric types, pointers, structs, arrays, enums,
      and hardware-visible types.
    }],
    "bool", "isRuntimeType", (ins),
    /*methodBody=*/[{}],
    /*defaultImpl=*/[{
      return true;
    }]>,

    InterfaceMethod<[{
      Returns whether this type contains per-PE layout information.

      Returns true for PerPEArrayType and for composite types (arrays, structs)
      that recursively contain per-PE arrays. Returns false for all other types.
    }],
    "bool", "hasPerPELayout", (ins),
    /*methodBody=*/[{}],
    /*defaultImpl=*/[{
      return false;
    }]>,

    InterfaceMethod<[{
      Returns the size of this type in bytes for a specific PE at
      coordinate (x, y), or std::nullopt if the size is not applicable.

      For types without per-PE layout, this delegates to getSizeInBytes().
      For PerPEArrayType, this computes the size based on the per-PE
      dimensions at the given coordinate.
    }],
    "::std::optional<uint32_t>", "getSizeInBytesForPE",
    (ins "int64_t":$x, "int64_t":$y),
    /*methodBody=*/[{}],
    /*defaultImpl=*/[{
      return $_type.getSizeInBytes();
    }]>,
  ];
}

#endif // CSL_TYPE_INTERFACES
