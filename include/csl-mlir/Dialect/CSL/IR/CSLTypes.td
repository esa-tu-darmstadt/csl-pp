//===- CSLTypes.td - CSL Type Definitions -----------------*- tablegen -*-===//
//
// Type definitions for the CSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_TYPES
#define CSL_TYPES

include "csl-mlir/Dialect/CSL/IR/CSLBase.td"
include "csl-mlir/Dialect/CSL/IR/CSLAttrs.td"
include "csl-mlir/Dialect/CSL/IR/CSLTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// CSL Type Definitions
//===----------------------------------------------------------------------===//

class CSL_Type<string name, string typeMnemonic, list<Trait> traits = [], 
        string baseCppClass = "::mlir::Type">
    : TypeDef<CSL_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def CSL_FunctionRefType : CSL_Type<"FunctionRef", "function_ref"> {
  let summary = "Compile-time function reference type";
  let description = [{
    Represents a reference to a function at compile time.
  }];
}

//===----------------------------------------------------------------------===//
// Comptime-Only Types
//===----------------------------------------------------------------------===//

def CSL_ComptimeIntType : CSL_Type<"ComptimeInt", "comptime_int"> {
  let summary = "Compile-time integer with arbitrary precision";
  let description = [{
    Represents an integer value known at compile time with arbitrary precision.
    Integer literals use this type, before they are coerced to specific bitwidths.
  }];
}

def CSL_ComptimeFloatType : CSL_Type<"ComptimeFloat", "comptime_float"> {
  let summary = "Compile-time floating-point value";
  let description = [{
    Represents a floating-point value known at compile time with double
    precision (IEEE-754 binary64).
    Floating-point literals use this type, before they are coerced to specific
    bitwidths.
  }];
}

def CSL_ComptimeStringType : CSL_Type<"ComptimeString", "comptime_string"> {
  let summary = "Compile-time string value";
  let description = [{
    Represents a string value known at compile time.

    Used for string literals and compile-time string operations.
  }];
}

def CSL_ImportedModuleType : CSL_Type<"ImportedModule", "imported_module"> {
  let summary = "Represents an imported module at compile time";
  let description = [{
    Represents a module at compile time.
  }];
}

def CSL_GenericFunctionType : CSL_Type<"GenericFunction", "generic_fn"> {
  let summary = "Opaque marker type for generic function references";
  let description = [{
    Marker type indicating a reference to a generic function that hasn't been
    instantiated yet. This is an opaque type that stores no information.

    Generic functions have comptime or anytype parameters and don't have a
    concrete FunctionType until they are instantiated at a call site.

    **Semantic rule**: Only `CallOp` accepts `GenericFunctionType` as callee,
    because only calls can instantiate generic functions. Any other use of a
    generic function reference (passing to builtins, storing in variables) is
    an error - generic functions cannot be referenced at runtime.

    Example:
    ```mlir
    // GetSymbolOp referencing a generic function infers to GenericFunctionType
    %f = parsed_csl.get_symbol @identity  // inferred type: !parsed_csl.generic_fn
    %result = parsed_csl.call %f(%x)    // OK: call instantiates the generic
    ```
  }];
}


//===----------------------------------------------------------------------===//
// Type Type (First-Class Types)
//===----------------------------------------------------------------------===//

def CSL_TypeType : CSL_Type<"Type", "type"> {
  let summary = "First-class type value";
  let description = [{
    Represents a type as a first-class value. This is the result type of
    type construction operations (builtin_type, ptr_type, array_type, etc.)
    and allows types to be computed and passed around at compile time.

    This is used in the ParsedCSL stage where types can be dynamically computed.

    Example:
    ```mlir
    !csl.type
    ```

    During comptime evaluation, operations with !csl.type results are
    evaluated to TypeAttr, which wraps a concrete MLIR Type.

    After conversion to TypedCSL, all type construction operations are eliminated
    and their results are used to resolve the actual types of values.
  }];
}

//===----------------------------------------------------------------------===//
// Integer Types (using MLIR builtins)
//===----------------------------------------------------------------------===//

// Individual integer type aliases
def CSL_I8Type : TypeAlias<SI8, "si8">;
def CSL_I16Type : TypeAlias<SI16, "si16">;
def CSL_I32Type : TypeAlias<SI32, "si32">;
def CSL_U8Type : TypeAlias<UI8, "ui8">;
def CSL_U16Type : TypeAlias<UI16, "ui16">;
def CSL_U32Type : TypeAlias<UI32, "ui32">;


// Signed integers: si8, si16, si32, si64
def CSL_SignedInt : SignedIntOfWidths<[8, 16, 32, 64]>;

// Unsigned integers: ui8, ui16, ui32, ui64
def CSL_UnsignedInt : UnsignedIntOfWidths<[8, 16, 32, 64]>;

// All CSL integers (signed or unsigned)
def CSL_Int : AnyTypeOf<[CSL_SignedInt, CSL_UnsignedInt], "CSL integer">;

//===----------------------------------------------------------------------===//
// Floating-Point Types (using MLIR builtins)
//===----------------------------------------------------------------------===//

// Individual float type aliases
def CSL_F16Type : TypeAlias<F16, "f16">;
def CSL_F32Type : TypeAlias<F32, "f32">;
def CSL_BF16Type : TypeAlias<BF16, "bf16">;

// CB16 (Cerebras Float 16): 1 sign, 6 exponent (bias 42), 9 mantissa bits
def CSL_CB16Type : CSL_Type<"CB16", "cb16",
    [DeclareTypeInterfaceMethods<FloatTypeInterface, ["getFloatSemantics"]>,
     DeclareTypeInterfaceMethods<CSL_TypeInfoInterface, ["getSizeInBytes"]>]> {
  let summary = "Cerebras 16-bit floating-point type";
  let description = [{
    A Cerebras-specific 16-bit floating-point format with 1 sign bit,
    6 exponent bits (bias 42), and 9 mantissa bits.
  }];
  let extraClassDefinition = [{
    ::std::optional<uint32_t> $cppClass::getSizeInBytes() const { return 2; }
  }];
}

// All CSL float types
def CSL_Float : AnyTypeOf<[F16, F32, BF16, CSL_CB16Type], "CSL float">;

//===----------------------------------------------------------------------===//
// Boolean and Void Types
//===----------------------------------------------------------------------===//

// Boolean type (using MLIR builtin i1)
def CSL_BoolType : TypeAlias<I1, "bool">;

def CSL_VoidType : CSL_Type<"Void", "void",
    [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface, ["isRuntimeType"]>]> {
  let summary = "Void type (absence of value)";
  let description = [{
    Represents the absence of a value, typically used for functions that
    do not return a value.
  }];
  let extraClassDefinition = [{
    bool $cppClass::isRuntimeType() const { return false; }
  }];
}

//===----------------------------------------------------------------------===//
// LValue Type (Resolved - used in TypedCSL)
//===----------------------------------------------------------------------===//

def CSL_LValueType : CSL_Type<"LValue", "lvalue",
    [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface, ["isRuntimeType"]>]> {
  let summary = "Storage location reference type";
  let description = [{
    Represents a reference to a storage location (lvalue) that can be loaded
    from and (if mutable) assigned to.

    This type is used in TypedCSL after type resolution and comptime evaluation.
    Both the valueType and isConst are always known and resolved.

    The isConst parameter indicates whether this lvalue refers to a const value.

    Examples:
    ```mlir
    !csl.lvalue<mutable si32>
    !csl.lvalue<const f32>
    ```

    In ParsedCSL, all values use !parsed_csl.unresolved. The lvalue/rvalue
    distinction is tracked during comptime analysis via inferred types.
    Values whose inferred type is LValueType represent storage locations.
  }];

  let parameters = (ins
    DefaultValuedParameter<"bool", "false">:$isConst,
    "Type":$valueType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$valueType,
                                         CArg<"bool", "false">:$isConst), [{
      return $_get(valueType.getContext(), isConst, valueType);
    }]>
  ];

  //  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "`<` custom<Mutability>($isConst) $valueType `>`";

  let extraClassDefinition = [{
    bool $cppClass::isRuntimeType() const {
      return ::mlir::csl::isRuntimeType(getValueType());
    }
  }];
}

// Type constraint for mutable lvalues (used by AssignOp in TypedCSL)
def CSL_MutableLValueType : Type<
  CPred<"::llvm::isa<::mlir::csl::LValueType>($_self) && "
        "!::llvm::cast<::mlir::csl::LValueType>($_self).getIsConst()">,
  "mutable lvalue">;

//===----------------------------------------------------------------------===//
// Pointer Type
//===----------------------------------------------------------------------===//

def CSL_PointerType : CSL_Type<"Pointer", "ptr",
    [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface,
        ["getSizeInBytes", "isRuntimeType"]>]> {
  let summary = "Pointer type";
  let description = [{
    A value of pointer type contains the memory address where a variable is located.

    There are two kinds of pointers in CSL: Single-item pointers and many-item pointers.
    - Single-item pointers point to a single element of the pointee type.
    - Many-item pointers point to an unknown number of elements of the pointee type

    Examples:
    ```mlir
    // Single-item pointers:
    !csl.ptr<ui16>       // *u16
    !csl.ptr<const !csl.array<10xui16>> // *const [10]u16

    // Many-item pointers:
    !csl.ptr<? x f32>    // [*]f32
    !csl.ptr<const ? x ui16> // [*]const u16
    ```
  }];

  let parameters = (ins
    "Type":$pointee_type,
    DefaultValuedParameter<"bool", "false">:$is_const,
    DefaultValuedParameter<"bool", "false">:$is_many_ptr
  );

  let assemblyFormat = "`<` custom<Mutability>($is_const) (`?` `x` $is_many_ptr^)? $pointee_type `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType,
                                         CArg<"bool", "false">:$isConst,
                                         CArg<"bool", "false">:$isArray), [{
      return $_get(pointeeType.getContext(), pointeeType, isConst, isArray);
    }]>
  ];

  let extraClassDefinition = [{
    std::optional<uint32_t> $cppClass::getSizeInBytes() const {
      return 2; // 16-bit pointers on WSE
    }

    // A pointer can only point to memory, so a pointer to a comptime-only
    // type is comptime-only.  Functions are the exception: a pointer to a
    // function is a runtime value.
    bool $cppClass::isRuntimeType() const {
      return ::mlir::isa<::mlir::FunctionType>(getPointeeType()) ||
             ::mlir::csl::isRuntimeType(getPointeeType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Array Type
//===----------------------------------------------------------------------===//

def CSL_ArrayType : CSL_Type<"Array", "array",
    [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface,
        ["getSizeInBytes", "isRuntimeType", "hasPerPELayout",
         "getSizeInBytesForPE"]>]> {
  let summary = "Multi-dimensional array type";
  let description = [{
    Represents a multi-dimensional array with a specific shape and element type.

    Examples:
    ```mlir
    !csl.array<10x20xsi32  // [10, 20]i32
    !csl.array<5xf32>           // [5]f32
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let assemblyFormat = "`<` custom<ArrayShape>($shape) `x` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                         "Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    // getShape() and getElementType() are automatically generated

    // Get the number of dimensions
    unsigned getRank() const { return getShape().size(); }

    // Get the total number of elements
    int64_t getNumElements() const;
  }];

  let extraClassDefinition = [{
    ::std::optional<uint32_t> $cppClass::getSizeInBytes() const {
      auto elemSize = ::mlir::csl::getSizeInBytes(getElementType());
      if (!elemSize)
        return ::std::nullopt;
      return *elemSize * static_cast<uint32_t>(getNumElements());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Per-PE Array Type
//===----------------------------------------------------------------------===//

def CSL_PerPEArrayType : CSL_Type<"PerPEArray", "per_pe_array",
    [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface,
        ["getSizeInBytes", "isRuntimeType", "hasPerPELayout",
         "getSizeInBytesForPE"]>]> {
  let summary = "Array type with per-PE dimensions";
  let description = [{
    Represents a multi-dimensional array where some dimensions may have
    different sizes for each PE in the grid. This enables workload distribution
    patterns where different PEs process different amounts of data.

    Each dimension is either static (one size for all PEs) or per-PE.
    Per-PE data covers a data_width x data_height sub-grid. PEs at
    coordinates (x, y) where x >= data_width or y >= data_height
    implicitly have dimension size 0.

    Parameters:
    - data_width, data_height: Dimensions of the per-PE data grid
    - offset_x, offset_y: Grid offset in the fabric (from fabricOffsets)
    - dim_is_per_pe: One bool per array dimension (true = per-PE)
    - dim_data: Packed dimension data (1 value per static dim,
      data_width*data_height values per per-PE dim)
    - elementType: Element type of the array

    Example:
    ```mlir
    // 1D per-PE on 2x2 grid:
    !csl.per_pe_array<grid 2x2, pp<10,20,30,40> x si32>

    // With offset:
    !csl.per_pe_array<grid 2x2, offset 3x5, pp<10,20,30,40> x si32>

    // 2D mixed (static first dim, per-PE second dim) on 3x1 grid:
    !csl.per_pe_array<grid 3x1, 53 x pp<100,200,300> x f32>
    ```
  }];

  let parameters = (ins
    "int64_t":$data_width,
    "int64_t":$data_height,
    "int64_t":$offset_x,
    "int64_t":$offset_y,
    ArrayRefParameter<"bool">:$dim_is_per_pe,
    ArrayRefParameter<"int64_t">:$dim_data,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "int64_t":$dataWidth, "int64_t":$dataHeight,
        "int64_t":$offsetX, "int64_t":$offsetY,
        "ArrayRef<bool>":$dimIsPerPE,
        "ArrayRef<int64_t>":$dimData,
        "Type":$elementType), [{
      return $_get(elementType.getContext(), dataWidth, dataHeight,
                   offsetX, offsetY, dimIsPerPE, dimData, elementType);
    }]>,
    // Convenience builder with zero offsets
    TypeBuilderWithInferredContext<(ins
        "int64_t":$dataWidth, "int64_t":$dataHeight,
        "ArrayRef<bool>":$dimIsPerPE,
        "ArrayRef<int64_t>":$dimData,
        "Type":$elementType), [{
      return $_get(elementType.getContext(), dataWidth, dataHeight,
                   0, 0, dimIsPerPE, dimData, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get the number of dimensions
    unsigned getRank() const { return getDimIsPerPe().size(); }

    /// Check if a specific dimension is per-PE
    bool isDimPerPE(unsigned dimIdx) const {
      assert(dimIdx < getRank() && "dimension index out of range");
      return getDimIsPerPe()[dimIdx];
    }

    /// Get the offset into dim_data where a specific dimension's data starts
    unsigned getDimDataOffset(unsigned dimIdx) const;

    /// Get a static dimension size (asserts the dim is static)
    int64_t getStaticDimSize(unsigned dimIdx) const;

    /// Get all per-PE sizes for a dimension (asserts the dim is per-PE)
    ArrayRef<int64_t> getPerPEDimSizes(unsigned dimIdx) const;

    /// Get the size of a dimension at a specific PE coordinate
    int64_t getDimSizeForPE(unsigned dimIdx, int64_t x, int64_t y) const;

    /// Get the full shape for a specific PE
    SmallVector<int64_t> getShapeForPE(int64_t x, int64_t y) const;

    /// Get a concrete ArrayType for a specific PE
    ArrayType getArrayTypeForPE(int64_t x, int64_t y) const;

    /// Get the total number of elements for a specific PE
    int64_t getNumElementsForPE(int64_t x, int64_t y) const;
  }];
}

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//

def CSL_StructType : CSL_Type<"Struct", "struct",
  [PredTypeTrait<"named tuples not allowed", CPred<"!($field_names.empty() && $def_location)">>,
   DeclareTypeInterfaceMethods<CSL_TypeInfoInterface,
       ["getSizeInBytes", "isRuntimeType", "hasPerPELayout",
        "getSizeInBytesForPE"]>]> {
  let summary = "Unified struct type (tuples, anonymous structs, named structs)";
  let description = [{
    Represents a struct type, covering three variants based on parameters:
    1. **Tuple**: field_names is nullopt, def_location is nullopt
       - Positional access only via compile-time indices
       - Syntax: type,type,...
       - Example: !csl.struct<f16,f16>

    2. **Anonymous struct**: field_names present, def_location is nullopt
       - Named field access
       - Structural typing (two structs equal if same field names and types)
       - Syntax: type:name,type:name,...
       - Example: !csl.struct<f16:real_part,f16:imag_part>

    3. **Named struct**: field_names present, def_location present
       - Named field access
       - Nominal typing (identity by definition location)
       - Syntax: loc(...),type:name,type:name,...
       - Example: !csl.struct<loc("./pe.csl":92:3),f16:real_part,f16:imag_part>

    Type equality: Two struct types are equal if and only if all three
    parameters match (field_names, field_types, def_location).
  }];

  let parameters = (ins
  OptionalArrayRefParameter<"StringAttr">:$field_names,  // nullopt for tuples
    ArrayRefParameter<"Type">:$field_types,                // always present
    OptionalParameter<"LocationAttr">:$def_location        // for nominal typing
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Builder for tuples (no field names, no location)
    TypeBuilder<(ins "ArrayRef<Type>":$field_types), [{
      return $_get($_ctxt, ArrayRef<StringAttr>(), field_types, LocationAttr());
    }]>,

    // Builder for anonymous structs (field names as StringAttr, no location)
    TypeBuilder<(ins "ArrayRef<StringAttr>":$field_names,
                     "ArrayRef<Type>":$field_types), [{
      assert(field_names.size() == field_types.size() &&
             "field names and types must have same size");
      return $_get($_ctxt, field_names, field_types, LocationAttr());
    }]>
  ];

  let extraClassDeclaration = [{
    // Check if this is a tuple (no field names)
    bool isTuple() const { return getFieldNames().empty(); }

    // Check if this is an anonymous struct or tuple (no location)
    bool isAnonymous() const {
      return !getDefLocation();
    }

    // Get number of fields
    size_t getNumFields() const { return getFieldTypes().size(); }

    // Get field type by name (returns null if not found or is a tuple)
    Type getFieldType(StringRef name) const;

    // Get field type by index
    Type getFieldType(size_t index) const {
      assert(index < getNumFields() && "field index out of range");
      return getFieldTypes()[index];
    }

    // Get field index by name (returns failure if not found or is a tuple)
    FailureOr<size_t> getFieldIndex(StringRef name) const;

    // Get field name by index (returns failure if tuple)
    FailureOr<StringRef> getFieldName(size_t index) const {
      if (getFieldNames().empty())
        return failure();
      assert(index < getNumFields() && "field index out of range");
      return getFieldNames()[index].getValue();
    }
  }];

  let extraClassDefinition = [{
    ::std::optional<uint32_t> $cppClass::getSizeInBytes() const {
      uint32_t total = 0;
      uint32_t maxAlign = 1;
      for (::mlir::Type ft : getFieldTypes()) {
        // A void field takes no memory.
        if (::mlir::isa<::mlir::csl::VoidType>(ft))
          continue;
        auto fieldSize = ::mlir::csl::getSizeInBytes(ft);
        if (!fieldSize)
          return ::std::nullopt;
        // Align the current offset to the field's natural alignment.
        uint32_t align = ::mlir::csl::getAlignmentForType(ft);
        total = (total + align - 1) & ~(align - 1);
        total += *fieldSize;
        if (align > maxAlign)
          maxAlign = align;
      }
      // Pad to struct alignment so arrays of structs stay aligned.
      total = (total + maxAlign - 1) & ~(maxAlign - 1);
      return total;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Enum Type
//===----------------------------------------------------------------------===//

def CSL_EnumType : CSL_Type<"Enum", "enum",
    [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface, ["getSizeInBytes"]>]> {
  let summary = "Enumeration type with underlying integer type";
  let description = [{
    Represents an enumeration type with named elements mapping to an underlying
    integer type. Each element has a unique integer value.

    Parameters:
    - underlying_type: The integer type (e.g., si16, ui32)
    - element_names: Array of enum element names (e.g., ["red", "white", "blue"])
    - element_values: Array of underlying integer values for each element
    - def_location: Source location for nominal typing

    Type equality: Two enum types are equal if and only if they have the same
    def_location (nominal typing). Two enums with identical structure but
    different source locations are different types.

    Example:
    ```mlir
    !csl.enum<loc("test.csl":5:3), si16, "red":0, "white":1, "blue":2>
    ```

    In CSL source:
    ```csl
    const colors = enum(u16) { red, white, blue };
    const favorite = colors.red;
    ```
  }];

  let parameters = (ins
    "IntegerType":$underlying_type,
    ArrayRefParameter<"StringAttr">:$element_names,
    ArrayRefParameter<"int64_t">:$element_values,
    "LocationAttr":$def_location
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Get the number of enum elements
    size_t getNumElements() const { return getElementNames().size(); }

    // Get element value by name (returns failure if not found)
    FailureOr<int64_t> getElementValue(StringRef name) const;

    // Get element name by value (returns failure if not found)
    FailureOr<StringRef> getElementName(int64_t value) const;

    // Check if a value is valid for this enum
    bool isValidValue(int64_t value) const;

    // Get element index by name (returns failure if not found)
    FailureOr<size_t> getElementIndex(StringRef name) const;
  }];

  let extraClassDefinition = [{
    ::std::optional<uint32_t> $cppClass::getSizeInBytes() const {
      return ::mlir::csl::getSizeInBytes(getUnderlyingType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Hardware-Specific Types
//===----------------------------------------------------------------------===//

def CSL_ColorType : CSL_Type<"Color", "color"> {
  let summary = "Virtual communication channel type";
  let description = [{
    Represents a virtual communication channel (color) on the Cerebras WSE.
  }];

  let parameters = (ins);
}

def CSL_InputQueueType : CSL_Type<"InputQueue", "input_queue"> {
  let summary = "Input queue type";

  let parameters = (ins);
}

def CSL_OutputQueueType : CSL_Type<"OutputQueue", "output_queue"> {
  let summary = "Output queue type for WSE-3";
}

def CSL_DataTaskIdType : CSL_Type<"DataTaskId", "data_task_id"> {
  let summary = "Data task identifier type";
}

def CSL_LocalTaskIdType : CSL_Type<"LocalTaskId", "local_task_id"> {
  let summary = "Local task identifier type";
}

def CSL_ControlTaskIdType : CSL_Type<"ControlTaskId", "control_task_id"> {
  let summary = "Control task identifier type";
}

def CSL_UtIdType : CSL_Type<"UtId", "ut_id"> {
  let summary = "Microthread ID type (WSE-3)";
}

def CSL_DirectionType : CSL_Type<"Direction", "direction"> {
  let summary = "Routing direction type";
  let description = [{
    Represents a routing direction on the Cerebras WSE fabric.
    Valid values are NORTH, SOUTH, EAST, WEST, and RAMP.
  }];
}

//===----------------------------------------------------------------------===//
// DSR (Data Structure Register) Types
//===----------------------------------------------------------------------===//

def CSL_DsrDestType : CSL_Type<"DsrDest", "dsr_dest"> {
  let summary = "Destination operand DSR type";
  let description = [{
    Represents a Data Structure Register (DSR) that can be used as a
    destination operand in DSD operations.
  }];
}

def CSL_DsrSrc0Type : CSL_Type<"DsrSrc0", "dsr_src0"> {
  let summary = "Source and destination operand DSR type";
  let description = [{
    Represents a DSR that can be used as both source and destination
    operand in DSD operations.
  }];
}

def CSL_DsrSrc1Type : CSL_Type<"DsrSrc1", "dsr_src1"> {
  let summary = "Source operand DSR type";
  let description = [{
    Represents a DSR that can be used as a source operand in DSD operations.
  }];
}

def CSL_DsrFifoDestType : CSL_Type<"DsrFifoDest", "dsr_fifo_dest"> {
  let summary = "FIFO destination DSR type";
  let description = [{
    Represents a FIFO DSR for destination operations. Can be used where
    dsr_dest is expected in DSD operations.
  }];
}

def CSL_DsrFifoSrc1Type : CSL_Type<"DsrFifoSrc1", "dsr_fifo_src1"> {
  let summary = "FIFO source DSR type";
  let description = [{
    Represents a FIFO DSR for source operations. Can be used where
    dsr_src1 is expected in DSD operations.
  }];
}

def CSL_XdsrType : CSL_Type<"Xdsr", "xdsr"> {
  let summary = "Extended DSR type";
  let description = [{
    Represents an extended Data Structure Register (XDSR) identifier.
  }];
}

def CSL_SrType : CSL_Type<"Sr", "sr"> {
  let summary = "stride-register type";
  let description = [{
    Represents a physical stride-register identifier.
  }];
}

//===----------------------------------------------------------------------===//
// DSD (Data Structure Descriptor) Types
//===----------------------------------------------------------------------===//

def CSL_Mem1dDsdType : CSL_Type<"Mem1dDsd", "mem1d_dsd"> {
  let summary = "1D memory DSD type";
  let description = [{
    Represents a 1D memory Data Structure Descriptor with a single induction
    variable. Used to encode memory tensors with base address, extent, stride,
    and optional offset.
  }];
}

def CSL_Mem4dDsdType : CSL_Type<"Mem4dDsd", "mem4d_dsd"> {
  let summary = "Multi-dimensional memory DSD type (up to 4D)";
  let description = [{
    Represents a multi-dimensional memory DSD with up to 4 induction variables.
    Used for complex memory access patterns in multi-dimensional tensors.
  }];
}

def CSL_FabinDsdType : CSL_Type<"FabinDsd", "fabin_dsd"> {
  let summary = "Fabric input DSD type";
  let description = [{
    Represents a fabric input DSD for receiving wavelets from the fabric.
    Configured with an input queue or fabric color, and an extent.
  }];
}

def CSL_FaboutDsdType : CSL_Type<"FaboutDsd", "fabout_dsd"> {
  let summary = "Fabric output DSD type";
  let description = [{
    Represents a fabric output DSD for sending wavelets to the fabric.
    Configured with an output queue or fabric color, an extent, and optional
    control/wavelet_index_offset properties.
  }];
}

def CSL_FifoDsdType : CSL_Type<"FifoDsd", "fifo_dsd"> {
  let summary = "FIFO DSD type";
  let description = [{
    Represents a FIFO Data Structure Descriptor. FIFOs are created via
    @allocate_fifo and provide first-in-first-out buffering.
  }];
}

//===----------------------------------------------------------------------===//
// Range Type
//===----------------------------------------------------------------------===//
def CSL_RangeType : CSL_Type<"Range", "range"> {
  let summary = "Range type representing a sequence of values";
  let description = [{
    Represents a range of values, typically used in iteration constructs.
    A range type encapsulates a start, stop, and step value.
    
    Example:
    ```mlir
    !csl.range<si32>
    ```
  }];

  let parameters = (ins "Type":$element_type);
  let assemblyFormat = "`<` $element_type `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Opaque Type
//===----------------------------------------------------------------------===//

def CSL_OpaqueType : CSL_Type<"Opaque", "opaque"> {
  let summary = "Opaque type emitted as-is";
  let description = [{
    Represents an opaque type that is emitted as-is in the generated code.
    Used for types that are not directly representable in the dialect.

    Example:
    ```mlir
    !csl.opaque<"implementation_defined_type">
    ```
  }];

  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Coroutine Frame Type
//===----------------------------------------------------------------------===//

def CSL_CoroFrameType : CSL_Type<"CoroFrame", "coroframe",
  [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface, ["isRuntimeType"]>]> {
  let summary = "Coroutine frame handle type";
  let description = [{
    Represents a handle to a suspended coroutine's frame. Parameterized by
    the symbol name (and optional module identifier) of the async function,
    which uniquely identifies the coroutine. The return type of the async
    function can be recovered by looking up the function's
    `async_result_type` attribute.

    Before module inlining, functions inside modules are identified by
    their module_identifier + symbol_name pair. After inlining, the
    module_identifier is absent and the symbol_name is the flattened name.

    Examples:
    ```mlir
    !csl.coroframe<@my_async_fn>        // top-level function
    !csl.coroframe<123, @module_fn>     // function in module 123
    ```
  }];

  let parameters = (ins
    OptionalParameter<"std::optional<uint32_t>">:$module_identifier,
    "::mlir::FlatSymbolRefAttr":$symbol_name
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns true if this is a top-level function (no module qualifier)
    bool isTopLevel() const { return !getModuleIdentifier().has_value(); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isRuntimeType() const {
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Future Type
//===----------------------------------------------------------------------===//

def CSL_FutureType : CSL_Type<"Future", "future",
  [DeclareTypeInterfaceMethods<CSL_TypeInfoInterface, ["isRuntimeType"]>]> {
  let summary = "Future handle type for async function calls";
  let description = [{
    Represents a future — a handle to a running coroutine. Semantically a
    pointer to the coroutine frame. Returned by async function calls and
    consumed by await.

    Parameterized by the symbol name (and optional module identifier) of
    the async function.

    Examples:
    ```mlir
    !csl.future<@my_async_fn>
    !csl.future<123, @module_fn>
    ```
  }];

  let parameters = (ins
    OptionalParameter<"std::optional<uint32_t>">:$module_identifier,
    "::mlir::FlatSymbolRefAttr":$symbol_name,
    OptionalParameter<"::mlir::Type">:$result_type
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool isTopLevel() const { return !getModuleIdentifier().has_value(); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isRuntimeType() const {
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Type Predicates
//===----------------------------------------------------------------------===//

// Runtime type predicates
def CSL_AnyRuntimeIntType : AnyTypeOf<[CSL_Int]> {
  let cppFunctionName = "isAnyRuntimeIntType";
}

def CSL_AnyIntegerType : AnyTypeOf<[CSL_Int, CSL_ComptimeIntType]> {
  let cppFunctionName = "isAnyIntegerType";
}

def CSL_AnyRuntimeFloatType : AnyTypeOf<[CSL_Float]> {
  let cppFunctionName = "isAnyRuntimeFloatType";
}

// DSD type predicates
def CSL_AnyDsdType : AnyTypeOf<[
  CSL_Mem1dDsdType,
  CSL_Mem4dDsdType,
  CSL_FabinDsdType,
  CSL_FaboutDsdType,
  CSL_FifoDsdType
], "DSD type"> {
  let cppFunctionName = "isAnyDsdType";
}

def CSL_AnyMemDsdType : AnyTypeOf<[
  CSL_Mem1dDsdType,
  CSL_Mem4dDsdType
], "memory DSD type"> {
  let cppFunctionName = "isAnyMemDsdType";
}

def CSL_AnyFabDsdType : AnyTypeOf<[
  CSL_FabinDsdType,
  CSL_FaboutDsdType
], "fabric DSD type"> {
  let cppFunctionName = "isAnyFabDsdType";
}

// DSR type predicates
def CSL_AnyDsrType : AnyTypeOf<[
  CSL_DsrDestType,
  CSL_DsrSrc0Type,
  CSL_DsrSrc1Type,
  CSL_DsrFifoDestType,
  CSL_DsrFifoSrc1Type,
  CSL_XdsrType
], "DSR type"> {
  let cppFunctionName = "isAnyDsrType";
}


// All runtime types — delegates to the csl::isRuntimeType() free function so
// the TableGen predicate and the C++ function stay in sync automatically.
def CSL_AnyRuntimeType : Type<
  CPred<"::mlir::csl::isRuntimeType($_self)">,
  "CSL runtime type">;

def CSL_AnyComptimeType : Type<
  CPred<"!::mlir::csl::isRuntimeType($_self)">,
  "CSL comptime type"> {
    let cppFunctionName = "isComptimeOnlyType";
}

#endif // CSL_TYPES
