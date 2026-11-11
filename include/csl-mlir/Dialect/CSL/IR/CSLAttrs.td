//===- CSLAttrs.td - CSL Attribute Definitions ------------*- tablegen -*-===//
//
// Attribute definitions for the CSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_ATTRS
#define CSL_ATTRS

include "csl-mlir/Dialect/CSL/IR/CSLBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// CSL Attribute Base Class
//===----------------------------------------------------------------------===//

class CSL_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CSL_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Binary Operation Kind
def CSL_BinaryOpKind : I32EnumAttr<"BinaryOpKind",
    "Binary operation kind", [
  // Arithmetic
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"Div", 3, "div">,
  I32EnumAttrCase<"Mod", 4, "mod">,

  // Bitwise
  I32EnumAttrCase<"BitAnd", 10, "and">,
  I32EnumAttrCase<"BitOr", 11, "or">,
  I32EnumAttrCase<"BitXor", 12, "xor">,
  I32EnumAttrCase<"Shl", 13, "shl">,
  I32EnumAttrCase<"Shr", 14, "shr">,

  // Comparison
  I32EnumAttrCase<"Eq", 20, "eq">,
  I32EnumAttrCase<"Ne", 21, "ne">,
  I32EnumAttrCase<"Lt", 22, "lt">,
  I32EnumAttrCase<"Le", 23, "le">,
  I32EnumAttrCase<"Gt", 24, "gt">,
  I32EnumAttrCase<"Ge", 25, "ge">,

  // Logical
  I32EnumAttrCase<"LogicalAnd", 30, "land">,
  I32EnumAttrCase<"LogicalOr", 31, "lor">
]> {
  let cppNamespace = "::mlir::csl";
}

// Short-circuit Operation Kind (for && and ||)
def CSL_ShortCircuitKind : I32EnumAttr<"ShortCircuitKind",
    "Short-circuit operation kind", [
  I32EnumAttrCase<"And", 0, "and">,  // &&: short-circuit if LHS is false
  I32EnumAttrCase<"Or", 1, "or">     // ||: short-circuit if LHS is true
]> {
  let cppNamespace = "::mlir::csl";
}

// Unary Operation Kind
def CSL_UnaryOpKind : I32EnumAttr<"UnaryOpKind",
    "Unary operation kind", [
  I32EnumAttrCase<"Neg", 0, "neg">,         // Arithmetic negation: -x
  I32EnumAttrCase<"BitNot", 1, "bitnot">,   // Bitwise complement: ~x
  I32EnumAttrCase<"LogicalNot", 2, "lnot">  // Logical negation: !x
]> {
  let cppNamespace = "::mlir::csl";
}

// Direction (for routing)
def CSL_Direction : I32EnumAttr<"Direction",
    "Routing direction", [
  I32EnumAttrCase<"NORTH", 0, "NORTH">,
  I32EnumAttrCase<"SOUTH", 1, "SOUTH">,
  I32EnumAttrCase<"EAST", 2, "EAST">,
  I32EnumAttrCase<"WEST", 3, "WEST">,
  I32EnumAttrCase<"RAMP", 4, "RAMP">
]> {
  let cppNamespace = "::mlir::csl";
  // Don't auto-generate DirectionAttr - we define a custom TypedAttr below
  let genSpecializedAttr = 0;
}

// Priority level for microthreads
def CSL_PriorityLevel : I32EnumAttr<"PriorityLevel",
    "Microthread priority level", [
  I32EnumAttrCase<"High", 0, "high">,
  I32EnumAttrCase<"Medium", 1, "medium">,
  I32EnumAttrCase<"Low", 2, "low">
]> {
  let cppNamespace = "::mlir::csl";
}

// SIMD mode for fabric DSDs
def CSL_SimdModeKind : I32EnumAttr<"SimdModeKind",
    "SIMD mode for fabric DSDs", [
  I32EnumAttrCase<"Simd32", 0, "simd_32">,
  I32EnumAttrCase<"Simd64", 1, "simd_64">,
  I32EnumAttrCase<"Simd32Or64", 2, "simd_32_or_64">,
  I32EnumAttrCase<"Simd16", 3, "simd_16">,
  I32EnumAttrCase<"SimdOff", 4, "simd_off">,
  I32EnumAttrCase<"SimdMax", 5, "simd_max">
]> {
  let cppNamespace = "::mlir::csl";
}

// Zero source for DSD reset
def CSL_ZeroSource : I32EnumAttr<"ZeroSource",
    "Which source operand to zero after DSD operation", [
  I32EnumAttrCase<"FirstSource", 0, "first_source">,
  I32EnumAttrCase<"SecondSource", 1, "second_source">
]> {
  let cppNamespace = "::mlir::csl";
}

// Builtin functions
def CSL_BuiltinKind : I32EnumAttr<"BuiltinKind",
    "CSL builtin function kind", [
  // Core language builtins
  I32EnumAttrCase<"Activate", 0, "activate">,
  I32EnumAttrCase<"AllocateFifo", 1, "allocate_fifo">,
  I32EnumAttrCase<"As", 2, "as">,
  I32EnumAttrCase<"Assert", 3, "assert">,
  I32EnumAttrCase<"Bitcast", 4, "bitcast">,
  I32EnumAttrCase<"BindControlTask", 5, "bind_control_task">,
  I32EnumAttrCase<"BindDataTask", 6, "bind_data_task">,
  I32EnumAttrCase<"BindLocalTask", 7, "bind_local_task">,
  I32EnumAttrCase<"Block", 8, "block">,
  I32EnumAttrCase<"ComptimeAssert", 9, "comptime_assert">,
  I32EnumAttrCase<"ComptimePrint", 10, "comptime_print">,
  I32EnumAttrCase<"Constants", 12, "constants">,
  I32EnumAttrCase<"Dimensions", 13, "dimensions">,
  I32EnumAttrCase<"ElementCount", 14, "element_count">,
  I32EnumAttrCase<"ElementType", 15, "element_type">,
  I32EnumAttrCase<"Field", 16, "field">,
  I32EnumAttrCase<"Fp16", 17, "fp16">,
  I32EnumAttrCase<"GetArray", 18, "get_array">,
  I32EnumAttrCase<"GetColor", 19, "get_color">,
  I32EnumAttrCase<"GetConfig", 20, "get_config">,
  I32EnumAttrCase<"GetConfigUnchecked", 21, "get_config_unchecked">,
  I32EnumAttrCase<"GetControlTaskId", 22, "get_control_task_id">,
  I32EnumAttrCase<"GetDataTaskId", 23, "get_data_task_id">,
  I32EnumAttrCase<"GetDsd", 24, "get_dsd">,
  I32EnumAttrCase<"GetDsr", 25, "get_dsr">,
  I32EnumAttrCase<"GetFilterId", 26, "get_filter_id">,
  I32EnumAttrCase<"GetInputQueue", 27, "get_input_queue">,
  I32EnumAttrCase<"GetInt", 28, "get_int">,
  I32EnumAttrCase<"GetLocalTaskId", 29, "get_local_task_id">,
  I32EnumAttrCase<"GetOutputQueue", 30, "get_output_queue">,
  I32EnumAttrCase<"GetRectangle", 31, "get_rectangle">,
  I32EnumAttrCase<"GetStringFromByte", 32, "get_string_from_byte">,
  I32EnumAttrCase<"HasField", 33, "has_field">,
  I32EnumAttrCase<"ImportComptimeValue", 34, "import_comptime_value">,
  I32EnumAttrCase<"ImportModule", 35, "import_module">,
  I32EnumAttrCase<"IncrementDsdOffset", 36, "increment_dsd_offset">,
  I32EnumAttrCase<"InitializeQueue", 37, "initialize_queue">,
  I32EnumAttrCase<"IsArch", 38, "is_arch">,
  I32EnumAttrCase<"IsComptime", 39, "is_comptime">,
  I32EnumAttrCase<"IsSameType", 40, "is_same_type">,
  I32EnumAttrCase<"LoadToDsr", 41, "load_to_dsr">,
  I32EnumAttrCase<"Map", 42, "map">,
  I32EnumAttrCase<"Ptrcast", 43, "ptrcast">,
  I32EnumAttrCase<"Random16", 44, "random16">,
  I32EnumAttrCase<"Range", 45, "range">,
  I32EnumAttrCase<"RangeStart", 46, "range_start">,
  I32EnumAttrCase<"RangeStop", 47, "range_stop">,
  I32EnumAttrCase<"RangeStep", 48, "range_step">,
  I32EnumAttrCase<"Rank", 49, "rank">,
  I32EnumAttrCase<"SetActivePrng", 50, "set_active_prng">,
  I32EnumAttrCase<"SetColorConfig", 51, "set_color_config">,
  I32EnumAttrCase<"SetLocalColorConfig", 52, "set_local_color_config">,
  I32EnumAttrCase<"SetConfig", 53, "set_config">,
  I32EnumAttrCase<"SetConfigUnchecked", 54, "set_config_unchecked">,
  I32EnumAttrCase<"SetDsdBaseAddr", 55, "set_dsd_base_addr">,
  I32EnumAttrCase<"SetDsdLength", 56, "set_dsd_length">,
  I32EnumAttrCase<"SetDsdStride", 57, "set_dsd_stride">,
  I32EnumAttrCase<"SetFifoReadLength", 58, "set_fifo_read_length">,
  I32EnumAttrCase<"SetFifoWriteLength", 59, "set_fifo_write_length">,
  I32EnumAttrCase<"SetRectangle", 60, "set_rectangle">,
  I32EnumAttrCase<"SetTeardownHandler", 61, "set_teardown_handler">,
  I32EnumAttrCase<"SetTileCode", 62, "set_tile_code">,
  I32EnumAttrCase<"Strcat", 63, "strcat">,
  I32EnumAttrCase<"Strlen", 64, "strlen">,
  I32EnumAttrCase<"TypeOf", 65, "type_of">,
  I32EnumAttrCase<"Unblock", 66, "unblock">,
  I32EnumAttrCase<"Zeros", 67, "zeros">,
  I32EnumAttrCase<"ExportName", 68, "export_name">,
  I32EnumAttrCase<"ExportSymbol", 69, "export_symbol">,
  I32EnumAttrCase<"GetSymbolId", 70, "get_symbol_id">,
  I32EnumAttrCase<"GetSymbolValue", 71, "get_symbol_value">,
  I32EnumAttrCase<"GetTensorPtr", 72, "get_tensor_ptr">,
  I32EnumAttrCase<"HasExportedTensors", 73, "has_exported_tensors">,
  I32EnumAttrCase<"Rpc", 74, "rpc">,
  I32EnumAttrCase<"FutureOf", 76, "future_of">,
  I32EnumAttrCase<"GetXdsr", 75, "get_xdsr">,
  I32EnumAttrCase<"GetSr", 80, "get_sr">,
  I32EnumAttrCase<"LoadToDsrXdsrSr", 81, "load_to_dsr_xdsr_sr">,

  // DSD/tensor operations
  I32EnumAttrCase<"Add16", 100, "add16">,
  I32EnumAttrCase<"Addc16", 101, "addc16">,
  I32EnumAttrCase<"And16", 102, "and16">,
  I32EnumAttrCase<"Clz", 103, "clz">,
  I32EnumAttrCase<"Ctz", 104, "ctz">,
  I32EnumAttrCase<"Dfilt", 140, "dfilt">,
  I32EnumAttrCase<"Fabsh", 105, "fabsh">,
  I32EnumAttrCase<"Fabss", 106, "fabss">,
  I32EnumAttrCase<"Faddh", 107, "faddh">,
  I32EnumAttrCase<"Faddhs", 108, "faddhs">,
  I32EnumAttrCase<"Fadds", 109, "fadds">,
  I32EnumAttrCase<"Fh2s", 110, "fh2s">,
  I32EnumAttrCase<"Fh2xp16", 111, "fh2xp16">,
  I32EnumAttrCase<"Fmach", 112, "fmach">,
  I32EnumAttrCase<"Fmachs", 113, "fmachs">,
  I32EnumAttrCase<"Fmacs", 114, "fmacs">,
  I32EnumAttrCase<"Fmaxh", 115, "fmaxh">,
  I32EnumAttrCase<"Fmaxs", 116, "fmaxs">,
  I32EnumAttrCase<"Fmovh", 117, "fmovh">,
  I32EnumAttrCase<"Fmovs", 118, "fmovs">,
  I32EnumAttrCase<"Fmulh", 119, "fmulh">,
  I32EnumAttrCase<"Fmuls", 120, "fmuls">,
  I32EnumAttrCase<"Fnegh", 121, "fnegh">,
  I32EnumAttrCase<"Fnegs", 122, "fnegs">,
  I32EnumAttrCase<"Fnormh", 123, "fnormh">,
  I32EnumAttrCase<"Fnorms", 124, "fnorms">,
  I32EnumAttrCase<"Fs2h", 125, "fs2h">,
  I32EnumAttrCase<"Fs2xp16", 126, "fs2xp16">,
  I32EnumAttrCase<"Fscaleh", 127, "fscaleh">,
  I32EnumAttrCase<"Fscales", 128, "fscales">,
  I32EnumAttrCase<"Fsubh", 129, "fsubh">,
  I32EnumAttrCase<"Fsubs", 130, "fsubs">,
  I32EnumAttrCase<"Mov16", 131, "mov16">,
  I32EnumAttrCase<"Mov32", 132, "mov32">,
  I32EnumAttrCase<"Or16", 133, "or16">,
  I32EnumAttrCase<"Popcnt", 134, "popcnt">,
  I32EnumAttrCase<"Sar16", 135, "sar16">,
  I32EnumAttrCase<"Sll16", 136, "sll16">,
  I32EnumAttrCase<"Slr16", 137, "slr16">,
  I32EnumAttrCase<"Sub16", 138, "sub16">,
  I32EnumAttrCase<"Xor16", 139, "xor16">,
  I32EnumAttrCase<"Xp162fh", 141, "xp162fh">,
  I32EnumAttrCase<"Xp162fs", 142, "xp162fs">,

// Internal builtins
  I32EnumAttrCase<"InternalRpc", 200, "__internal_rpc">,
  I32EnumAttrCase<"InternalPromoteToConstGlobal", 201, "__internal_promote_to_const_global">,

  // Builtins for WSE-3
  I32EnumAttrCase<"GetUtId", 300, "get_ut_id">,
  I32EnumAttrCase<"QueueFlush", 301, "queue_flush">,
  I32EnumAttrCase<"SetControlTaskTable", 302, "set_control_task_table">,
  I32EnumAttrCase<"SetEmptyQueueHandler", 303, "set_empty_queue_handler">,

  // Coroutine builtins
  I32EnumAttrCase<"Await", 77, "await">,
  I32EnumAttrCase<"AwaitParallel", 78, "await_parallel">,
  I32EnumAttrCase<"CurrentTask", 79, "current_task">,
  I32EnumAttrCase<"FrameAddress", 83, "frame_address">,
  // Marks where the coroutine runtime is initialized. Lowered to a call to
  // the runtime's init() in fiber mode; erased in trampoline mode, which has
  // no runtime state to set up.
  I32EnumAttrCase<"CoroRuntimeInit", 84, "coro_runtime_init">,

  // Resource claim/release builtins
  I32EnumAttrCase<"ClaimColor", 400, "claim_color">,
  I32EnumAttrCase<"ReleaseColor", 401, "release_color">,
  I32EnumAttrCase<"ClaimLocalTask", 402, "claim_local_task">,
  I32EnumAttrCase<"ReleaseLocalTask", 403, "release_local_task">,
  I32EnumAttrCase<"ClaimDataTask", 404, "claim_data_task">,
  I32EnumAttrCase<"ReleaseDataTask", 405, "release_data_task">,
  I32EnumAttrCase<"ClaimInputQueue", 406, "claim_input_queue">,
  I32EnumAttrCase<"ReleaseInputQueue", 407, "release_input_queue">,
  I32EnumAttrCase<"ClaimOutputQueue", 408, "claim_output_queue">,
  I32EnumAttrCase<"ReleaseOutputQueue", 409, "release_output_queue">,
  I32EnumAttrCase<"ClaimControlTask", 410, "claim_control_task">,
  I32EnumAttrCase<"ReleaseControlTask", 411, "release_control_task">,
  // Number of free resources of the kind its ID type names.
  I32EnumAttrCase<"CountFree", 412, "count_free">,
]> {
  let cppNamespace = "::mlir::csl";
}

//===----------------------------------------------------------------------===//
// Comptime Value Attributes
//===----------------------------------------------------------------------===//

def CSL_ComptimeIntAttr : CSL_Attr<"ComptimeInt", "comptime_int",
    [TypedAttrInterface]> {
  let summary = "Compile-time integer with arbitrary precision";
  let description = [{
    Represents an integer literal during parsing with arbitrary precision.
    These values are computed at compile time and can exceed the width of
    any runtime integer type.

    During parsing, all integer literals create ComptimeIntAttr. During
    comptime evaluation, these may be converted to csl.int attributes
    with specific widths for runtime use.

    Examples:
    ```mlir
    #csl.comptime_int<42>
    #csl.comptime_int<12345678901234567890>
    #csl.comptime_int<-100>
    ```
  }];

  let parameters = (ins APIntParameter<"">:$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    // Get as signed 64-bit integer (asserts if doesn't fit)
    int64_t getInt() const { return getValue().getSExtValue(); }

    // TypedAttrInterface - get the type this attribute represents
    Type getType() const { return ComptimeIntType::get(getContext()); }
  }];
}

def CSL_ComptimeFloatAttr : CSL_Attr<"ComptimeFloat", "comptime_float",
    [TypedAttrInterface]> {
  let summary = "Compile-time floating-point value";
  let description = [{
    Represents a floating-point literal during parsing with double precision.
    These values are computed at compile time and may later be converted to
    specific runtime float types (f32, f16, etc.).

    During parsing, all float literals create ComptimeFloatAttr. During
    comptime evaluation, these may be converted to csl.fp attributes
    with specific kinds for runtime use.

    Examples:
    ```mlir
    #csl.comptime_float<3.14>
    #csl.comptime_float<-0.5>
    #csl.comptime_float<2.71828>
    ```
  }];

  let parameters = (ins APFloatParameter<"">:$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    // Get as double precision value
    double getValueAsDouble() const {
      return getValue().convertToDouble();
    }

    // TypedAttrInterface - get the type this attribute represents
    Type getType() const { return ComptimeFloatType::get(getContext()); }
  }];
}

def CSL_ComptimeStringAttr : CSL_Attr<"ComptimeString", "comptime_string",
    [TypedAttrInterface]> {
  let summary = "Compile-time string value";
  let description = [{
    Represents a string literal during parsing.
    These values are computed at compile time and can be used in
    comptime expressions.

    Examples:
    ```mlir
    #csl.comptime_string<"Hello, World!">
    #csl.comptime_string<"CSL Language">
    ```
  }];

  let parameters = (ins StringRefParameter<"">:$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    // TypedAttrInterface - get the type this attribute represents
    Type getType() const { return ComptimeStringType::get(getContext()); }
  }];
}

def CSL_TypeAttr : CSL_Attr<"Type", "type",
    [TypedAttrInterface]> {
  let summary = "Type value attribute";
  let description = [{
    Represents a type as a first-class value in CSL. This is used when
    types are passed as compile-time arguments or stored in variables.

    Unlike MLIR's built-in TypeAttr, this conforms to TypedAttr interface
    and has TypeType as its type, making it compatible with the rest of the
    CSL infrastructure.

    Examples:
    ```mlir
    #csl.type<!csl.int<32, signed>>
    #csl.type<!csl.comptime_int>
    ```
  }];

  let parameters = (ins "Type":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$value), [{
      assert(value && "invalid type value");
      return $_get(value.getContext(), value);
    }]>
  ];

  let assemblyFormat = "$value";

  let extraClassDeclaration = [{
    // TypedAttrInterface - returns TypeType as this attribute's type
    Type getType() const { return TypeType::get(getContext()); }
  }];
}

def CSL_VoidAttr : CSL_Attr<"Void", "void", [TypedAttrInterface] > {
  let summary = "Void type attribute";
  let description = [{
    Represents the void type as a first-class value in CSL.

    Examples:
    ```mlir
    #csl.void
    ```
  }];

  let assemblyFormat = "";

  let extraClassDeclaration = [{
    // TypedAttrInterface - returns VoidType as this attribute's type
    Type getType() const { return VoidType::get(getContext()); }
  }];
}

def CSL_ColorAttr : CSL_Attr<"Color", "color", [TypedAttrInterface]> {
  let summary = "Color value attribute";
  let description = [{
    Represents a color ID value in CSL. Colors are used for routing and
    communication between PEs (Processing Elements).

    The value is stored as a 16-bit unsigned integer.

    Examples:
    ```mlir
    #csl.color<0>
    #csl.color<42>
    ```
  }];

  let parameters = (ins "uint16_t":$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    // TypedAttrInterface - returns ColorType as this attribute's type
    Type getType() const { return ColorType::get(getContext()); }
  }];
}

def CSL_LocalTaskIdAttr : CSL_Attr<"LocalTaskId", "local_task_id", [TypedAttrInterface]> {
  let summary = "Local task ID value attribute";
  let description = [{
    Represents a local task ID value in CSL.
    The value is stored as a 16-bit unsigned integer.

    Examples:
    ```mlir
    #csl.local_task_id<0>
    #csl.local_task_id<5>
    ```
  }];

  let parameters = (ins "uint16_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return LocalTaskIdType::get(getContext()); }
  }];
}

def CSL_DataTaskIdAttr : CSL_Attr<"DataTaskId", "data_task_id", [TypedAttrInterface]> {
  let summary = "Data task ID value attribute";
  let description = [{
    Represents a data task ID value in CSL. A data task ID is created from
    either a color (WSE-2) or an input_queue (WSE-3).

    The source attribute stores the ColorAttr or InputQueueAttr used to
    create this data task ID, enabling proper emission as CSL source.

    Examples:
    ```mlir
    #csl.data_task_id<#csl.color<0>>           // from color (WSE-2)
    #csl.data_task_id<#csl.input_queue<3>>     // from input_queue (WSE-3)
    ```
  }];

  let parameters = (ins "TypedAttr":$source);
  let assemblyFormat = "`<` $source `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DataTaskIdType::get(getContext()); }

    /// Check if this was created from a color (WSE-2 style)
    bool isFromColor() const;

    /// Check if this was created from an input_queue (WSE-3 style)
    bool isFromInputQueue() const;

    /// Get the color if this was created from one, or nullptr
    ColorAttr getColor() const;

    /// Get the input queue if this was created from one, or nullptr
    InputQueueAttr getInputQueue() const;
  }];
}

def CSL_ControlTaskIdAttr : CSL_Attr<"ControlTaskId", "control_task_id", [TypedAttrInterface]> {
  let summary = "Control task ID value attribute";
  let description = [{
    Represents a control task ID value in CSL.
    The value is stored as a 16-bit unsigned integer.

    Examples:
    ```mlir
    #csl.control_task_id<0>
    #csl.control_task_id<5>
    ```
  }];

  let parameters = (ins "uint16_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return ControlTaskIdType::get(getContext()); }
  }];
}

def CSL_UtIdAttr : CSL_Attr<"UtId", "ut_id", [TypedAttrInterface]> {
  let summary = "Microthread ID value attribute (WSE-3)";
  let description = [{
    Represents a microthread ID value in CSL (WSE-3 only).
    The value is stored as a 16-bit unsigned integer.

    Examples:
    ```mlir
    #csl.ut_id<0>
    #csl.ut_id<42>
    ```
  }];

  let parameters = (ins "uint16_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return UtIdType::get(getContext()); }
  }];
}

def CSL_DirectionAttr : CSL_Attr<"Direction", "direction", [TypedAttrInterface]> {
  let summary = "Routing direction value attribute";
  let description = [{
    Represents a routing direction value in CSL. Valid directions are
    NORTH, SOUTH, EAST, WEST, and RAMP.

    Examples:
    ```mlir
    #csl.direction<NORTH>
    #csl.direction<RAMP>
    ```
  }];

  let parameters = (ins "Direction":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DirectionType::get(getContext()); }
  }];
}

def CSL_RangeAttr : CSL_Attr<"Range", "range", [TypedAttrInterface]> {
  let summary = "Compile-time range value";
  let description = [{
    Represents a range value at compile time, containing start, stop, and step.
    Values are stored as int64_t and must fit within the element type's range.

    Examples:
    ```mlir
    #csl.range<i32, 0, 10, 1>
    #csl.range<u16, 5, 100, 5>
    ```
  }];

  let parameters = (ins
    "Type":$element_type,
    "int64_t":$start,
    "int64_t":$stop,
    "int64_t":$step
  );

  let assemblyFormat = "`<` $element_type `,` $start `,` $stop `,` $step `>`";

  let extraClassDeclaration = [{
    Type getType() const { return RangeType::get(getElementType()); }
  }];
}

//===----------------------------------------------------------------------===//
// DSR Value Attributes
//===----------------------------------------------------------------------===//

def CSL_DsrDestAttr : CSL_Attr<"DsrDest", "dsr_dest", [TypedAttrInterface]> {
  let summary = "DSR destination value attribute";
  let description = [{
    Represents a DSR destination value. The value is the DSR register ID.

    Examples:
    ```mlir
    #csl.dsr_dest<0>
    #csl.dsr_dest<4>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DsrDestType::get(getContext()); }
  }];
}

def CSL_DsrSrc0Attr : CSL_Attr<"DsrSrc0", "dsr_src0", [TypedAttrInterface]> {
  let summary = "DSR src0 value attribute";
  let description = [{
    Represents a DSR src0 value. The value is the DSR register ID.

    Examples:
    ```mlir
    #csl.dsr_src0<0>
    #csl.dsr_src0<3>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DsrSrc0Type::get(getContext()); }
  }];
}

def CSL_DsrSrc1Attr : CSL_Attr<"DsrSrc1", "dsr_src1", [TypedAttrInterface]> {
  let summary = "DSR src1 value attribute";
  let description = [{
    Represents a DSR src1 value. The value is the DSR register ID.

    Examples:
    ```mlir
    #csl.dsr_src1<0>
    #csl.dsr_src1<5>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DsrSrc1Type::get(getContext()); }
  }];
}

def CSL_DsrFifoDestAttr : CSL_Attr<"DsrFifoDest", "dsr_fifo_dest", [TypedAttrInterface]> {
  let summary = "FIFO DSR destination value attribute";
  let description = [{
    Represents a FIFO DSR destination value. The value is the DSR register ID.

    Examples:
    ```mlir
    #csl.dsr_fifo_dest<0>
    #csl.dsr_fifo_dest<2>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DsrFifoDestType::get(getContext()); }
  }];
}

def CSL_DsrFifoSrc1Attr : CSL_Attr<"DsrFifoSrc1", "dsr_fifo_src1", [TypedAttrInterface]> {
  let summary = "FIFO DSR src1 value attribute";
  let description = [{
    Represents a FIFO DSR src1 value. The value is the DSR register ID.

    Examples:
    ```mlir
    #csl.dsr_fifo_src1<0>
    #csl.dsr_fifo_src1<3>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return DsrFifoSrc1Type::get(getContext()); }
  }];
}

def CSL_XdsrAttr : CSL_Attr<"Xdsr", "xdsr", [TypedAttrInterface]> {
  let summary = "XDSR value attribute";
  let description = [{
    Represents an extended DSR (XDSR) value. The value is the XDSR register ID.

    Examples:
    ```mlir
    #csl.xdsr<0>
    #csl.xdsr<1>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return XdsrType::get(getContext()); }
  }];
}

def CSL_SrAttr : CSL_Attr<"Sr", "sr", [TypedAttrInterface]> {
  let summary = "stride-register value attribute";
  let description = [{
    Represents an SR value. The value is the physical stride-register ID.
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return SrType::get(getContext()); }
  }];
}

//===----------------------------------------------------------------------===//
// Runtime Value Attributes (using MLIR builtins)
//===----------------------------------------------------------------------===//
// Note: Runtime integer, float, and boolean attributes are now represented using
// MLIR builtin attributes:
// - BoolAttr (MLIR builtin) for boolean values (true, false)
// - IntegerAttr (MLIR builtin) for runtime integers (42 : si32, 255 : ui8)
// - FloatAttr (MLIR builtin) for runtime floats (3.14 : f32)

def CSL_UndefAttr : CSL_Attr<"Undef", "undef"> {
  let summary = "Undefined value attribute";
  let description = [{
    Represents an undefined/uninitialized value.

    Example:
    ```mlir
    #csl.undef
    ```
  }];

  let assemblyFormat = "";
}

def CSL_OpaqueAttr : CSL_Attr<"Opaque", "opaque"> {
  let summary = "Opaque attribute emitted as-is";
  let description = [{
    Represents an opaque attribute that is emitted as-is in the generated code.

    Example:
    ```mlir
    #csl.opaque<"special_value">
    ```
  }];

  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
}

def CSL_LValueAttr : CSL_Attr<"LValue", "lvalue",
    [TypedAttrInterface]> {
  let summary = "LValue type attribute with mutability";
  let description = [{
    Represents the (known) value of a const lvalue.
    Used by folders of VariableOps and LoadOps to track known values of immutable
    variables.

    This unified attribute works in both ParsedCSL and TypedCSL stages.

    Examples:
    ```mlir
    #csl.lvalue<#csl.int<42 : 32, signed>>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$rvalue
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$rvalue), [{
      assert(rvalue && "invalid rvalue");
      return $_get(rvalue.getContext(), rvalue);
    }]>
  ];

  let assemblyFormat = "`<` $rvalue `>`";

  let extraClassDeclaration = [{
    /// Returns the const LValueType wrapping the value's type.
    LValueType getType() const {
      return LValueType::get(getRvalue().getType(), /*isConst=*/true);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Struct Value Attribute
//===----------------------------------------------------------------------===//

def CSL_StructValueAttr : CSL_Attr<"StructValue", "struct_value",
    [TypedAttrInterface]> {
  let summary = "Struct value attribute (for comptime evaluation)";
  let description = [{
    Represents a struct value at compile time. Stores field values, optional
    field names, and an optional definition location. Its concrete structural
    or nominal `StructType` is derived from that information.

    Used during comptime evaluation to store struct values as attributes.

    Examples:
    ```mlir
    // Anonymous struct with type !csl.struct<x:si32,y:si32>
    #csl.struct_value<{x:si32,y:si32}, [#csl.comptime_int<10>, #csl.comptime_int<20>]>

    // Tuple with type !csl.struct<si32,f32>
    #csl.struct_value<{si32,f32}, [#csl.comptime_int<10>, #csl.comptime_float<3.14>]>

    // Named struct with location
    #csl.struct_value<{{loc(...)},real:f16,imag:f16}, [...]>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"TypedAttr">:$field_values,         // always present
    OptionalParameter<"LocationAttr">:$def_location,      // for nominal typing
    OptionalArrayRefParameter<"StringAttr">:$field_names   // nullopt for tuples
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Check if this is a tuple (no field names)
    bool isTuple() const { return getFieldNames().empty(); }

    // Check if this is an anonymous struct or tuple (no location)
    bool isAnonymous() const {
      return !getDefLocation();
    }

    // Get number of fields
    size_t getNumFields() const { return getFieldValues().size(); }

    // Get field value by name (returns null if not found or is tuple)
    TypedAttr getField(StringRef name) const;

    // Get field value by index
    TypedAttr getField(size_t index) const {
      assert(index < getNumFields() && "field index out of range");
      return getFieldValues()[index];
    }

    // Get field index by name (returns failure if not found or is tuple)
    FailureOr<size_t> getFieldIndex(StringRef name) const;

    // Get field name by index (returns failure if tuple)
    FailureOr<StringRef> getFieldName(size_t index) const {
      if (getFieldNames().empty())
        return failure();
      assert(index < getNumFields() && "field index out of range");
      return getFieldNames()[index].getValue();
    }

    // The concrete structural or nominal type inferred from the fields.
    Type getType();

    // Update a field value, returning a new StructValueAttr as attributes are immutable in MLIR
    static StructValueAttr updateField(StructValueAttr base, StringRef fieldName,
                                        TypedAttr newValue);

    static StructValueAttr updateField(StructValueAttr base, size_t index,
                                        TypedAttr newValue);
  }];
}

//===----------------------------------------------------------------------===//
// Array Value Attribute
//===----------------------------------------------------------------------===//

def CSL_ArrayValueAttr : CSL_Attr<"ArrayValue", "array_value",
    [TypedAttrInterface]> {
  let summary = "Array value attribute (for comptime evaluation)";
  let description = [{
    Represents a multi-dimensional array value at compile time. Stores elements
    in row-major order as a flat array.

    Used during comptime evaluation to store array values as attributes.

    Examples:
    ```mlir
    // 1D array [3]si32
    #csl.array_value<[3] x si32, [1 : si32, 2 : si32, 3 : si32]>

    // 2D array [2, 3]f16
    #csl.array_value<[2, 3] x f16, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$element_type,
    ArrayRefParameter<"TypedAttr">:$elements
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Get the ArrayType this attribute represents
    ArrayType getType() const;

    // Get total number of elements
    int64_t getNumElements() const;

    // Get element at linear index
    TypedAttr getElement(int64_t linearIndex) const;

    // Get element at multi-dimensional indices
    TypedAttr getElement(ArrayRef<int64_t> indices) const;

    // Convert multi-dimensional indices to linear index (row-major)
    int64_t linearizeIndices(ArrayRef<int64_t> indices) const;

    // Update element at linear index, returning new ArrayValueAttr
    static ArrayValueAttr updateElement(ArrayValueAttr base, int64_t linearIndex,
                                        TypedAttr newValue);
  }];
}

//===----------------------------------------------------------------------===//
// Enum Value Attribute
//===----------------------------------------------------------------------===//

def CSL_EnumValueAttr : CSL_Attr<"EnumValue", "enum_value", [TypedAttrInterface]> {
  let summary = "Enum value attribute (for comptime evaluation)";
  let description = [{
    Represents an enum value at compile time. Stores the enum type and the
    specific element name and integer value.

    Used during comptime evaluation to store enum values as attributes.

    Examples:
    ```mlir
    #csl.enum_value<"red", 0 : !csl.enum<...>>
    ```
  }];

  let parameters = (ins
    "EnumType":$enum_type,
    "StringAttr":$element_name,
    "int64_t":$int_value
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "EnumType":$enumType,
                                        "StringAttr":$elementName,
                                        "int64_t":$intValue), [{
      return $_get(enumType.getContext(), enumType, elementName, intValue);
    }]>
  ];

  let extraClassDeclaration = [{
    // TypedAttrInterface - return the enum type
    Type getType() const { return getEnumType(); }
  }];
}

def CSL_NoValueAttr : CSL_Attr<"NoValue", "no_value", [TypedAttrInterface]> {
  let summary = "Type-only marker attribute with no associated value";
  let description = [{
    Represents a typed slot with no associated value. Used in contexts like
    generic function type bodies where we need to reason about types (e.g.,
    @type_of(someArg)) without having actual values.

    Example:
    ```mlir
    #csl.no_value<i32>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $type `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>,
  ];
}

def CSL_FuncRefAttr : CSL_Attr<"FuncRef", "func_ref",
    [TypedAttrInterface]> {
  let summary = "Function reference attribute";
  let description = [{
    Represents a reference to a function symbol by name, optionally qualified
    by a module identifier.

    If module_identifier is absent (nullopt), this is a top-level function.
    If present, it's the unique module ID for functions defined inside a module.

    The type can be either:
    - FunctionType: for concrete (non-generic) functions
    - GenericFunctionType: for generic functions (opaque marker)

    After ComptimeEvalPass, all FuncRefAttr should have concrete FunctionType.
    For async functions, the FunctionType returns FutureType as its result
    (e.g., `(args) -> !csl.future<@foo>`), matching the function's
    `function_type` attribute.

    Examples:
    ```mlir
    #csl.func_ref<"my_function">             // top-level function
    #csl.func_ref<123, "bar">                // function in module with ID 123
    ```
  }];

  let parameters = (ins
    OptionalParameter<"std::optional<uint32_t>">:$module_identifier,
    "mlir::FlatSymbolRefAttr":$symbol_name,
    AttributeSelfTypeParameter<"", "::mlir::Type">:$type);

  let assemblyFormat = "`<` (`module` $module_identifier^ `,`)? $symbol_name `>`";

  let extraClassDeclaration = [{
    /// Returns true if this is a top-level function (no module qualifier)
    bool isTopLevel() const { return !getModuleIdentifier().has_value(); }

    /// Returns true if this references a generic function
    bool isGeneric() const;

    /// Get the type as FunctionType (asserts if GenericFunctionType)
    FunctionType getFunctionType() const;
  }];
}

def CSL_FuncAddrAttr : CSL_Attr<"FuncAddr", "func_addr",
    [TypedAttrInterface]> {
  let summary = "Function address attribute";
  let description = [{
    Represents the address of a function identified by a FuncRefAttr.

    Examples:
    ```mlir
    #csl.func_addr<#csl.func_ref<"my_function">>
    ```
  }];

  let parameters = (ins
    "::mlir::csl::FuncRefAttr":$func_ref);

  let assemblyFormat = "`<` $func_ref `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::csl::FuncRefAttr":$func_ref), [{
      return $_get(func_ref.getContext(), func_ref);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the FunctionType of the referenced function
    csl::PointerType getType() const {
      auto funcType = getFuncRef().getType();
      return csl::PointerType::get(funcType, /*isConst=*/true, /*isArray=*/false);
    }
  }];
}

def CSL_GlobalRefAttr : CSL_Attr<"GlobalRef", "global_ref",
    [TypedAttrInterface]> {
  let summary = "Global variable reference attribute";
  let description = [{
    Represents a reference to a global variable symbol by name, optionally
    qualified by a module identifier.

    If module_identifier is absent (nullopt), this is a top-level global.
    If present, it's the unique module ID for globals defined inside a module.

    Examples:
    ```mlir
    #csl.global_ref<"my_global">             // top-level global
    #csl.global_ref<module 123, "bar">       // global in module with ID 123
    ```
  }];

  let parameters = (ins
    OptionalParameter<"std::optional<uint32_t>">:$module_identifier,
    "mlir::FlatSymbolRefAttr":$symbol_name,
    AttributeSelfTypeParameter<"", "::mlir::csl::LValueType">:$type);

  let assemblyFormat = "`<` (`module` $module_identifier^ `,`)? $symbol_name `>`";

  let extraClassDeclaration = [{
    /// Returns true if this is a top-level global (no module qualifier)
    bool isTopLevel() const { return !getModuleIdentifier().has_value(); }
  }];
}

def CSL_GlobalAddrAttr : CSL_Attr<"GlobalAddr", "global_addr",
    [TypedAttrInterface]> {
  let summary = "Global variable address attribute";
  let description = [{
    Represents the address of a global variable identified by a GlobalRefAttr.

    Examples:
    ```mlir
    #csl.global_addr<#csl.global_ref<"my_global">>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "csl::PointerType">:$type,
    "::mlir::csl::GlobalRefAttr":$global_ref);

  let assemblyFormat = "`<` $global_ref `>`";

  let builders = [
    // Builder with explicit pointer type. Can be used if the type was casted.
    AttrBuilderWithInferredContext<(ins "csl::PointerType":$type, "::mlir::csl::GlobalRefAttr":$global_ref), [{
      return $_get(type.getContext(), type, global_ref);
    }]>,
    // Builder that infers pointer type from the global ref's lvalue type
    AttrBuilderWithInferredContext<(ins "::mlir::csl::GlobalRefAttr":$global_ref), [{
      auto lvalueType = global_ref.getType();
      auto pointerType = csl::PointerType::get(lvalueType.getValueType(),
                                              /*isConst=*/lvalueType.getIsConst(),
                                              /*isArray=*/false);
      return $_get(global_ref.getContext(), pointerType, global_ref);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DSD Supporting Attributes
//===----------------------------------------------------------------------===//

def CSL_InputQueueAttr : CSL_Attr<"InputQueue", "input_queue", [TypedAttrInterface]> {
  let summary = "Input queue ID attribute";
  let description = [{
    Represents an input queue ID for fabric DSDs.

    Examples:
    ```mlir
    #csl.input_queue<0>
    #csl.input_queue<3>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return InputQueueType::get(getContext()); }
  }];
}

def CSL_OutputQueueAttr : CSL_Attr<"OutputQueue", "output_queue", [TypedAttrInterface]> {
  let summary = "Output queue ID attribute";
  let description = [{
    Represents an output queue ID for fabric DSDs.

    Examples:
    ```mlir
    #csl.output_queue<0>
    #csl.output_queue<3>
    ```
  }];

  let parameters = (ins "uint8_t":$value);
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    Type getType() const { return OutputQueueType::get(getContext()); }
  }];
}

def CSL_PriorityAttr : CSL_Attr<"Priority", "priority", [TypedAttrInterface]> {
  let summary = "Microthread priority attribute";
  let description = [{
    Represents a microthread priority level for asynchronous DSD operations.
    Priority levels are: high, medium, low.

    Examples:
    ```mlir
    #csl.priority<high>
    #csl.priority<medium>
    #csl.priority<low>
    ```
  }];

  let parameters = (ins "PriorityLevel":$level);
  let assemblyFormat = "`<` $level `>`";

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

def CSL_SimdModeAttr : CSL_Attr<"SimdMode", "simd_mode", [TypedAttrInterface]> {
  let summary = "SIMD mode attribute";
  let description = [{
    Represents a SIMD mode for fabric DSDs.
    Modes include the WSE-2 modes simd_16, simd_32, simd_64, and
    simd_32_or_64, plus the WSE-3 modes simd_off and simd_max.

    Examples:
    ```mlir
    #csl.simd_mode<simd_32>
    #csl.simd_mode<simd_64>
    ```
  }];

  let parameters = (ins "SimdModeKind":$mode);
  let assemblyFormat = "`<` $mode `>`";

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

def CSL_ZeroAttr : CSL_Attr<"Zero", "zero", [TypedAttrInterface]> {
  let summary = "Zero attribute for DSD source reset";
  let description = [{
    Represents which source operand to reset to zero after a DSD operation.
    Options are: first_source, second_source.

    Examples:
    ```mlir
    #csl.zero<first_source>
    #csl.zero<second_source>
    ```
  }];

  let parameters = (ins "ZeroSource":$source);
  let assemblyFormat = "`<` $source `>`";

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// DSD (Data Structure Descriptor) Attributes
//===----------------------------------------------------------------------===//

def CSL_Mem1dDsdAttr : CSL_Attr<"Mem1dDsd", "mem1d_dsd", [TypedAttrInterface]> {
  let summary = "1D memory DSD attribute";
  let description = [{
    Represents a comptime-known 1D memory Data Structure Descriptor.
    Used for describing 1D memory access patterns with a single induction variable.

    Examples:
    ```mlir
    #csl.mem1d_dsd<#csl.global_addr<...>, extent = 10, stride = 1, offset = 0>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$base_address,
    "uint16_t":$extent,
    "int8_t":$stride,
    "int16_t":$offset,
    "bool":$wavelet_index_offset
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Type getType() const { return Mem1dDsdType::get(getContext()); }
  }];
}

def CSL_Mem4dDsdAttr : CSL_Attr<"Mem4dDsd", "mem4d_dsd", [TypedAttrInterface]> {
  let summary = "Multi-dimensional memory DSD attribute (up to 4D)";
  let description = [{
    Represents a comptime-known multi-dimensional memory Data Structure Descriptor.
    Used for describing memory access patterns with up to 4 induction variables.

    Examples:
    ```mlir
    #csl.mem4d_dsd<#csl.global_addr<...>, extents = [10, 20], strides = [1, 10], offset = 0>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$base_address,
    ArrayRefParameter<"uint16_t">:$extents,
    ArrayRefParameter<"int16_t">:$strides,
    "int16_t":$offset,
    "bool":$wavelet_index_offset
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Type getType() const { return Mem4dDsdType::get(getContext()); }
  }];
}

def CSL_FabinDsdAttr : CSL_Attr<"FabinDsd", "fabin_dsd", [TypedAttrInterface]> {
  let summary = "Fabric input DSD attribute";
  let description = [{
    Represents a comptime-known fabric input Data Structure Descriptor.
    Used for receiving wavelets from the fabric.

    Examples:
    ```mlir
    #csl.fabin_dsd<extent = 10, input_queue = #csl.input_queue<0>>
    ```
  }];

  let parameters = (ins
    OptionalParameter<"ColorAttr">:$fabric_color,
    "uint16_t":$extent,
    OptionalParameter<"InputQueueAttr">:$input_queue,
    OptionalParameter<"PriorityAttr">:$priority,
    "bool":$control_transform,
    OptionalParameter<"SimdModeAttr">:$simd_mode
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Type getType() const { return FabinDsdType::get(getContext()); }
  }];
}

def CSL_FaboutDsdAttr : CSL_Attr<"FaboutDsd", "fabout_dsd", [TypedAttrInterface]> {
  let summary = "Fabric output DSD attribute";
  let description = [{
    Represents a comptime-known fabric output Data Structure Descriptor.
    Used for sending wavelets to the fabric.

    Examples:
    ```mlir
    #csl.fabout_dsd<extent = 10, output_queue = #csl.output_queue<0>>
    ```
  }];

  let parameters = (ins
    OptionalParameter<"ColorAttr">:$fabric_color,
    "uint16_t":$extent,
    OptionalParameter<"OutputQueueAttr">:$output_queue,
    "bool":$control,
    "bool":$wavelet_index_offset,
    "bool":$advance_switch,
    "bool":$control_transform,
    OptionalParameter<"SimdModeAttr">:$simd_mode,
    OptionalParameter<"ZeroAttr">:$zero
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Type getType() const { return FaboutDsdType::get(getContext()); }
  }];
}

def CSL_FifoDsdAttr : CSL_Attr<"FifoDsd", "fifo_dsd", [TypedAttrInterface]> {
  let summary = "FIFO DSD attribute";
  let description = [{
    Represents a comptime-known FIFO Data Structure Descriptor.
    Created via @allocate_fifo with a buffer and optional configuration.

    Examples:
    ```mlir
    #csl.fifo_dsd<#csl.global_ref<"buffer">>
    ```
  }];

  let parameters = (ins
    "GlobalRefAttr":$buffer,
    OptionalParameter<"FuncRefAttr">:$activate_pop,
    OptionalParameter<"FuncRefAttr">:$activate_push,
    OptionalParameter<"PriorityAttr">:$priority,
    OptionalParameter<"SimdModeAttr">:$simd_mode
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Type getType() const { return FifoDsdType::get(getContext()); }
  }];
}

#endif // CSL_ATTRS
