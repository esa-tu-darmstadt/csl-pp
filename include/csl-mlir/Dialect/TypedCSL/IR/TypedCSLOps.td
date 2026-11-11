//===- TypedCSLOps.td - TypedCSL Operation Definitions ----*- tablegen -*-===//
//
// Operation definitions for the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_CSL_OPS
#define TYPED_CSL_OPS

include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLBase.td"
include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLTypes.td"
include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLAttrs.td"
include "csl-mlir/Dialect/CSL/IR/CSLAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "csl-mlir/Dialect/CSL/IR/CSLOpInterfaces.td"
include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class for TypedCSL Operations
//===----------------------------------------------------------------------===//

class TypedCSL_Op<string mnemonic, list<Trait> traits = []>
    : Op<TypedCSL_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// ConstantOp - Typed constant values
//===----------------------------------------------------------------------===//

def TypedCSL_ConstantOp : TypedCSL_Op<"constant", [
  ConstantLike, Pure, AllTypesMatch<["value", "result"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Typed constant operation";
  let description = [{
    The `typed_csl.constant` operation produces a runtime-typed constant value.

    Unlike ParsedCSL constants which may have comptime types, TypedCSL constants
    always have concrete runtime types (IntType, FloatType, BoolType, etc.).

    Example:
    ```mlir
    %0 = typed_csl.constant 42 : si32
    %1 = typed_csl.constant 3.14 : f32
    %2 = typed_csl.constant true
    %3 = typed_csl.constant #csl.comptime_int<42> : !csl.comptime_int
    ```

    References to functions and globals are constants as well, so that a
    reference has the same form whether it stands alone or is folded into an
    aggregate or a global initializer:

    ```mlir
    // The function @myFunc, as passed to builtins like @export_symbol
    %f = typed_csl.constant #csl.func_ref<@myFunc> : () -> ()
    // The lvalue of the global variable @foo
    %x = typed_csl.constant #csl.global_ref<@foo> : !csl.lvalue<mutable f32>
    // The address of @myFunc, a function pointer
    %p = typed_csl.constant #csl.func_addr<<@myFunc>> : !csl.ptr<const () -> ()>
    ```

    Every function and global referenced in the value, at any nesting depth,
    must exist and have the type the reference carries.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value";

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BinaryOp - Binary operations with folder helpers
//===----------------------------------------------------------------------===//

def TypedCSL_BinaryOp : TypedCSL_Op<"binary", [Pure, InferTypeOpInterface]> {
  let summary = "Binary operation on typed values";
  let description = [{
    The `typed_csl.binary` operation performs binary operations on fully-typed operands.
    Both operands must have the same concrete type, and the result type is determined
    by the operation kind.

    Supported operations:
    - Arithmetic: add, sub, mul, div, mod
    - Bitwise: and, or, xor, shl, shr
    - Comparison: eq, ne, lt, le, gt, ge (return bool)
    - Logical: land, lor (bool operands, return bool)

    Example:
    ```mlir
    %sum = typed_csl.binary add, %a, %b : (si32, si32) -> si32
    %cmp = typed_csl.binary eq, %x, %y : (si32, si32) -> i1
    ```
  }];

  let arguments = (ins
    CSL_BinaryOpKind:$opcode,
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $opcode `,` $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Interface method for InferTypeOpInterface.
    /// Infers the result types from operands and attributes.
    static ::mlir::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context,
        ::std::optional<::mlir::Location> location,
        ::mlir::ValueRange operands,
        ::mlir::DictionaryAttr attributes,
        ::mlir::PropertyRef properties,
        ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);

  }];
}

//===----------------------------------------------------------------------===//
// UnaryOp - Unary operations with folder helpers
//===----------------------------------------------------------------------===//

def TypedCSL_UnaryOp : TypedCSL_Op<"unary", [
    Pure,
    AllTypesMatch<["operand", "result"]>
  ]> {
  let summary = "Unary operation on typed values";
  let description = [{
    The `typed_csl.unary` operation performs unary operations on fully-typed operands.

    Supported operations:
    - neg: Arithmetic negation (-x)
    - bitnot: Bitwise complement (~x)
    - lnot: Logical negation (!x)

    Example:
    ```mlir
    %negated = typed_csl.unary neg, %value : si32 -> si32
    %inverted = typed_csl.unary bitnot, %bits : ui32 -> ui32
    %negbool = typed_csl.unary lnot, %bool_val : i1 -> i1
    ```
  }];

  let arguments = (ins
    CSL_UnaryOpKind:$opcode,
    AnyType:$operand
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $opcode `,` $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

}

//===----------------------------------------------------------------------===//
// CastOp - Type conversion with folder helpers
//===----------------------------------------------------------------------===//

def TypedCSL_CastOp : TypedCSL_Op<"cast", [Pure, CastOpInterface]> {
  let summary = "Type conversion operation";
  let description = [{
    The `typed_csl.cast` operation converts a value from one concrete type to another.
    All types must be fully resolved runtime types.

    Supported casts:
    - Integer to integer (different widths/signedness)
    - Float to float (different precisions)
    - Integer to float
    - Float to integer (truncating)

    Example:
    ```mlir
    // Widen i16 to i32
    %wide = typed_csl.cast %narrow : si16 to si32

    // Convert int to float
    %f = typed_csl.cast %i : si32 to f32
    ```
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$dest);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($dest)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Interface method for CastOpInterface.
    /// Returns true if the given set of input and result types are compatible to cast.
    static bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);

  }];
}

def TypedCSL_ConstCastOp : TypedCSL_Op<"const_cast", [Pure,
  TypesMatchWith<"value types must match",
                 "source", "result",
                 "::mlir::csl::LValueType::get("
                 "::llvm::cast<::mlir::csl::LValueType>($_self).getValueType(),"
                 "!::llvm::cast<::mlir::csl::LValueType>($_self).getIsConst())">
]> {
  let summary = "Cast between const and mutable lvalue";
  let description = [{
    Reinterprets an lvalue's constness without changing the underlying value
    type. Supports both directions:
    - `!csl.lvalue<mutable T>` → `!csl.lvalue<const T>` (safe, add const)
    - `!csl.lvalue<const T>` → `!csl.lvalue<mutable T>` (unsafe, remove const)

    Example:
    ```mlir
    %const_ref = typed_csl.const_cast %mutable_ref
        : !csl.lvalue<mutable si32> to !csl.lvalue<const si32>
    ```
  }];

  let arguments = (ins TypedCSL_LValueType:$source);
  let results = (outs TypedCSL_LValueType:$result);

  let assemblyFormat = [{
    $source attr-dict `:` qualified(type($source)) `to` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Variable Operations
//===----------------------------------------------------------------------===//

def TypedCSL_VariableOp : TypedCSL_Op<"var", [
  PredOpTrait<"result type isConst matches is_const attribute",
    CPred<"::llvm::cast<::mlir::csl::LValueType>($result.getType()).getIsConst() == "
          "static_cast<bool>(getIsConst())">>,
  OptionalTypesMatchWith<"initializer type matches lvalue value type",
                         "result", "initializer",
                         "::llvm::cast<::mlir::csl::LValueType>($_self).getValueType()">
]> {
  let summary = "Variable declaration";
  let description = [{
    Declares a typed variable with optional initializer.

    - If `initializer` is provided, the variable is initialized with that value
    - Variables can be marked as `const` (immutable)
    - Result is an lvalue that can be loaded from (and stored to if mutable)

    The result type's `isConst` parameter is automatically inferred from the `is_const` attribute.

    Examples:
    ```mlir
    // Variable with initializer - result type is !csl.lvalue<mutable si32>
    %init = typed_csl.constant 42 : si32
    %v1 = typed_csl.var "x" init(%init) : !csl.lvalue<mutable si32>

    // Const variable - result type is !csl.lvalue<const si32>
    %v2 = typed_csl.var "y" init(%init) const : !csl.lvalue<const si32>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    Optional<AnyType>:$initializer,
    UnitAttr:$is_const
  );

  let results = (outs TypedCSL_LValueType:$result);

  let assemblyFormat = [{
    $name (`init` `(` $initializer^ `)`)? (`const` $is_const^)? attr-dict `:` qualified(type($result))
  }];

  let hasFolder = 1;
}

def TypedCSL_LoadOp : TypedCSL_Op<"load", [
  TypesMatchWith<"result type matches lvalue value type",
                 "lvalue", "result",
                 "::llvm::cast<::mlir::csl::LValueType>($_self).getValueType()">
]> {
  let summary = "Load value from lvalue";
  let description = [{
    Loads the current value from an lvalue (variable).

    Not Pure: the load reads memory, so it must not be merged with or moved
    across an op that may write it. An unused load is still dead.

    Example:
    ```mlir
    %var = typed_csl.var "x" init(%c42) : !csl.lvalue<mutable si32>
    %val = typed_csl.load from %var : !csl.lvalue<mutable si32>
    ```
  }];

  let arguments = (ins Arg<TypedCSL_LValueType, "lvalue to read from", [MemRead]>:$lvalue);
  let results = (outs AnyType:$result);

  let assemblyFormat = "`from` $lvalue attr-dict `:` qualified(type($lvalue))";

  let hasFolder = 1;
}

def TypedCSL_AssignOp : TypedCSL_Op<"assign", [
  TypesMatchWith<"value type matches lvalue value type",
                 "lvalue", "value",
                 "::llvm::cast<::mlir::csl::LValueType>($_self).getValueType()">
]> {
  let summary = "Assign value to lvalue";
  let description = [{
    Assigns a value to an lvalue (variable).
    The lvalue must be mutable (not const).

    Example:
    ```mlir
    %var = typed_csl.var "x" init(%c42) : !csl.lvalue<mutable si32>
    %val = typed_csl.constant 100 : si32
    typed_csl.assign %val to %var : !csl.lvalue<mutable si32>
    ```
  }];

  let arguments = (ins
    Arg<TypedCSL_MutableLValueType, "lvalue to write to", [MemWrite]>:$lvalue,
    AnyType:$value
  );

  let assemblyFormat = "$value `to` $lvalue attr-dict `:` type($lvalue)";

  let hasCanonicalizer = 1;
}

def TypedCSL_InlineAsmOp : TypedCSL_Op<"inline_asm", [
    AttrSizedOperandSegments]> {
  let summary = "Inline assembly statement";
  let description = [{
    Represents a CSL inline assembly statement with concrete types.
    The asm_string is a resolved string attribute (not an SSA operand).
    Outputs are lvalue operands, inputs are rvalue operands.

    Example:
    ```mlir
    typed_csl.inline_asm "mov16 %[retval] = %[argval1]"
        outputs(%var : !csl.lvalue<mutable u16>)
        inputs(%val : u16)
        output_names = ["retval"]
        output_constraints = ["=r"]
        input_names = ["argval1"]
        input_constraints = ["r"]
        clobbers = ["r6"]
        is_volatile
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$outputs,
    Variadic<AnyType>:$inputs,
    StrAttr:$asm_string,
    StrArrayAttr:$output_names,
    StrArrayAttr:$output_constraints,
    StrArrayAttr:$input_names,
    StrArrayAttr:$input_constraints,
    StrArrayAttr:$clobbers,
    UnitAttr:$is_volatile
  );

  let hasVerifier = 1;
}

def TypedCSL_AddressOfOp : TypedCSL_Op<"address_of", [Pure]> {
  let summary = "Take address of an lvalue";
  let description = [{
    Takes the address of an lvalue and produces a pointer.

    The operand must be an lvalue type (!csl.lvalue<T>).
    Result type is !csl.ptr<T> or !csl.ptr<const T> depending on lvalue constness.

    Examples:
    ```mlir
    %var = typed_csl.var "x" init(%val) : !csl.lvalue<mutable si32>
    %ptr = typed_csl.address_of %var : !csl.lvalue<mutable si32> -> !csl.ptr<si32>

    %const_var = typed_csl.var "y" const init(%val) : !csl.lvalue<const si32>
    %const_ptr = typed_csl.address_of %const_var : !csl.lvalue<const si32> -> !csl.ptr<const si32>
    ```
  }];

  let arguments = (ins TypedCSL_LValueType:$operand);
  let results = (outs CSL_PointerType:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_DereferenceOp : TypedCSL_Op<"dereference", [Pure]> {
  let summary = "Dereference a pointer";
  let description = [{
    Dereferences a pointer to produce an lvalue to the pointed-to location.

    Operand must be a pointer type (!csl.ptr<T>).
    Result is an lvalue with constness matching the pointer's constness.

    Examples:
    ```mlir
    %ptr = ... : !csl.ptr<si32>
    %lval = typed_csl.dereference %ptr : !csl.ptr<si32> -> !csl.lvalue<mutable si32>

    %const_ptr = ... : !csl.ptr<const si32>
    %const_lval = typed_csl.dereference %const_ptr : !csl.ptr<const si32> -> !csl.lvalue<const si32>
    ```
  }];

  let arguments = (ins CSL_PointerType:$operand);
  let results = (outs TypedCSL_LValueType:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_GlobalOp : TypedCSL_Op<"global", [
  Symbol,
  SymbolName,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  PredOpTrait<"type of initializer matches variable type",
    CPred<"!$initializer || $type == $initializer->getType()">>,
  PredOpTrait<"const global has initializer (unless extern)",
    CPred<"!getIsConst() || $initializer || getIsExtern()">>,
  PredOpTrait<"extern global has no initializer",
    CPred<"!getIsExtern() || !$initializer">>,
  PredOpTrait<"export and extern are mutually exclusive",
    CPred<"!getIsExport() || !getIsExtern()">>
  ]> {
  let summary = "A global variable";
  let description = [{
    The `typed_csl.global` operation declares or defines a named global variable.
    The backing memory for the variable is allocated statically and described by
    the variable's type. Optionally, an `initial_value` can be provided.
    The global variable can also be marked constant using the `const_specifier`
    unit attribute.

    The global variable is accessed through a `typed_csl.constant` holding a
    `#csl.global_ref`, which yields its lvalue.

    Example:

    ```mlir
    // Global variable declaration without initial value.
    %0 = typed_csl.global @x : si32
    // Constant global variable with an initial value.
    %1 = typed_csl.global const @y : si32 = 0 : si32
    ```
  }];

  let arguments = (ins TypeAttr:$type,
                       SymbolNameAttr:$sym_name,
                       OptionalAttr<TypedAttrInterface>:$initializer,
                       UnitAttr:$is_const,
                       UnitAttr:$is_export,
                       UnitAttr:$is_extern,
                       OptionalAttr<StrAttr>:$linkname,
                       OptionalAttr<UI32Attr>:$alignment);
  let results = (outs);

  let assemblyFormat = [{
       (`export` $is_export^)?
       (`extern` $is_extern^)?
       (`const` $is_const^)?
       $sym_name
       `:` $type (`=` $initializer^)?
       (`linkname` `(` $linkname^ `)`)?
       (`alignment` `(` $alignment^ `)`)?
       attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::SymbolTable::Visibility getVisibility() {
      return getIsExport() ? mlir::SymbolTable::Visibility::Public
                           : mlir::SymbolTable::Visibility::Private;
    }
    void setVisibility(mlir::SymbolTable::Visibility visibility) {
      assert(visibility != mlir::SymbolTable::Visibility::Nested &&
             "CSL symbols are either exported (public) or private");
      setIsExport(visibility == mlir::SymbolTable::Visibility::Public);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

def TypedCSL_FuncOp : TypedCSL_Op<"func", [
    IsolatedFromAbove,
    Symbol,
    SymbolName,
    AutomaticAllocationScope,
    HasBreakingControlFlowOpInterface,
    HasNestedTerminator<["ReturnOp", "UnreachableOp"]>,
    CallableOpInterface
]> {
  let summary = "Function definition with concrete types";
  let description = [{
    Defines a function with typed parameters and body.

    All parameter and return types are fully concrete TypedCSL types.

    Example with return type:
    ```mlir
    typed_csl.func @add(%x: si32, %y: si32) -> si32 {
      %result = typed_csl.binary add, %x, %y : (si32, si32) -> si32
      typed_csl.return %result : si32
    }
    ```

    Example without return type (void):
    ```mlir
    typed_csl.func @do_something(%x: si32) {
      typed_csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    UnitAttr:$is_task,
    UnitAttr:$is_inline,
    UnitAttr:$is_noinline,
    UnitAttr:$is_export,
    UnitAttr:$is_extern,
    UnitAttr:$is_async,
    OptionalAttr<UI32Attr>:$stack_size,
    OptionalAttr<StrAttr>:$linkname,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<TypeAttr>:$async_result_type,
    OptionalAttr<UI32Attr>:$max_recursion_depth
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    mlir::SymbolTable::Visibility getVisibility() {
      // Functions are private by default, unless marked 'export'
      return getIsExport() ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private;
    }
    void setVisibility(mlir::SymbolTable::Visibility visibility) {
      assert(visibility != mlir::SymbolTable::Visibility::Nested &&
             "CSL symbols are either exported (public) or private");
      setIsExport(visibility == mlir::SymbolTable::Visibility::Public);
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the body region of this function.
    Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns true if this function is external (has no body).
    bool isExternal() { return getBody().empty(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TypedCSL_ReturnOp : TypedCSL_Op<"return", [
    Terminator,
    RegionExitTerminatorOpInterface,
    TypedCSL_BreakingTerminatorOpInterface,
    RegionBranchTerminatorOpInterface
]> {
  let summary = "Return from function";
  let description = [{
    Returns zero or one value from a function.

    The return value type must match the function's return type.

    Example:
    ```mlir
    typed_csl.return 1 %val : si32
    ```

    Example (void return):
    ```mlir
    typed_csl.return 1
    ```
  }];

  let arguments = (ins I32Prop:$num_breaking_regions,
                       Optional<AnyType>:$operand);

  let assemblyFormat = "$num_breaking_regions ($operand^ `:` type($operand))? attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    // Void return with explicit N
    OpBuilder<(ins "int":$numBreakingRegions), [{
      $_state.getOrAddProperties<Properties>().num_breaking_regions =
          numBreakingRegions;
    }]>,
    // Return with value and explicit N
    OpBuilder<(ins "int":$numBreakingRegions, "Value":$operand), [{
      if (operand) $_state.addOperands(operand);
      $_state.getOrAddProperties<Properties>().num_breaking_regions =
          numBreakingRegions;
    }]>
  ];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_UnreachableOp : TypedCSL_Op<"unreachable", [
    Terminator,
    RegionExitTerminatorOpInterface,
    TypedCSL_BreakingTerminatorOpInterface,
    RegionBranchTerminatorOpInterface
]> {
  let summary = "Unreachable terminator";
  let description = [{
    Marks a point that is unreachable because all control flow paths have
    already returned. This is a proper MLIR terminator that satisfies the
    requirement that all blocks must end with a terminator.

    The CSL emitter outputs this as a `// unreachable` comment since the
    function body already has all necessary returns in nested control flow.

    Example:
    ```mlir
    typed_csl.unreachable 1
    ```
  }];

  let arguments = (ins I32Prop:$num_breaking_regions);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "int":$numBreakingRegions), [{
      $_state.getOrAddProperties<Properties>().num_breaking_regions =
          numBreakingRegions;
    }]>
  ];

  let assemblyFormat = "$num_breaking_regions attr-dict";
  let hasCanonicalizer = 1;
}

// Non-terminator suspend/await: used inside structured CF regions (IfOp etc.)
// LowerSuspendingSCFToCFPass replaces these with suspend_br/await_br.

def TypedCSL_SuspendOp : TypedCSL_Op<"suspend"> {
  let summary = "Coroutine suspension point (structured)";
  let description = [{
    Marks a coroutine suspension point inside structured control flow.
    LowerSuspendingSCFToCFPass replaces this with typed_csl.suspend_br
    when flattening structured CF ops to the CFG.
  }];

  let assemblyFormat = "attr-dict";
}

def TypedCSL_AwaitOp : TypedCSL_Op<"await"> {
  let summary = "Await a future (structured)";
  let description = [{
    Awaits completion of a coroutine represented by a future handle.

    When `offload` is present, the callee gets its own worker task for
    direct hardware resume instead of running inline on the caller's task.

    LowerSuspendingSCFToCFPass replaces this with typed_csl.await_br
    when flattening structured CF ops to the CFG.
  }];

  let arguments = (ins CSL_FutureType:$future, OptionalAttr<UnitAttr>:$offload);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    $future attr-dict `:` type($future) (`->` type($result)^)?
  }];
}

def TypedCSL_AwaitParallelOp : TypedCSL_Op<"await_parallel"> {
  let summary = "Await multiple coroutines concurrently (structured)";
  let description = [{
    Runs multiple async function calls concurrently and suspends the caller
    until all of them complete. Each callee gets a dedicated worker task.

    The optional result is a tuple of the callee results in callee order.
    Callees without a result have no field; if no callee has a result, the
    op has none either.

    LowerSuspendingSCFToCFPass handles the join loop's suspend point.
  }];

  let arguments = (ins Variadic<CSL_FutureType>:$futures);
  let results = (outs Optional<CSL_StructType>:$result);
  let assemblyFormat = [{
    `(` $futures `)` attr-dict `:` type($futures) (`->` type($result)^)?
  }];
  let hasVerifier = 1;
}


def TypedCSL_FrameAddressOp : TypedCSL_Op<"frame_address", [Pure]> {
  let summary = "Get the [*]u8 pointer to the current coroutine frame";
  let description = [{
    Returns a `[*]u8` pointer to the coroutine frame of the enclosing async
    function or task. Useful for debugging frame contents at runtime.

    Lowered by CoroPrepareCFPass to the frame byte pointer: for async fns
    this is the first block argument, for tasks it is the address of the
    global frame variable.
  }];

  let results = (outs CSL_PointerType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// Not Pure: the caller stores params into the frame space this op reserves.
// DCE must not remove it even when results are unused — otherwise the callee
// frame shrinks and the caller's param stores write out of bounds.
def TypedCSL_CoroParamsOp : TypedCSL_Op<"coro_params", []> {
  let summary = "Extract coroutine parameter lvalues from frame pointer";
  let description = [{
    Given a pointer to a coroutine frame, returns const lvalues for each
    parameter at their calling convention offsets.

    The convention places parameters consecutively starting at offset 4
    (after the u16 state field and u16 workerid field) with natural
    alignment. The result value shares the same starting offset (offset 4,
    aligned) because it is written only after the callee finishes, when
    parameters are dead.

    This op is a marker for the frame analysis: each result is preplaced
    at its convention offset. The op is lowered to byte-level frame access
    IR during trampoline emission.

    Example:
    ```mlir
    %p0, %p1 = typed_csl.coro_params %ptr
        : !csl.ptr<mutable !csl.coroframe<@fn>>
       -> (!csl.lvalue<const si32>, !csl.lvalue<const f32>)
    ```
  }];

  let arguments = (ins CSL_PointerType:$frame_ptr);
  let results = (outs Variadic<TypedCSL_LValueType>:$params);
  let assemblyFormat = [{
    $frame_ptr attr-dict `:` type($frame_ptr) `->` type($params)
  }];
}

// Terminator suspend_br/await_br: used in flattened CFG after lowering.

def TypedCSL_SuspendBrOp : TypedCSL_Op<"suspend_br", [
    Terminator,
    DeclareOpInterfaceMethods<BranchOpInterface>
]> {
  let summary = "Coroutine suspension point (CFG terminator)";
  let description = [{
    Suspends the current coroutine, yielding control back to the caller.
    Has a single successor which is the resume point.

    Example:
    ```mlir
    typed_csl.suspend_br [^resume]
    ```
  }];

  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
  let hasCanonicalizer = 1;
}

def TypedCSL_AwaitBrOp : TypedCSL_Op<"await_br", [
    Terminator,
    DeclareOpInterfaceMethods<BranchOpInterface>
]> {
  let summary = "Non-mergeable block boundary at async call sites (CFG terminator)";
  let description = [{
    Unconditional branch that the canonicalizer will not fold into its
    successor.  Used to separate frame-init ops from the async call so
    they end up in different trampoline states — preventing re-execution
    of init ops on task reactivation after nop.term.

    Semantically identical to cf.br, but the canonicalizer only collapses
    pass-through successors (empty blocks with a single cf.br), never
    merges the predecessor into the successor.

    Example:
    ```mlir
    typed_csl.await_br ^call_block
    ```
  }];

  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
  let hasCanonicalizer = 1;
}

def TypedCSL_CallOp : TypedCSL_Op<"call", [
  CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Function call with concrete types";
  let description = [{
    Calls a function with typed arguments.

    All argument types must match the callee's parameter types.

    Example:
    ```mlir
    %result = typed_csl.call @add(%a, %b) : (si32, si32) -> si32
    ```

    Example (void return):
    ```mlir
    typed_csl.call @do_something(%x) : (si32) -> ()
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let results = (outs Optional<AnyType>:$result);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }

    bool hasSideEffects() {
      return false;
    }
  }];

  let hasVerifier = 1;
}

def TypedCSL_IndirectCallOp : TypedCSL_Op<"indirect_call", []> {
  let summary = "Indirect function call through function pointer";
  let description = [{
    Calls a function through a function pointer value.

    The callee must be a function pointer type (*const fn(...) -> ...).
    Function references (fn(...) -> ...) are implicitly cast to function
    pointers in NormalizeCastsPass before reaching TypedCSL.

    Example:
    ```mlir
    // Indirect call through function pointer
    %result = typed_csl.indirect_call %func_ptr(%a, %b) : !csl.ptr<const (si32, si32) -> si32>, (si32, si32) -> si32
    ```
  }];

  let arguments = (ins
    AnyType:$callee,
    Variadic<AnyType>:$args
  );

  let results = (outs Optional<AnyType>:$result);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` type($callee) `,` functional-type($args, results)
  }];

  let extraClassDeclaration = [{
    /// Get the function type from the callee (unwrapping pointer).
    FunctionType getCalleeType();
  }];

  let hasVerifier = 1;
}

def TypedCSL_BuiltinCallOp : TypedCSL_Op<"builtin_call", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Builtin function call with concrete types";
  let description = [{
    Calls a builtin function with typed arguments.

    Builtins are language-defined functions like @get_color, @add16, etc.
    Memory effects are determined by whether the builtin is pure (no side effects).

    Example:
    ```mlir
    %color = typed_csl.builtin_call get_color(%id) : (ui16) -> !csl.color
    ```
  }];

  let arguments = (ins
    CSL_BuiltinKind:$builtin,
    Variadic<AnyType>:$args
  );

  let results = (outs Optional<AnyType>:$result);

  let assemblyFormat = [{
    $builtin `(` $args `)` attr-dict `:` functional-type($args, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def TypedCSL_IfOp : TypedCSL_Op<"if", [
  AutomaticAllocationScope,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  PropagateControlFlowBreak
]> {
  let summary = "Conditional control flow / if-expression";
  let description = [{
    Executes the then region if condition is true, otherwise executes the
    optional else region.

    Can be used as a statement (no result) or as an expression (has result).
    When used as an expression, both regions must be present.

    The condition must be of type i1.

    Statement form:
    ```mlir
    typed_csl.if %cond : i1 {
      typed_csl.yield
    } else {
      typed_csl.yield
    }
    ```

    Expression form:
    ```mlir
    %result = typed_csl.if %cond : i1 {
      typed_csl.yield %true_val : si32
    } else {
      typed_csl.yield %false_val : si32
    } : si32
    ```
  }];

  let arguments = (ins CSL_BoolType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let results = (outs Optional<AnyType>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Returns true if this is an if-expression (has result)
    bool isExpression() { return getResult() != nullptr; }
  }];
}

def TypedCSL_SwitchOp : TypedCSL_Op<"switch", [
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  PropagateControlFlowBreak
]> {
  let summary = "Switch statement/expression with typed input";
  let description = [{
    Switch that matches input against case values. Can be used as a statement
    (no result) or expression (has result).

    Each case region yields a value via YieldOp. The case_values attribute
    contains arrays of resolved constant values for each case.

    When used as a statement, all regions yield void.
    When used as an expression, all regions yield values of compatible types.

    Requirements:
    - Input must be a fixed-width integer or enum type
    - All case values must be unique
    - All values must be covered (explicitly or via else)

    Assembly format:
      typed_csl.switch <input> : <type> cases <values> <regions> [default <region>] [: <result_type>]

    The "default" keyword introduces the fallback region.
  }];

  let arguments = (ins
    AnyType:$input,  // Fixed-width integer or enum type
    ArrayAttr:$case_values  // Array of arrays of TypedAttr
  );

  // Note: else_region comes first in TableGen because MLIR requires variadic regions last.
  // The custom assembly format prints/parses in the natural order (cases, then else).
  let regions = (region
    AnyRegion:$else_region,
    VariadicRegion<AnyRegion>:$case_regions
  );

  let results = (outs Optional<AnyType>:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns true if this switch has an else branch
    bool hasElseBranch() { return !getElseRegion().empty(); }

    /// Get the number of case branches (not including else)
    unsigned getNumCases() { return getCaseRegions().size(); }

    /// Returns true if this is a switch expression (has non-void result)
    bool isExpression() { return getResult() != nullptr; }
  }];
}

def TypedCSL_ComptimeOp : TypedCSL_Op<"comptime", [
  RecursiveMemoryEffects, SingleBlock, NoRegionArguments,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  PropagateControlFlowBreak
]> {
  let summary = "Comptime block for final compiler evaluation";
  let description = [{
    A block of code that will be evaluated at compile-time by the final CSL compiler.
    This is used for operations that cannot be evaluated in our compiler but must
    be deferred to the CSL compiler (e.g., @bind_local_task, @set_rectangle).

    Example:
    ```mlir
    typed_csl.comptime {
      typed_csl.builtin_call bind_local_task(%task, %id) : (!csl.task, !csl.local_task_id) -> ()
    }
    ```
  }];

  let arguments = (ins);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";
}

def TypedCSL_LayoutOp : TypedCSL_Op<"layout", [
  RecursiveMemoryEffects, SingleBlock, NoRegionArguments,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  PropagateControlFlowBreak
]> {
  let summary = "Layout block for tile configuration";
  let description = [{
    A block of code that configures the tile layout at compile-time.
    Semantically identical to comptime, but emitted as `layout { }` syntax.

    Example:
    ```mlir
    typed_csl.layout {
      typed_csl.builtin_call set_rectangle(%width, %height) : (ui16, ui16) -> ()
    }
    ```
  }];

  let arguments = (ins);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";
}

def TypedCSL_ForOp : TypedCSL_Op<"for", [
  AutomaticAllocationScope,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  PropagateControlFlowBreak,
  HasBreakingControlFlowOpInterface,
  HasNestedTerminator<["BreakOp", "ContinueOp"]>,
  CSL_LabeledOpInterface
]> {
  let summary = "For loop iterating over a range or array";
  let description = [{
    Iterates over elements of a range or array. The body region receives:
    - First block argument: the current element value
    - Second block argument (optional): the index (always u16)

    The iterable must be either:
    - A range type (!csl.range<T>) created by @range builtin
    - An array type

    Example with element only:
    ```mlir
    typed_csl.for(%range : !csl.range<si32>) {elem_name = "x"} {
    ^bb0(%elem: si32):
      // loop body using %elem
    }
    ```

    Example with element and index:
    ```mlir
    typed_csl.for(%array : !csl.array<10, si32>) {elem_name = "x", index_name = "i"} {
    ^bb0(%elem: si32, %idx: ui16):
      // loop body using %elem and %idx
    }
    ```

    Example with label:
    ```mlir
    typed_csl.for(%range : !csl.range<si32>) {elem_name = "x", label = "outer"} {
    ^bb0(%elem: si32):
      // can use break :outer; to exit this loop
    }
    ```
  }];

  let arguments = (ins
    AnyType:$iterable,
    StrAttr:$elem_name,
    OptionalAttr<StrAttr>:$index_name,
    OptionalAttr<StrAttr>:$label,
    DefaultValuedAttr<BoolAttr, "false">:$is_inline
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Get the element block argument
    BlockArgument getElemArg() { return getBody().front().getArgument(0); }

    /// Get the index block argument (if present)
    BlockArgument getIndexArg() {
      if (getBody().front().getNumArguments() > 1)
        return getBody().front().getArgument(1);
      return BlockArgument();
    }

    /// Check if this for loop has an index variable
    bool hasIndex() { return getIndexName().has_value(); }
  }];

  let assemblyFormat = [{
    `(` $iterable `:` type($iterable) `)` attr-dict-with-keyword $body
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_WhileOp : TypedCSL_Op<"while", [
  AutomaticAllocationScope,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  PropagateControlFlowBreak,
  HasBreakingControlFlowOpInterface,
  HasNestedTerminator<["BreakOp", "ContinueOp"]>,
  CSL_LabeledOpInterface
]> {
  let summary = "While loop with condition region";
  let description = [{
    Standard while loop. The condition region is evaluated before each iteration
    and must yield a boolean value (i1) via `typed_csl.yield`.
    Optional continuation region is executed at the end of each iteration
    (including iterations ended by continue).

    Example without continuation:
    ```mlir
    typed_csl.while cond {
      %cond = ...
      typed_csl.yield %cond : i1
    } do {
      // loop body
    }
    ```

    Example with continuation:
    ```mlir
    typed_csl.while cond {
      %cond = ...
      typed_csl.yield %cond : i1
    } do {
      // loop body
    } continuation {
      // executed at end of each iteration
    }
    ```

    Example with label:
    ```mlir
    typed_csl.while {label = "outer"} cond {
      %cond = ...
      typed_csl.yield %cond : i1
    } do {
      // can use break :outer; to exit this loop
    }
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$label
  );

  let regions = (region
    SizedRegion<1>:$cond_region,
    SizedRegion<1>:$body,
    AnyRegion:$continuation
  );

  let assemblyFormat = [{
    attr-dict-with-keyword `cond` $cond_region `do` $body (`continuation` $continuation^)?
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_BreakOp : TypedCSL_Op<"break", [
    Terminator,
    RegionExitTerminatorOpInterface,
    TypedCSL_BreakingTerminatorOpInterface,
    RegionBranchTerminatorOpInterface
]> {
  let summary = "Break out of a loop or labeled block, optionally with a value";
  let description = [{
    Exits the enclosing loop or labeled block. If a target label is specified,
    breaks out of the construct with that label. An optional value can be
    provided when breaking out of a labeled block (not a loop).

    The `num_breaking_regions` integer N indicates how many region levels
    to exit. N=1 means exit the immediately enclosing region.

    Example:
    ```mlir
    typed_csl.break 1                           // break innermost loop
    typed_csl.break 2 "outer"                   // break labeled construct through if
    typed_csl.break 2 "blk" %value : si32       // break labeled block with value
    ```
  }];

  let arguments = (ins
    I32Prop:$num_breaking_regions,
    OptionalAttr<StrAttr>:$target_label,
    Optional<AnyType>:$value
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "int":$numBreakingRegions,
               "StringAttr":$targetLabel, "Value":$value), [{
      if (targetLabel)
        $_state.getOrAddProperties<Properties>().target_label = targetLabel;
      if (value) $_state.addOperands(value);
      $_state.getOrAddProperties<Properties>().num_breaking_regions =
          numBreakingRegions;
    }]>
  ];

  let assemblyFormat = [{ $num_breaking_regions ($target_label^)? ($value^ `:` type($value))? attr-dict }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_ContinueOp : TypedCSL_Op<"continue", [
    Terminator,
    RegionExitTerminatorOpInterface,
    TypedCSL_BreakingTerminatorOpInterface,
    RegionBranchTerminatorOpInterface
]> {
  let summary = "Continue to next iteration of a loop";
  let description = [{
    Continues to the next iteration of the enclosing loop. If a target label
    is specified, continues the loop with that label (for nested loops).

    The `num_breaking_regions` integer N indicates how many region levels
    to exit. N=1 means continue the immediately enclosing loop.

    Example:
    ```mlir
    typed_csl.continue 1            // continue innermost loop
    typed_csl.continue 2 "outer"    // continue loop labeled "outer" through if
    ```
  }];

  let arguments = (ins
    I32Prop:$num_breaking_regions,
    OptionalAttr<StrAttr>:$target_label
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "int":$numBreakingRegions,
               "StringAttr":$targetLabel), [{
      if (targetLabel)
        $_state.getOrAddProperties<Properties>().target_label = targetLabel;
      $_state.getOrAddProperties<Properties>().num_breaking_regions =
          numBreakingRegions;
    }]>
  ];

  let assemblyFormat = [{ $num_breaking_regions ($target_label^)? attr-dict }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Labeled Block Operations
//===----------------------------------------------------------------------===//

def TypedCSL_LabeledBlockOp : TypedCSL_Op<"labeled_block", [
  AutomaticAllocationScope,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
  PropagateControlFlowBreak,
  HasBreakingControlFlowOpInterface,
  HasNestedTerminator<["BreakOp"]>,
  CSL_LabeledOpInterface
]> {
  let summary = "Labeled block that can be exited with break, optionally yielding a value";
  let description = [{
    A labeled block establishes a scope that can be exited early using
    `break :label [value]`. Unlike loops, labeled blocks do not support `continue`.

    Example (void):
    ```mlir
    typed_csl.labeled_block "outer" {
      typed_csl.if %cond : i1 {
        typed_csl.break "outer"
      }
    }
    ```

    Example (valued):
    ```mlir
    %result = typed_csl.labeled_block "blk" -> si32 {
      typed_csl.if %cond : i1 {
        typed_csl.break "blk" %val1 : si32
      }
      typed_csl.break "blk" %val2 : si32
    }
    ```
  }];

  let arguments = (ins StrAttr:$label);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Optional<AnyType>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp - Return value from expression regions
//===----------------------------------------------------------------------===//

def TypedCSL_YieldOp : TypedCSL_Op<"yield", [
    Pure, Terminator, ReturnLike,
    DeclareOpInterfaceMethods<RegionExitTerminatorOpInterface>
    // ParentOneOf<["SelectOp", "ShortCircuitOp", "WhileOp", "SwitchOp"]>
]> {
  let summary = "Yield value from expression region";
  let description = [{
    Indicates that control flows back from this region to the parent region.
    If a value is provided, it becomes the result of the parent operation.

    Example:
    ```mlir
    typed_csl.yield %value : si32
    ```
  }];

  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = "($value^ `:` type($value))? attr-dict";
}

//===----------------------------------------------------------------------===//
// CondYieldOp - While loop condition terminator
//===----------------------------------------------------------------------===//

def TypedCSL_CondYieldOp : TypedCSL_Op<"cond_yield", [
    HasParent<"WhileOp">,
    Pure, Terminator,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
      ["getMutableSuccessorOperands", "getSuccessorRegions"]>
]> {
  let summary = "while loop continuation condition";
  let description = [{
    Terminates the condition region of a `typed_csl.while` loop. If the
    condition is true, the loop body is entered. Otherwise, the loop exits.

    The condition value is consumed by the while loop's branching logic and
    is not forwarded to any successor.

    Example:
    ```mlir
    typed_csl.while cond {
      %cond = ...
      typed_csl.cond_yield %cond : i1
    } do {
      // loop body
    }
    ```
  }];

  let arguments = (ins I1:$condition);
  let assemblyFormat = "$condition `:` type($condition) attr-dict";
}

//===----------------------------------------------------------------------===//
// SelectOp - Conditional select with short-circuit evaluation
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ShortCircuitOp - Short-circuit logical operations
//===----------------------------------------------------------------------===//

def TypedCSL_ShortCircuitOp : TypedCSL_Op<"short_circuit", [
    RecursivelySpeculatable, RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
      ["getSuccessorInputs", "getEntrySuccessorOperands"]>
]> {
  let summary = "Short-circuit logical operation";
  let description = [{
    Evaluates a short-circuit logical operation (AND or OR).

    For AND (&&): Returns false immediately if LHS is false, otherwise evaluates
    and returns RHS.

    For OR (||): Returns true immediately if LHS is true, otherwise evaluates
    and returns RHS.

    The RHS is in a region that is only entered when needed. The region must be
    terminated by a `typed_csl.yield` operation.

    Examples:
    ```mlir
    // Logical AND: lhs && rhs
    %result = typed_csl.short_circuit %lhs and {
      typed_csl.yield %rhs : i1
    }

    // Logical OR: lhs || rhs
    %result = typed_csl.short_circuit %lhs or {
      typed_csl.yield %rhs : i1
    }
    ```
  }];

  let arguments = (ins
    CSL_ShortCircuitKind:$kind,
    CSL_BoolType:$lhs
  );

  let regions = (region SizedRegion<1>:$rhs_region);
  let results = (outs CSL_BoolType:$result);

  let assemblyFormat = "$lhs $kind $rhs_region attr-dict";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Struct Construction Operations
//===----------------------------------------------------------------------===//

def TypedCSL_ConstructTupleOp : TypedCSL_Op<"construct_tuple", [Pure]> {
  let summary = "Construct tuple value";
  let description = [{
    Creates a tuple value with unnamed, positional fields.

    Tuples are structs with positional fields (no field names).
    Field values are provided as SSA operands, and the result type
    is a concrete struct type.

    Example:
    ```mlir
    %x = typed_csl.constant 10 : si32
    %y = typed_csl.constant 3.14 : f32
    %tuple = typed_csl.construct_tuple %x, %y : si32, f32 -> !csl.struct<si32, f32>
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$field_values
  );

  let results = (outs CSL_StructType:$result);

  let assemblyFormat = "$field_values attr-dict `:` type($field_values) `->` type($result)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_ConstructStructOp : TypedCSL_Op<"construct_struct", [Pure]> {
  let summary = "Construct struct value with named fields";
  let description = [{
    Creates a struct value with named fields. Supports both anonymous structs
    (structural typing) and named structs (nominal typing) based on the
    result type.

    Anonymous structs use structural typing - two structs with the same
    field names and types are considered equal regardless of where they
    were defined.

    Named structs use nominal typing - they are identified by their
    definition location. The struct type carries both the field information
    and the definition location.

    Field names are specified as a string array attribute, and field values
    are provided as SSA operands. The result is a concrete struct type.

    Example (anonymous struct):
    ```mlir
    %real = typed_csl.constant 1.0 : f16
    %imag = typed_csl.constant 2.0 : f16
    %complex = typed_csl.construct_struct ["real", "imag"] %real, %imag
                 : f16, f16 -> !csl.struct<real:f16, imag:f16>
    ```

    Example (named struct):
    ```mlir
    %real = typed_csl.constant 1.0 : f16
    %imag = typed_csl.constant 2.0 : f16
    %complex = typed_csl.construct_struct ["real", "imag"] %real, %imag
                 : f16, f16 -> !csl.struct<{loc("./pe.csl":92:3)}, real:f16, imag:f16>
    ```
  }];

  let arguments = (ins
    StrArrayAttr:$field_names,
    Variadic<AnyType>:$field_values
  );

  let results = (outs CSL_StructType:$result);

  let assemblyFormat = "$field_names $field_values attr-dict `:` type($field_values) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Array Construction Operations
//===----------------------------------------------------------------------===//

def TypedCSL_ConstructArrayOp : TypedCSL_Op<"construct_array", [Pure]> {
  let summary = "Construct array value from element values";
  let description = [{
    Creates an array value from individual element values.
    Elements are provided in row-major (flattened) order.

    For multi-dimensional arrays, elements are listed in row-major order.
    For example, a 2x2 array created with elements {1, 2, 3, 4} produces:
      [[1, 2], [3, 4]]

    Example:
    ```mlir
    %e1 = typed_csl.constant 1 : si32
    %e2 = typed_csl.constant 2 : si32
    %e3 = typed_csl.constant 3 : si32
    %arr = typed_csl.construct_array %e1, %e2, %e3 : si32, si32, si32 -> !csl.array<3 x si32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);

  let results = (outs CSL_ArrayType:$result);

  let assemblyFormat = "$elements attr-dict `:` type($elements) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Struct Access Operations
//===----------------------------------------------------------------------===//

def TypedCSL_FieldAccessOp : TypedCSL_Op<"field_access", [Pure]> {
  let summary = "Access struct field by name";
  let description = [{
    Accesses a named field of a struct, returning an lvalue to that field.

    This operation takes an lvalue to a struct and produces an lvalue to
    the specified field. This allows for nested field access and supports
    both reading (via load) and writing (via assign).

    Works with both anonymous and named structs (not tuples).
    The result lvalue has the same constness as the input lvalue.

    Example:
    ```mlir
    %real = typed_csl.constant 1.0 : f16
    %imag = typed_csl.constant 2.0 : f16
    %struct_literal = typed_csl.construct_struct ["real", "imag"] %real, %imag
                       : !csl.struct<real:f16, imag:f16>
    %struct_lval = typed_csl.var "my_struct" init(%struct_literal)
                    : !csl.lvalue<mutable !csl.struct<real:f16, imag:f16>>
    %field_lval = typed_csl.field_access %struct_lval["real"]
                   : !csl.lvalue<mutable !csl.struct<real:f16, imag:f16>>
                   -> !csl.lvalue<mutable f16>
    %real_val = typed_csl.load from %field_lval : !csl.lvalue<mutable f16>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[TypedCSL_LValueType, CSL_StructType]>:$struct_lvalue,
    StrAttr:$field_name
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$struct_lvalue `[` $field_name `]` attr-dict `:` type($struct_lvalue) `->` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_TupleIndexAccessOp : TypedCSL_Op<"tuple_index_access", [Pure]> {
  let summary = "Access tuple/struct element by comptime index";
  let description = [{
    Accesses a tuple or struct element by numeric index.

    This operation takes either an lvalue or rvalue of a tuple/struct and
    produces an lvalue or rvalue (respectively) to the element at the
    specified index. The index is a compile-time constant attribute
    (required since tuple/struct fields are heterogeneous).

    Works with tuples and structs. For named structs, accesses fields in
    declaration order.

    Example (lvalue access):
    ```mlir
    %tuple_lval = typed_csl.var "my_tuple" init(...)
                   : !csl.lvalue<mutable !csl.struct<f16, f16>>
    %elem_lval = typed_csl.tuple_index_access %tuple_lval[0]
                  : !csl.lvalue<mutable !csl.struct<f16, f16>>
                  -> !csl.lvalue<mutable f16>
    %val = typed_csl.load from %elem_lval : !csl.lvalue<mutable f16>
    ```

    Example (rvalue access):
    ```mlir
    %tuple = typed_csl.construct_tuple %a, %b : !csl.struct<f16, f16>
    %elem = typed_csl.tuple_index_access %tuple[0]
             : !csl.struct<f16, f16> -> f16
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[TypedCSL_LValueType, CSL_StructType]>:$base,
    SI64Attr:$index
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$base `[` $index `]` attr-dict `:` type($base) `->` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TypedCSL_ArrayIndexAccessOp : TypedCSL_Op<"array_index_access", [Pure]> {
  let summary = "Access array element by runtime indices";
  let description = [{
    Accesses an array element by numeric indices.

    This operation takes either an lvalue or rvalue of an array and produces
    an lvalue or rvalue (respectively) to the element at the specified indices.
    The indices are runtime values (arrays are homogeneous, so the result type
    is always the element type).

    Supports multi-dimensional indexing with multiple indices.

    Example (lvalue access):
    ```mlir
    %arr_lval = typed_csl.var "matrix" : !csl.lvalue<mutable !csl.array<[3, 3] x f16>>
    %i = typed_csl.constant 1 : ui16
    %j = typed_csl.constant 2 : ui16
    %elem_lval = typed_csl.array_index_access %arr_lval[%i, %j]
                  : !csl.lvalue<mutable !csl.array<[3, 3] x f16>>
                  -> !csl.lvalue<mutable f16>
    ```

    Example (rvalue access):
    ```mlir
    %arr = ... : !csl.array<[3, 3] x f16>
    %elem = typed_csl.array_index_access %arr[%i, %j]
             : !csl.array<[3, 3] x f16> -> f16
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[TypedCSL_LValueType, CSL_ArrayType]>:$base,
    Variadic<AnyType>:$indices
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$base `[` $indices `]` attr-dict `:` type($base) `,` `(` type($indices) `)` `->` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

def TypedCSL_PtrIndexAccessOp : TypedCSL_Op<"ptr_index_access", [Pure]> {
  let summary = "Access element of many-item pointer by runtime index";
  let description = [{
    Accesses an element of a many-item pointer by a runtime index.

    This operation takes a many-item pointer (rvalue) and produces an lvalue
    to the element at the specified index. The result is always an lvalue
    because pointer indexing provides a reference to the memory location.

    The index is a runtime value. Only many-item pointers (`[*]T`) can be
    indexed; single-item pointers must be dereferenced with `.*` instead.

    Example:
    ```mlir
    %ptr = ... : !csl.ptr<? x f16>
    %i = typed_csl.constant 5 : u16
    %elem_lval = typed_csl.ptr_index_access %ptr[%i]
                  : !csl.ptr<? x f16> -> !csl.lvalue<mutable f16>
    %val = typed_csl.load from %elem_lval : !csl.lvalue<mutable f16>
    ```
  }];

  let arguments = (ins
    CSL_PointerType:$base,
    AnyType:$index
  );

  let results = (outs TypedCSL_LValueType:$result);

  let assemblyFormat = "$base `[` $index `]` attr-dict `:` type($base) `,` type($index) `->` type($result)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

#endif // TYPED_CSL_OPS
