//===- ParsedCSLOps.td - ParsedCSL Operation Definitions --*- tablegen -*-===//
//
// Operation definitions for the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_OPS
#define PARSED_CSL_OPS

include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLBase.td"
include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLTypes.td"
include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLAttrs.td"
include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLInterfaces.td"
include "csl-mlir/Dialect/CSL/IR/CSLOpInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


//===----------------------------------------------------------------------===//
// ParsedCSL Operation Base Class
//===----------------------------------------------------------------------===//

class ParsedCSL_Op<string mnemonic, list<Trait> traits = []>
    : Op<ParsedCSL_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Trait Definitions
//===----------------------------------------------------------------------===//

// Ops that represent expressions (BinaryOp, ConstantOp, etc.). 
// Expression ops are always allowed at top-level to calculate initializer values.
def ParsedCSL_Expression : NativeOpTrait<"Expression"> {
  let cppNamespace = "::mlir::parsed_csl::OpTrait";
}

// Ops that represent statements (AssignOp, IfOp, etc.)
def ParsedCSL_Statement : NativeOpTrait<"Statement"> {
  let cppNamespace = "::mlir::parsed_csl::OpTrait";
}

// Ops that define a comptime scope (ComptimeExprOp, ComptimeBlockOp)
def ParsedCSL_ComptimeScope : NativeOpTrait<"ComptimeScope"> {
  let cppNamespace = "::mlir::parsed_csl::OpTrait";
}

// Ops that terminate a region (return, break, continue, yield, unreachable).
// These are not necessarily MLIR Terminators (since most ParsedCSL regions use
// NoTerminator), but they are the last op the comptime evaluator should see.
def ParsedCSL_PseudoTerminator : NativeOpTrait<"PseudoTerminator"> {
  let cppNamespace = "::mlir::parsed_csl::OpTrait";
}


//===----------------------------------------------------------------------===//
// Value Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_ConstantOp : ParsedCSL_Op<"constant", [
    InferredTypeInterface, ConstantLike, Pure, ParsedCSL_Expression]> {
  let summary = "Constant value";
  let description = [{
    Represents a constant value with a known compile-time value.
    The attribute contains the value and its type information.

    All constants have result type !parsed_csl.unresolved. The actual type
    information is preserved in the attribute.

    Examples:
    ```mlir
    %c = parsed_csl.constant #csl.comptime_int<12345678901234567890>
    %f = parsed_csl.constant #csl.comptime_float<3.14>
    %b = parsed_csl.constant true : s1
    %i = parsed_csl.constant -100 : si32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Implement InferredTypeInterface by deriving type from value attribute
    ::std::optional<::mlir::Type> getInferredType() {
      return getValue().getType();
    }

    void setInferredType(::std::optional<::mlir::Type> attrValue) {
      if (attrValue && *attrValue != getValue().getType())
        llvm_unreachable("Cannot set inferred type on ConstantOp: type is always derived from value attribute");
      // Otherwise no-op: type is always derived from value attribute
    }
  }];
}

def ParsedCSL_NoValueOp : ParsedCSL_Op<"no_value", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Create a type-only marker with no value";
  let description = [{
    Creates a typed marker that has no associated value. Used in contexts
    like generic function type bodies where we need to reason about types
    (e.g., @type_of(someArg)) without having actual values.

    During comptime evaluation, this resolves to a NoValueAttr with the
    inferred type. The result can be used with type introspection operations
    like @type_of, but has no comptime-known value.

    Example:
    ```mlir
    %i32_type = parsed_csl.builtin_type si32
    %no_val = parsed_csl.no_value %i32_type
    %t = parsed_csl.builtin_call type_of(%no_val)  // returns i32 type
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$typeValue,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$typeValue attr-dict";
}

//===----------------------------------------------------------------------===//
// Global Variable Operations (Symbol-based)
//===----------------------------------------------------------------------===//

def ParsedCSL_GlobalOp : ParsedCSL_Op<"global", [
  Symbol,
  SymbolName,
  InferredTypeInterface,
  NoRegionArguments,
  ParsedCSL_Statement
]> {
  let summary = "Global variable declaration";
  let description = [{
    Declares a global variable at module scope. Unlike VariableOp, this operation
    has no SSA result - globals are accessed via GetSymbolOp.

    The dynamicType operand specifies the variable's type as an SSA value.
    The optional initRegion contains the initializer expression, terminated by
    a YieldOp that yields the initial value.

    During comptime evaluation, inferred_type is populated with the resolved LValueType.

    Note: Globals cannot be marked `comptime` - only `const` or mutable.

    Example:
    ```mlir
    %i32 = parsed_csl.builtin_type si32
    parsed_csl.global @x dynamicType(%i32) {
      %init = parsed_csl.constant #csl.comptime_int<42>
      parsed_csl.yield %init
    }
    parsed_csl.global @y const {
      %init = parsed_csl.constant #csl.comptime_int<42>
      parsed_csl.yield %init
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Optional<ParsedCSL_UnresolvedType>:$dynamicType,
    UnitAttr:$is_const,
    UnitAttr:$is_export,
    UnitAttr:$is_extern,
    UnitAttr:$is_unbound_param,
    OptionalAttr<StrAttr>:$linkname,
    OptionalAttr<UI32Attr>:$align,
    OptionalAttr<TypeAttr>:$inferred_type,
    OptionalAttr<TypedAttrInterface>:$inferred_initializer
  );

  let regions = (region
    AnyRegion:$initRegion,
    AnyRegion:$linkname_region,  // Optional: comptime expression yielding comptime_string
    AnyRegion:$align_region    // Optional: comptime expression yielding alignment value
  );

  let results = (outs);  // No SSA result - Symbol ops cannot have results

  let assemblyFormat = [{
    $sym_name
    (`dynamicType` `(` $dynamicType^ `)`)?
    (`const` $is_const^)?
    (`export` $is_export^)?
    (`extern` $is_extern^)?
    (`unbound_param` $is_unbound_param^)?
    (`linkname` `(` $linkname^ `)`)?
    (`align` `(` $align^ `)`)?
    attr-dict-with-keyword
    ($initRegion^)?
    (`linkname_expr` $linkname_region^)?
    (`align_expr` $align_region^)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Get the name of this global variable (for use in GetSymbolOp).
    StringRef getName() { return getSymName(); }

    /// Check if this global has an initializer (non-empty init region).
    bool hasInitializer() { return !getInitRegion().empty() || getInferredInitializer().has_value(); }

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

def ParsedCSL_GetSymbolOp : ParsedCSL_Op<"get_symbol", [
  InferredTypeInterface,
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ParsedCSL_Expression
]> {
  let summary = "Reference to a named symbol (function or global) in the same scope";
  let description = [{
    Emitted by the parser for any identifier that is not a local variable or
    builtin. References a global variable or function within the same module
    (or compilation unit, for top-level code). Cross-module references go
    through FieldAccessOp on the module handle instead.

    The op's position in the IR normally determines the lookup scope: MLIR's
    nearest-symbol-from lookup walks up to the enclosing symbol table
    (ModuleOp or CompilationUnitOp) and finds the symbol there. When an inline
    function body is cloned into a caller in another module, the optional
    lexical module identifier preserves the definition scope. Comptime eval
    then replaces the op with a ConstantOp containing GlobalRefAttr or
    FuncRefAttr accordingly.

    The verifier checks that the symbol exists (as a GlobalOp, GenericFuncOp,
    or FuncOp). Full semantic validation is deferred to comptime eval.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$name,
    OptionalAttr<I64Attr>:$lexical_module_identifier,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$name attr-dict";
}

//===----------------------------------------------------------------------===//
// Local Variable Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_VariableOp : ParsedCSL_Op<"var", [
  InferredTypeInterface,
  AttrSizedOperandSegments,
  ParsedCSL_Statement,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Variable declaration";
  let description = [{
    Declares a variable with optional initializer and optional type.

    - If `dynamicType` is provided, it specifies the variable's type as an SSA value
    - If `initializer` is provided, the variable is initialized with that value
    - Variables can be marked as `const` (immutable) or `comptime` (compile-time)

    Examples:
    ```mlir
    // Variable with explicit type
    %t = parsed_csl.builtin_type si32
    %v1 = parsed_csl.var "x" dynamicType(%t)

    // Variable with initializer (type inferred)
    %init = parsed_csl.constant 42 : si32
    %v2 = parsed_csl.var "y" init(%init)

    // Const variable
    %v3 = parsed_csl.var "z" const init(%init)
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    Optional<ParsedCSL_UnresolvedType>:$dynamicType,
    Optional<ParsedCSL_UnresolvedType>:$initializer,
    UnitAttr:$is_const,
    UnitAttr:$is_comptime,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{
    $name
    (`dynamicType` `(` $dynamicType^ `)`)?
    (`init` `(` $initializer^ `)`)?
    (`const` $is_const^)?
    (`comptime` $is_comptime^)?
    attr-dict
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def ParsedCSL_ReadOp : ParsedCSL_Op<"read", [
    InferredTypeInterface,
    ParsedCSL_Expression]> {
  let summary = "Value of an expression at this point of the evaluation";
  let description = [{
    The value that the source reads where it uses an expression as a value:
    if `source` denotes a location (a variable, a parameter, a field or
    element of one, or a dereferenced pointer), the current content of that
    location; otherwise `source` itself, e.g. a function, a type, a module or
    a constant.

    The parser emits a read right after the expression that a construct uses
    as a value, before it evaluates anything that follows. The position of
    the read in its block is thus where the source reads the value: in
    `.{ x, g(), x }` the first read of x is before the call and the second
    after it. The parser does not need to know whether `source` denotes a
    location, since the read is defined for both.

    Operand positions that take a location instead of a value, e.g. the
    target of an assignment or the operand of `&`, get no read; see
    `parsed_csl::isUnreadOperand`.

    Example:
    ```mlir
    %x = parsed_csl.var "x" init(%init)
    %v = parsed_csl.read %x
    ```
  }];

  let arguments = (ins
    Arg<ParsedCSL_UnresolvedType, "", [MemRead]>:$source,
    OptionalAttr<TypeAttr>:$inferred_type
  );
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$source attr-dict";

  let hasFolder = 1;
}

def ParsedCSL_LoadOp : ParsedCSL_Op<"load", [
    InferredTypeInterface,
    ParsedCSL_Expression]> {
  let summary = "Load the value of a location";
  let description = [{
    Loads the current value of a location, e.g. a variable. ResolveReadsPass
    turns each read of a location into a load once comptime evaluation has
    inferred the types; reads of anything else disappear. Afterwards, every
    location that is used as a value is loaded explicitly.

    Example:
    ```mlir
    %var = parsed_csl.var "x"
    %val = parsed_csl.load from %var
    ```
  }];

  let arguments = (ins
    Arg<ParsedCSL_UnresolvedType, "", [MemRead]>:$lvalue,
    OptionalAttr<TypeAttr>:$inferred_type
  );
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "`from` $lvalue attr-dict";

  let hasFolder = 1;
}

def ParsedCSL_LocOp : ParsedCSL_Op<"loc", [
    InferredTypeInterface,
    ParsedCSL_Expression,
    Pure]> {
  let summary = "Transparent location wrapper";
  let description = [{
    Wraps a value to attach a source location without changing semantics.
    Used in LSP mode so that references to local variables (which are plain
    SSA values without their own operation) get an operation with a location
    that the LSP server can map to hover/go-to-definition requests.

    The operation is a semantic identity: it forwards the operand's type
    and comptime value unchanged.

    Example:
    ```mlir
    %x = parsed_csl.var "x" ...
    %x_ref = parsed_csl.loc %x
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$value,
    OptionalAttr<TypeAttr>:$inferred_type
  );
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$value attr-dict";
}

def ParsedCSL_SyntaxErrorOp : ParsedCSL_Op<"syntax_error", [
    InferredTypeInterface,
    ParsedCSL_Expression,
    Pure]> {
  let summary = "Placeholder for an expression that failed during parsing";
  let description = [{
    Represents a source expression that the parser could not emit while
    recovering from an earlier syntax error.

    During comptime evaluation, this op always produces a SemanticError,
    which propagates to any operation that depends on its result. In LSP
    mode, these errors are non-fatal and analysis continues past them.

    Example:
    ```mlir
    %err = parsed_csl.syntax_error "some raw text"
    %x = parsed_csl.var "x" init(%err)
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$error_text
  );
  let results = (outs Optional<ParsedCSL_UnresolvedType>:$result);

  let assemblyFormat = "($error_text^)? attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    // Implement InferredTypeInterface: result is always TypeType
    ::std::optional<::mlir::Type> getInferredType() {
      return std::nullopt;
    }

    void setInferredType(::std::optional<::mlir::Type> attrValue) {
      llvm_unreachable("Cannot set inferred type on SyntaxErrorOp");
    }
  }];
}

def ParsedCSL_DeferredErrorOp : ParsedCSL_Op<"deferred_error", [
    InferredTypeInterface,
    ParsedCSL_Expression,
    Pure]> {
  let summary = "Deferred error that fires during comptime evaluation";
  let description = [{
    Represents a semantic error detected during parsing that is deferred
    to comptime evaluation. If the result of this op is used at comptime,
    it produces a SemanticError with the stored message.

    Used for:
    - Unresolved identifiers in system modules (-Wunresolved-system-symbol=warn)

    Example:
    ```mlir
    %err = parsed_csl.deferred_error "unknown identifier: foo"
    ```
  }];

  let arguments = (ins StrAttr:$message);
  let results = (outs Optional<ParsedCSL_UnresolvedType>:$result);
  let assemblyFormat = "$message attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    ::std::optional<::mlir::Type> getInferredType() { return std::nullopt; }
    void setInferredType(::std::optional<::mlir::Type> attrValue) {
      llvm_unreachable("Cannot set inferred type on DeferredErrorOp");
    }
  }];
}

def ParsedCSL_AssignOp : ParsedCSL_Op<"assign", [ParsedCSL_Statement]> {
  let summary = "Assign value to lvalue";
  let description = [{
    Assigns a value to an lvalue (variable).
    The lvalue must be mutable (not const).

    Example:
    ```mlir
    %var = parsed_csl.var "x"
    %val = parsed_csl.constant 42 : si32
    parsed_csl.assign %val to %var
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$lvalue,
    ParsedCSL_UnresolvedType:$value
  );

  let assemblyFormat = "$value `to` $lvalue attr-dict";
}

def ParsedCSL_InlineAsmOp : ParsedCSL_Op<"inline_asm", [
    ParsedCSL_Statement, AttrSizedOperandSegments]> {
  let summary = "Inline assembly statement";
  let description = [{
    Represents a CSL inline assembly statement.
    The asm_string, constraints, and clobbers are SSA operands (comptime string
    expressions). Names are plain identifier attributes. Outputs are lvalue
    operands, inputs are rvalue operands.
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$asm_string,
    Variadic<ParsedCSL_UnresolvedType>:$output_constraints,
    Variadic<ParsedCSL_UnresolvedType>:$outputs,
    Variadic<ParsedCSL_UnresolvedType>:$input_constraints,
    Variadic<ParsedCSL_UnresolvedType>:$inputs,
    Variadic<ParsedCSL_UnresolvedType>:$clobbers,
    StrArrayAttr:$output_names,
    StrArrayAttr:$input_names,
    UnitAttr:$is_volatile,
    // Inferred during comptime eval from the SSA string operands
    OptionalAttr<StrAttr>:$inferred_asm_string,
    OptionalAttr<StrArrayAttr>:$inferred_output_constraints,
    OptionalAttr<StrArrayAttr>:$inferred_input_constraints,
    OptionalAttr<StrArrayAttr>:$inferred_clobbers
  );

  let hasVerifier = 1;
}

def ParsedCSL_BinaryOp : ParsedCSL_Op<"binary", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Binary operation";
  let description = [{
    Performs a binary operation on two operands.
    The operation kind is specified by the BinaryOpKind attribute.

    Examples:
    ```mlir
    %sum = parsed_csl.binary add, %lhs, %rhs
    %cmp = parsed_csl.binary lt, %a, %b
    ```
  }];

  let arguments = (ins
    CSL_BinaryOpKind:$kind,
    ParsedCSL_UnresolvedType:$lhs,
    ParsedCSL_UnresolvedType:$rhs,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$kind `,` $lhs `,` $rhs attr-dict";

  let hasFolder = 1;

}

def ParsedCSL_UnaryOp : ParsedCSL_Op<"unary", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Unary operation";
  let description = [{
    Performs a unary operation on one operand.
    The operation kind is specified by the UnaryOpKind attribute.

    Supported operations:
    - neg: Arithmetic negation (-x)
    - bitnot: Bitwise complement (~x)
    - lnot: Logical negation (!x)

    Examples:
    ```mlir
    %negated = parsed_csl.unary neg, %value
    %inverted = parsed_csl.unary bitnot, %bits
    %negbool = parsed_csl.unary lnot, %bool_val
    ```
  }];

  let arguments = (ins
    CSL_UnaryOpKind:$kind,
    ParsedCSL_UnresolvedType:$operand,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$kind `,` $operand attr-dict";

  let hasFolder = 1;

}

def ParsedCSL_CastOp : ParsedCSL_Op<"cast", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Type cast operation";
  let description = [{
    Converts a value from one type to another.
    Supports casts between comptime types and runtime types.

    This operation is used to represent explicit type conversions,
    typically from @as() builtin function calls.

    The 'implicit' attribute marks casts that were automatically inserted
    by the compiler (e.g., during normalization) rather than written
    explicitly by the user.

    Examples:
    ```mlir
    %f32_type = parsed_csl.builtin_type f32
    %i32_val = parsed_csl.cast %value to %i32_type
    %auto_cast = parsed_csl.cast %comptime_val to %i32_type {implicit}
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$source,
    ParsedCSL_UnresolvedType:$targetType,
    OptionalAttr<TypeAttr>:$inferred_type,
    UnitAttr:$implicit
  );
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$source `to` $targetType attr-dict";

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Check if this cast was implicitly inserted by the compiler.
    bool isImplicit() { return getImplicit(); }

  }];
}

def ParsedCSL_AddressOfOp : ParsedCSL_Op<"address_of", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Take address of an lvalue";
  let description = [{
    Takes the address of an lvalue (variable, field, array element, etc.)
    and produces a pointer to that location.

    Operand must be an addressable lvalue (not a temporary).
    Result type is inferred during comptime evaluation as !csl.ptr<T>.

    If the operand is a const lvalue, the result is a const pointer.

    Examples:
    ```mlir
    %var = parsed_csl.var "x" init(%val)
    %ptr = parsed_csl.address_of %var  // Result: !csl.ptr<valueType>

    %const_var = parsed_csl.var "y" const init(%val)
    %const_ptr = parsed_csl.address_of %const_var  // Result: !csl.ptr<const valueType>
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$operand,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$operand attr-dict";
}

def ParsedCSL_DereferenceOp : ParsedCSL_Op<"dereference", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Dereference a pointer to access the pointed-to value";
  let description = [{
    Dereferences a pointer to produce an lvalue to the pointed-to location.

    Operand must have pointer type (!csl.ptr<T>).
    Result is an lvalue (!csl.lvalue<T>) allowing both reading and writing
    (if the pointer is not const).

    Examples:
    ```mlir
    %ptr = ...  // !csl.ptr<si32>
    %lval = parsed_csl.dereference %ptr  // Result: !csl.lvalue<mutable si32>
    %val = parsed_csl.load from %lval

    %const_ptr = ...  // !csl.ptr<const si32>
    %const_lval = parsed_csl.dereference %const_ptr  // Result: !csl.lvalue<const si32>
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$operand,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$operand attr-dict";
}

def ParsedCSL_SelectOp : ParsedCSL_Op<"select", [
    InferredTypeInterface, ParsedCSL_Expression,
    RecursivelySpeculatable, RecursiveMemoryEffects]> {
  let summary = "Conditional select operation";
  let description = [{
    Selects between two branch regions based on a condition (ternary operator /
    if expression). This is an expression-level operation that produces a value,
    as opposed to the statement-level `parsed_csl.if` operation.

    Corresponds to CSL's if-expression syntax: `if (cond) true_val else false_val`
    or ternary operator: `cond ? true_val : false_val`

    Both branches are regions that are only entered when needed. This enables
    proper short-circuit evaluation where only the taken branch is evaluated,
    both at comptime and runtime.

    If the else region is not provided, this represents an if-expression without
    else. Such expressions are only valid when the condition is comptime-known
    to be true. Similiarly, if the then region is not provided, the condition must be
    comptime-known to be false.

    Examples:
    ```mlir
    // With else branch:
    %result = parsed_csl.select %cond then {
      %true_val = ...
      parsed_csl.yield %true_val
    } else {
      %false_val = ...
      parsed_csl.yield %false_val
    }
    // Without else branch (requires comptime-true condition):
    %result = parsed_csl.select %cond then {
      %true_val = ...
      parsed_csl.yield %true_val
    }
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$condition,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let regions = (region AnyRegion:$then_region, AnyRegion:$else_region);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$condition (`then` $then_region^)? (`else` $else_region^)? attr-dict";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Returns true if this select has a then branch (non-empty then_region)
    bool hasThenBranch() { return !getThenRegion().empty(); }

    /// Returns true if this select has an else branch (non-empty else_region)
    bool hasElseBranch() { return !getElseRegion().empty(); }
  }];
}

def ParsedCSL_ShortCircuitOp : ParsedCSL_Op<"short_circuit", [
    InferredTypeInterface, ParsedCSL_Expression,
    RecursivelySpeculatable, RecursiveMemoryEffects]> {
  let summary = "Short-circuit logical operation";
  let description = [{
    Represents short-circuiting logical AND (&&) or OR (||) operations.

    The LHS operand is always evaluated. The RHS region is only entered when
    needed based on the LHS value:
    - For OR (||): RHS region is evaluated only if LHS is false
    - For AND (&&): RHS region is evaluated only if LHS is true

    The result type is always boolean (i1), following C-style semantics.

    This enables proper short-circuit evaluation where the RHS is not evaluated
    if the result can be determined from the LHS alone. Any side effects in the
    RHS region (including comptime assertions) will not occur if short-circuited.

    Examples:
    ```mlir
    // Logical OR: evaluates RHS only if LHS is false
    %result = parsed_csl.short_circuit or, %lhs {
      %rhs = ...
      parsed_csl.yield %rhs
    }

    // Logical AND: evaluates RHS only if LHS is true
    %result = parsed_csl.short_circuit and, %lhs {
      %rhs = ...
      parsed_csl.yield %rhs
    }
    ```
  }];

  let arguments = (ins
    CSL_ShortCircuitKind:$kind,
    ParsedCSL_UnresolvedType:$lhs,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let regions = (region SizedRegion<1>:$rhs_region);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$lhs $kind $rhs_region attr-dict";
}

//===----------------------------------------------------------------------===//
// Type Construction Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_BuiltinTypeOp : ParsedCSL_Op<"builtin_type",
    [InferredTypeInterface, ConstantLike, Pure, ParsedCSL_Expression]> {
  let summary = "Construct builtin type value";
  let description = [{
    Creates a type value representing a builtin type (i32, f32, bool, etc.).

    The type is encoded as an attribute and the result is an SSA value that
    can be used in type positions. In ParsedCSL, types are values like any
    other value, and all values have unresolved static type.

    Examples:
    ```mlir
    %i32 = parsed_csl.builtin_type si32
    %f32 = parsed_csl.builtin_type f32
    %bool = parsed_csl.builtin_type i1
    ```
  }];

  let arguments = (ins CSL_TypeAttr:$value);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$value attr-dict";

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Implement InferredTypeInterface: result is always TypeType
    ::std::optional<::mlir::Type> getInferredType() {
      return csl::TypeType::get(getContext());
    }

    void setInferredType(::std::optional<::mlir::Type> attrValue) {
      if (attrValue && *attrValue != csl::TypeType::get(getContext()))
        llvm_unreachable("Cannot set inferred type on BuiltinTypeOp: type is always TypeType");
      // Otherwise no-op: type is always TypeType
    }
  }];
}

def ParsedCSL_PtrTypeOp : ParsedCSL_Op<"ptr_type", [Pure, ParsedCSL_Expression]> {
  let summary = "Construct pointer type value";
  let description = [{
    Creates a type value of pointer type contains the memory address 
    where a variable is located.

    There are two kinds of pointers in CSL: Single-item pointers and many-item pointers.
    - Single-item pointers point to a single element of the pointee type.
    - Many-item pointers point to an unknown number of elements of the pointee type

    Example:
    ```mlir
    %i32 = parsed_csl.builtin_type si32
    
    // Single-item pointer:
    %ptr_type = parsed_csl.ptr_type %i32 // *i32
    %array_type = parsed_csl.array_type 10 x %i32 // [10]i32
    %array_ptr_type = parsed_csl.ptr_type const %array_type // *const [10]i32

    // Many-item pointer:
    %variadic_ptr_type = parsed_csl.ptr_type ? x %i32 // [*]i32
    %variadic_arr_ptr = parsed_csl.ptr_type %i32 const ? x %i32 // [*]const i32
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$pointee,
    UnitAttr:$is_const,
    UnitAttr:$is_many_ptr
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{
    (`const` $is_const^)?
    (`?` `x` $is_many_ptr^)?
    $pointee
    attr-dict
  }];
}

def ParsedCSL_ArrayTypeOp : ParsedCSL_Op<"array_type", [Pure, ParsedCSL_Expression]> {
  let summary = "Construct array type value";
  let description = [{
    Creates a type value representing an array with specified dimensions.

    Dimensions are provided as SSA values (allowing comptime-computed sizes).

    Example:
    ```mlir
    %i32 = parsed_csl.builtin_type si32
    %d1 = parsed_csl.constant #csl.comptime_int<10> : !csl.comptime_int
    %d2 = parsed_csl.constant #csl.comptime_int<20> : !csl.comptime_int
    %arr = parsed_csl.array_type %i32, %d1, %d2
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$element_type,
    Variadic<ParsedCSL_UnresolvedType>:$dimensions
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{
    $element_type
    (`,` $dimensions^)?
    attr-dict
  }];
}

def ParsedCSL_FuncTypeOp : ParsedCSL_Op<"func_type", [Pure, AttrSizedOperandSegments, ParsedCSL_Expression]> {
  let summary = "Construct function type value";
  let description = [{
    Creates a type value representing a function type with parameter and return types.

    Example:
    ```mlir
    %i32 = parsed_csl.builtin_type si32
    %func_ty = parsed_csl.func_type (%i32, %i32) -> (%i32)
    ```
  }];

  let arguments = (ins
    Variadic<ParsedCSL_UnresolvedType>:$param_types,
    Variadic<ParsedCSL_UnresolvedType>:$return_types,
    UnitAttr:$is_task,
    UnitAttr:$is_async
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{
    (`task` $is_task^)? (`async` $is_async^)? `(` $param_types `)` `->` `(` $return_types `)` attr-dict
  }];
}

def ParsedCSL_NamedStructTypeOp : ParsedCSL_Op<"named_struct_type", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Construct named struct type value";
  let description = [{
    Creates a type value for a named struct with nominal typing.

    Named structs are identified by their definition location - this operation
    should appear at the location where the struct is defined in the source.
    The def_location attribute captures this location for type identity.

    The field types are provided as SSA values, and field names are specified
    alongside each field type.

    Example:
    ```mlir
    %f16 = parsed_csl.builtin_type f16
    %complex_type = parsed_csl.named_struct_type loc("./pe_program.csl":92:3) real : %f16, imag : %f16
    ```
  }];

  let arguments = (ins
    StrArrayAttr:$field_names,
    Variadic<ParsedCSL_UnresolvedType>:$field_types,
    LocationAttr:$def_location
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Implement InferredTypeInterface: result is always TypeType
    ::std::optional<::mlir::Type> getInferredType() {
      return csl::TypeType::get(getContext());
    }

    void setInferredType(::std::optional<::mlir::Type> attrValue) {
      if (attrValue && *attrValue != csl::TypeType::get(getContext()))
        llvm_unreachable("Cannot set inferred type on NamedStructTypeOp: type is always TypeType");
      // Otherwise no-op: type is always TypeType
    }
  }];
}

//===----------------------------------------------------------------------===//
// Enum Type Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_EnumTypeOp : ParsedCSL_Op<"enum_type", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Construct enum type value";
  let description = [{
    Creates a type value for an enum type with nominal typing.

    Enum types are identified by their definition location - this operation
    should appear at the location where the enum is defined in the source.
    The def_location attribute captures this location for type identity.

    The underlying_type operand specifies the integer type used to represent
    enum values (e.g., u16, i32). Element names are string attributes, and
    explicit_values are optional SSA values for elements with explicit values.
    Elements without explicit values will be auto-assigned during comptime
    evaluation.

    Example:
    ```mlir
    %u16 = parsed_csl.builtin_type ui16
    %two = parsed_csl.constant #csl.comptime_int<2>
    // enum(u16) { red, white = 2, blue }
    %enum_type = parsed_csl.enum_type %u16 ["red", "white", "blue"] [%two]
                    {explicit_indices = [1]}
                    def_location(loc("test.csl":5:3))
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$underlying_type,
    StrArrayAttr:$element_names,
    Variadic<ParsedCSL_UnresolvedType>:$explicit_values,
    I64ArrayAttr:$explicit_indices,
    LocationAttr:$def_location
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Implement InferredTypeInterface: result is always TypeType
    ::std::optional<::mlir::Type> getInferredType() {
      return csl::TypeType::get(getContext());
    }

    void setInferredType(::std::optional<::mlir::Type> attrValue) {
      if (attrValue && *attrValue != csl::TypeType::get(getContext()))
        llvm_unreachable("Cannot set inferred type on EnumTypeOp: type is always TypeType");
      // Otherwise no-op: type is always TypeType
    }
  }];
}

//===----------------------------------------------------------------------===//
// Struct Construction Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_ConstructTupleOp : ParsedCSL_Op<"construct_tuple", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Construct tuple value";
  let description = [{
    Creates a tuple value with unnamed fields.

    Tuples are structs with positional fields (no field names).
    Field values are provided as SSA operands.

    Example:
    ```mlir
    %x = parsed_csl.constant 10 : si32
    %y = parsed_csl.constant 3.14 : f32
    %tuple = parsed_csl.construct_tuple %x, %y
    ```
  }];

  let arguments = (ins
    Variadic<ParsedCSL_UnresolvedType>:$field_values,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$field_values attr-dict";
}

def ParsedCSL_ConstructStructOp : ParsedCSL_Op<"construct_struct", [
    InferredTypeInterface, Pure, ParsedCSL_Expression, AttrSizedOperandSegments]> {
  let summary = "Construct struct value with named fields";
  let description = [{
    Creates a struct value with named fields. Supports both anonymous structs
    (structural typing) and named structs (nominal typing).

    Anonymous structs use structural typing - two structs with the same
    field names and types are considered equal regardless of where they
    were defined. Field names are specified as a string array attribute.

    Named structs use nominal typing - they are identified by their
    definition location. An optional struct_type operand (from named_struct_type
    op) carries the nominal type identity.

    Example (anonymous struct):
    ```mlir
    %real = parsed_csl.constant 1.0 : f16
    %imag = parsed_csl.constant 2.0 : f16
    %complex = parsed_csl.construct_struct ["real", "imag"] %real, %imag
    ```

    Example (named struct):
    ```mlir
    %complex_type = parsed_csl.named_struct_type ["real", "imag"] (%f16, %f16)
                      def_location(loc("./pe_program.csl":92:3))
    %real = parsed_csl.constant 1.0 : f16
    %imag = parsed_csl.constant 2.0 : f16
    %complex = parsed_csl.construct_struct ["real", "imag"] (%complex_type) %real, %imag
    ```
  }];

  let arguments = (ins
    StrArrayAttr:$field_names,
    Optional<ParsedCSL_UnresolvedType>:$struct_type,
    Variadic<ParsedCSL_UnresolvedType>:$field_values,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$field_names (`(` $struct_type^ `)`)? $field_values attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Array Construction Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_ConstructArrayOp : ParsedCSL_Op<"construct_array", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Construct array value from element values";
  let description = [{
    Creates an array value from individual element values.

    The array_type operand specifies the target array type as an SSA value.
    The elements are provided as SSA operands in row-major (flattened) order.

    For multi-dimensional arrays, elements are listed in row-major order.
    For example, `[2, 2]i32{1, 2, 3, 4}` creates a 2x2 array:
      [[1, 2], [3, 4]]

    Example:
    ```mlir
    %arr_type = parsed_csl.array_type %i32, %dim3
    %e1 = parsed_csl.constant 1 : si32
    %e2 = parsed_csl.constant 2 : si32
    %e3 = parsed_csl.constant 3 : si32
    %arr = parsed_csl.construct_array (%arr_type) %e1, %e2, %e3
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$array_type,
    Variadic<ParsedCSL_UnresolvedType>:$elements,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "`(` $array_type `)` $elements attr-dict";
}

//===----------------------------------------------------------------------===//
// Struct Access Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_FieldAccessOp : ParsedCSL_Op<"field_access", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Access struct field by name";
  let description = [{
    Accesses a named field of a struct or module.

    For struct lvalues, this produces an lvalue to the specified field,
    allowing for nested field access and supporting both reading (via load)
    and writing (via assign). For rvalue structs, it extracts the field value.

    For modules, this accesses module members (variables, functions, etc.).
    The result type is determined during comptime evaluation based on the
    actual field type.

    Works with both anonymous and named structs (not tuples).

    Example:
    ```mlir
    %struct_lval = parsed_csl.var "my_struct" init(%struct_literal)
    %field = parsed_csl.field_access %struct_lval["x"]
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$base,
    StrAttr:$field_name,
    OptionalAttr<TypeAttr>:$inferred_type,
    OptionalAttr<TypeAttr>:$inferred_base_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$base `[` $field_name `]` attr-dict";

  let hasFolder = 1;
}

def ParsedCSL_IndexAccessOp : ParsedCSL_Op<"index_access", [
    InferredTypeInterface, Pure, ParsedCSL_Expression]> {
  let summary = "Access array/tuple/struct element by index";
  let description = [{
    Accesses an array, tuple, or struct element by numeric index.

    For lvalue bases, this produces an lvalue to the element at the specified
    index. For rvalue bases, it extracts the element value. Indices are
    provided as SSA values to support comptime-computed indices.

    For multi-dimensional arrays, multiple indices can be provided to access
    nested elements in a single operation (e.g., arr[i, j]).

    Works with arrays, tuples, and structs. For named structs, accesses fields
    in declaration order.

    Example:
    ```mlir
    // Single index access
    %idx = parsed_csl.constant #csl.comptime_int<0>
    %tuple_lval = parsed_csl.var "my_tuple" init(%tuple_literal)
    %elem = parsed_csl.index_access %tuple_lval[%idx]

    // Multi-dimensional array access
    %i = parsed_csl.constant #csl.comptime_int<1>
    %j = parsed_csl.constant #csl.comptime_int<2>
    %elem2d = parsed_csl.index_access %array_2d[%i, %j]
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$base,
    Variadic<ParsedCSL_UnresolvedType>:$indices,
    OptionalAttr<TypeAttr>:$inferred_type,
    OptionalAttr<DenseI64ArrayAttr>:$constant_indices
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$base `[` $indices `]` attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_GenericFuncOp : ParsedCSL_Op<"generic_func", [
    IsolatedFromAbove,
    Symbol,
    SymbolName,
    NoTerminator
]> {
  let summary = "Generic function definition (template form)";
  let description = [{
    Defines a generic function with a type computation region and body region.
    This is the representation produced by the parser for ALL functions.

    The type_region computes the types of all parameters and the return type.
    It has block arguments corresponding to all function parameters (as
    !parsed_csl.unresolved values). The region is terminated by a type_yield
    op that yields (param_type_0, param_type_1, ..., param_type_n, return_type).

    The body region contains the function implementation.

    For generic functions (those with comptime params or anytype), the
    comptime_params attribute lists the indices of comptime parameters.

    During comptime evaluation, GenericFuncOp is instantiated into FuncOp
    with concrete types.

    Example (non-generic):
    ```mlir
    parsed_csl.generic_func @add {
      ^types(%a: !parsed_csl.unresolved, %b: !parsed_csl.unresolved):
        %i32 = parsed_csl.builtin_type si32
        parsed_csl.type_yield %i32, %i32, %i32
    } {
      ^body(%a: !parsed_csl.unresolved, %b: !parsed_csl.unresolved):
        %r = parsed_csl.binary add, %a, %b
        parsed_csl.return %r
    }
    ```

    Example (generic with comptime type param):
    ```mlir
    // fn identity(comptime T: type, x: T) T
    parsed_csl.generic_func @identity comptime_params = [0] {
      ^types(%T: !parsed_csl.unresolved, %x: !parsed_csl.unresolved):
        %type_type = parsed_csl.builtin_type type
        parsed_csl.type_yield %type_type, %T, %T
    } {
      ^body(%T: !parsed_csl.unresolved, %x: !parsed_csl.unresolved):
        parsed_csl.return %x
    }
    ```

    Example (anytype using @type_of builtin):
    ```mlir
    // fn ignore_x(x: anytype, y: @type_of(x)) @type_of(x)
    parsed_csl.generic_func @ignore_x comptime_params = [] {
      ^types(%x: !parsed_csl.unresolved, %y: !parsed_csl.unresolved):
        %x_type = parsed_csl.builtin_call type_of(%x)
        parsed_csl.type_yield %x_type, %x_type, %x_type
    } {
      ^body(%x: !parsed_csl.unresolved, %y: !parsed_csl.unresolved):
        parsed_csl.return %y
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DenseI32ArrayAttr:$comptime_params,
    DenseI32ArrayAttr:$anytype_params,
    I32Attr:$num_params,
    UnitAttr:$is_task,
    UnitAttr:$is_inline,
    UnitAttr:$is_noinline,
    UnitAttr:$is_export,
    UnitAttr:$is_extern,
    UnitAttr:$is_async,
    OptionalAttr<UI32Attr>:$stack_size,
    OptionalAttr<StrArrayAttr>:$param_names,
    OptionalAttr<StrAttr>:$doc_string,
    OptionalAttr<UI32Attr>:$max_recursion_depth
  );

  let regions = (region
    SizedRegion<1>:$type_region,
    AnyRegion:$body,  // Empty for extern functions (declarations)
    AnyRegion:$linkname_region  // Optional: comptime expression yielding comptime_string
  );

  let assemblyFormat = [{
    $sym_name
    (`task` $is_task^)?
    (`inline` $is_inline^)?
    (`noinline` $is_noinline^)?
    (`export` $is_export^)?
    (`extern` $is_extern^)?
    (`async` $is_async^)?
    (`stack_size` `(` $stack_size^ `)`)?
    (`max_recursion_depth` `(` $max_recursion_depth^ `)`)?
    $type_region $body (`linkname` $linkname_region^)? attr-dict
  }];

  let extraClassDeclaration = [{
    /// Returns block arguments of the body region (parameter values).
    Block::BlockArgListType getBodyArgs();

    /// Returns block arguments of the type region (parameter references for type computation).
    Block::BlockArgListType getTypeRegionArgs();

    bool isActuallyGeneric() {
      return !getComptimeParams().empty() || !getAnytypeParams().empty();
    }

    /// Generic functions are templates that remain visible until instantiated.
    mlir::SymbolTable::Visibility getVisibility() {
      return mlir::SymbolTable::Visibility::Public;
    }
    void setVisibility(mlir::SymbolTable::Visibility visibility) {
      assert(visibility == mlir::SymbolTable::Visibility::Public &&
             "parsed_csl.generic_func symbols are always public");
    }
  }];

  let hasVerifier = 1;
}

def ParsedCSL_FuncOp : ParsedCSL_Op<"func", [
    IsolatedFromAbove,
    Symbol,
    SymbolName,
    NoTerminator,
    AutomaticAllocationScope
]> {
  let summary = "Instantiated function definition";
  let description = [{
    Represents a concrete function with fully resolved types.
    This is the result of instantiating a GenericFuncOp.

    The function_type attribute contains the concrete MLIR FunctionType
    specifying parameter and return types.

    Example:
    ```mlir
    parsed_csl.func @add {
      function_type = (si32, si32) -> si32
    } {
      ^body(%a: !parsed_csl.unresolved, %b: !parsed_csl.unresolved):
        %r = parsed_csl.binary add, %a, %b
        parsed_csl.return %r
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    StrAttr:$original_name,
    UnitAttr:$is_task,
    UnitAttr:$is_inline,
    UnitAttr:$is_noinline,
    UnitAttr:$is_export,
    UnitAttr:$is_extern,
    UnitAttr:$is_async,
    OptionalAttr<UI32Attr>:$stack_size,
    OptionalAttr<StrAttr>:$linkname,
    OptionalAttr<StrArrayAttr>:$param_names,
    OptionalAttr<StrAttr>:$doc_string,
    OptionalAttr<TypeAttr>:$async_result_type,
    OptionalAttr<UI32Attr>:$max_recursion_depth
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $sym_name
    (`task` $is_task^)?
    (`inline` $is_inline^)?
    (`noinline` $is_noinline^)?
    (`export` $is_export^)?
    (`extern` $is_extern^)?
    (`async` $is_async^)?
    (`stack_size` `(` $stack_size^ `)`)?
    (`max_recursion_depth` `(` $max_recursion_depth^ `)`)?
    (`linkname` `(` $linkname^ `)`)?
    $body attr-dict
  }];

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

    /// Returns the number of parameters from the function type.
    unsigned getNumParams();

    /// Returns the parameter types from the function type.
    ArrayRef<Type> getParamTypes();

    /// Returns the type that the function body returns (what ReturnOps carry).
    /// For async functions, this is async_result_type; for non-async, it is
    /// function_type's result. Always returns a valid type (VoidType for void).
    Type getBodyReturnType();

    /// Returns the type that calling this function produces.
    /// For async functions, this is CoroFrameType; for non-async, it is
    /// function_type's result.
    Type getCallableReturnType();
  }];

  let hasVerifier = 1;
}

def ParsedCSL_ReturnOp : ParsedCSL_Op<"return", [
    ParsedCSL_Statement,
    ParsedCSL_PseudoTerminator
    // HasParent<"FuncOp">,
    // Terminator
]> {
  let summary = "Return from function";
  let description = [{
    Returns zero or one value from a function.

    Example:
    ```mlir
    parsed_csl.return %val
    ```
  }];

  let arguments = (ins Optional<ParsedCSL_UnresolvedType>:$operand);

  let assemblyFormat = "($operand^)? attr-dict";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, {});
    }]>
  ];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ParsedCSL_UnreachableOp : ParsedCSL_Op<"unreachable", [
    ParsedCSL_Statement,
    ParsedCSL_PseudoTerminator
]> {
  let summary = "Unreachable pseudo-terminator";
  let description = [{
    Marks a point that is unreachable because all control flow paths in the
    function body have already returned. Inserted by ComptimeEvalPass at the
    end of function bodies where every path contains a return statement.

    This is NOT a Terminator in ParsedCSL because ParsedCSL FuncOp uses
    NoTerminator. It becomes a Terminator after lowering to TypedCSL.

    Example:
    ```mlir
    parsed_csl.unreachable
    ```
  }];

  let assemblyFormat = "attr-dict";
  let hasCanonicalizer = 1;
}

def ParsedCSL_SuspendOp : ParsedCSL_Op<"suspend", [
    ParsedCSL_Statement
]> {
  let summary = "Coroutine suspension point";
  let description = [{
    Suspends the current coroutine, yielding control back to the caller.
    Only valid inside an `async fn`.

    In ParsedCSL, this is NOT a Terminator because ParsedCSL FuncOp uses
    NoTerminator. It becomes a Terminator after lowering to TypedCSL.

    Example:
    ```mlir
    parsed_csl.suspend
    ```
  }];

  let assemblyFormat = "attr-dict";
}

def ParsedCSL_AwaitOp : ParsedCSL_Op<"await", [
    InferredTypeInterface,
    ParsedCSL_Expression
]> {
  let summary = "Await a coroutine frame handle";
  let description = [{
    Awaits completion of a coroutine represented by the given frame handle.
    The result is the return value of the awaited coroutine.

    Example:
    ```mlir
    %result = parsed_csl.await %frame
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$frame_handle,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$frame_handle attr-dict";
}

def ParsedCSL_AwaitParallelOp : ParsedCSL_Op<"await_parallel", [
    ParsedCSL_Statement
]> {
  let summary = "Await multiple coroutines concurrently";
  let description = [{
    Runs multiple async function calls concurrently and suspends the caller
    until all of them complete. Each callee gets a dedicated worker task.

    Does not return a value. Parallel callees must write results through
    pointers passed as arguments.

    Example:
    ```mlir
    parsed_csl.await_parallel %send_future, %recv_future
    ```
  }];

  let arguments = (ins
    Variadic<ParsedCSL_UnresolvedType>:$futures
  );

  let assemblyFormat = "$futures attr-dict";
}

def ParsedCSL_CallOp : ParsedCSL_Op<"call", [
  InferredTypeInterface,
  // Only comptime calls are speculatable
  DeclareOpInterfaceMethods<ConditionallySpeculatable>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Function call (direct or indirect)";
  let description = [{
    Calls a function with the given arguments.

    The callee operand can be:
    - Result of GetSymbolOp (for direct calls to known functions)
    - A function pointer value (*const fn(...) ...)
    - A function reference value (fn(...) ...)

    For direct calls, the parser emits GetSymbolOp to produce the callee.
    For module.func() calls, FieldAccessOp on the module produces the callee.

    Example:
    ```mlir
    // Direct call via GetSymbolOp
    %func = parsed_csl.get_symbol @add
    %result = parsed_csl.call %func(%a, %b)

    // Indirect call via function pointer
    %result = parsed_csl.call %func_ptr(%x, %y)
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$callee,
    Variadic<ParsedCSL_UnresolvedType>:$args,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$callee `(` $args `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_IfOp : ParsedCSL_Op<"if", [
  NoTerminator, 
  ParsedCSL_Statement,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  AutomaticAllocationScope]> {
  let summary = "Conditional control flow";
  let description = [{
    Executes the then region if condition is true, otherwise executes the
    optional else region.

    The condition has unresolved static type and will be type-checked to
    boolean during the type checking phase.

    Example with else:
    ```mlir
    parsed_csl.if %cond {
      parsed_csl.var "x" init(%val)
    } else {
      parsed_csl.var "y" init(%other)
    }
    ```

    Example without else:
    ```mlir
    parsed_csl.if %cond {
      parsed_csl.var "x" init(%val)
    }
    ```
  }];

  let arguments = (ins ParsedCSL_UnresolvedType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let assemblyFormat = [{
    $condition $thenRegion (`else` $elseRegion^)? attr-dict
  }];
}

def ParsedCSL_SwitchOp : ParsedCSL_Op<"switch", [
  InferredTypeInterface,
  NoTerminator,
  RecursivelySpeculatable,
  RecursiveMemoryEffects]> {
  let summary = "Switch statement/expression";
  let description = [{
    Switch that matches input against case values. Can be used as a statement
    (inferred type is void) or expression (inferred type is common type of yields).

    Each case region corresponds to one entry in case_operands (grouped by
    case_operand_segments). Each entry is a list of SSA values representing
    case values for that branch. Regions that complete normally must yield a
    value via YieldOp. The case_values attribute is populated during comptime
    evaluation with the resolved constant values.

    When used as a statement, all regions yield void.
    When used as an expression, all regions yield values of compatible types.

    Note: This op has the NoTerminator trait because case regions may contain
    control flow statements (return, break, continue) that transfer control
    elsewhere without yielding a value. These CSL control flow ops are not
    MLIR terminators, so regions ending with them have no terminator. Dead
    code (including trailing YieldOp) after such control flow is removed by
    the comptime pass.

    Requirements:
    - Input must be a fixed-width integer or enum type (not comptime_int)
    - Case values must be comptime-known, coercible to input type, and unique
    - All values must be covered (explicitly or via else)
    - No fallthrough between cases
    - If input is comptime-known, only matching branch is semantically checked

    Assembly format:
      [%result =] parsed_csl.switch(<input>) cases(<case_operands>) <regions> [default <region>]

    The "default" keyword introduces the fallback region.
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$input,
    Variadic<ParsedCSL_UnresolvedType>:$case_operands,
    DenseI32ArrayAttr:$case_operand_segments,
    OptionalAttr<ArrayAttr>:$case_values,  // Filled during comptime: array of arrays of TypedAttr
    OptionalAttr<TypeAttr>:$inferred_type
  );

  // Note: else_region comes first in TableGen because MLIR requires variadic regions last.
  // The custom assembly format prints/parses in the natural order (cases, then else).
  let regions = (region
    AnyRegion:$else_region,
    VariadicRegion<AnyRegion>:$case_regions
  );

  let results = (outs ParsedCSL_UnresolvedType:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns true if this switch has an else branch
    bool hasElseBranch() { return !getElseRegion().empty(); }

    /// Get the number of case branches (not including else)
    unsigned getNumCases() { return getCaseRegions().size(); }
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$input,
      "ValueRange":$caseOperands,
      "ArrayRef<int32_t>":$caseOperandSegments,
      "unsigned":$numCases)>
  ];
}

def ParsedCSL_ForOp : ParsedCSL_Op<"for", [
  NoTerminator,
  ParsedCSL_Statement,
  CSL_LabeledOpInterface,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  AutomaticAllocationScope]> {
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
    parsed_csl.for(%range) {elem_name = "x"} {
    ^bb0(%elem: !parsed_csl.unresolved):
      // loop body using %elem
    }
    ```

    Example with element and index:
    ```mlir
    parsed_csl.for(%array) {elem_name = "x", index_name = "i"} {
    ^bb0(%elem: !parsed_csl.unresolved, %idx: !parsed_csl.unresolved):
      // loop body using %elem and %idx
    }
    ```

    Example with label:
    ```mlir
    parsed_csl.for(%range) {elem_name = "x", label = "outer"} {
    ^bb0(%elem: !parsed_csl.unresolved):
      // can use break :outer; to exit this loop
    }
    ```
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$iterable,
    StrAttr:$elem_name,
    OptionalAttr<StrAttr>:$index_name,
    OptionalAttr<StrAttr>:$label,
    DefaultValuedAttr<BoolAttr, "false">:$is_inline,
    OptionalAttr<TypeAttr>:$inferred_elem_type,
    OptionalAttr<TypeAttr>:$inferred_index_type
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
    `(` $iterable `)` attr-dict-with-keyword $body
  }];

  let hasVerifier = 1;
}

def ParsedCSL_WhileOp : ParsedCSL_Op<"while", [
  NoTerminator,
  ParsedCSL_Statement,
  CSL_LabeledOpInterface,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  AutomaticAllocationScope]> {
  let summary = "While loop with condition region";
  let description = [{
    Standard while loop. The condition region is evaluated before each iteration
    and must yield a boolean value via `parsed_csl.yield`. This enables proper
    re-evaluation of the condition at comptime.
    Optional continuation region is evaluated at the end of each iteration
    (including iterations ended by continue).

    Example without continuation:
    ```mlir
    parsed_csl.while cond {
      %cond = ...
      parsed_csl.yield %cond
    } do {
      // loop body
    }
    ```

    Example with continuation:
    ```mlir
    parsed_csl.while cond {
      %cond = ...
      parsed_csl.yield %cond
    } do {
      // loop body
    } continuation {
      // executed at end of each iteration
    }
    ```

    Example with label:
    ```mlir
    parsed_csl.while {label = "outer"} cond {
      %cond = ...
      parsed_csl.yield %cond
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
}

def ParsedCSL_BreakOp : ParsedCSL_Op<"break", [
  ParsedCSL_Statement,
  ParsedCSL_PseudoTerminator]> {
  let summary = "Break out of a loop or labeled block, optionally with a value";
  let description = [{
    Exits the enclosing loop or labeled block. If a target label is specified,
    breaks out of the construct with that label. An optional value can be
    provided when breaking out of a labeled block (not a loop).

    Example:
    ```mlir
    parsed_csl.break                    // break innermost loop
    parsed_csl.break "outer"            // break loop/block labeled "outer"
    parsed_csl.break "blk" %value       // break labeled block with value
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$target_label,
    Optional<ParsedCSL_UnresolvedType>:$value
  );

  let assemblyFormat = [{ ($target_label^)? ($value^)? attr-dict }];
  let hasVerifier = 1;
}

def ParsedCSL_ContinueOp : ParsedCSL_Op<"continue", [
  ParsedCSL_Statement,
  ParsedCSL_PseudoTerminator]> {
  let summary = "Continue to next iteration of a loop";
  let description = [{
    Continues to the next iteration of the enclosing loop. If a target label
    is specified, continues the loop with that label (for nested loops).

    Example:
    ```mlir
    parsed_csl.continue          // continue innermost loop
    parsed_csl.continue "outer"  // continue loop labeled "outer"
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$target_label);

  let assemblyFormat = [{ ($target_label^)? attr-dict }];
}

//===----------------------------------------------------------------------===//
// Labeled Block Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_LabeledBlockOp : ParsedCSL_Op<"labeled_block", [
  NoTerminator,
  ParsedCSL_Expression,
  InferredTypeInterface,
  CSL_LabeledOpInterface,
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  AutomaticAllocationScope]> {
  let summary = "Labeled block that can be exited with break, optionally yielding a value";
  let description = [{
    A labeled block establishes a scope that can be exited early using
    `break :label [value]`. Unlike loops, labeled blocks do not support `continue`.

    The block yields a value determined by the break statements:
    - `break :label value` yields the specified value
    - `break :label` yields void
    - Fall-through (no break) yields void

    All breaks targeting this block must yield type-compatible values.

    Example (void):
    ```mlir
    parsed_csl.labeled_block "outer" {
      parsed_csl.if %cond {
        parsed_csl.break "outer"
      }
    }
    ```

    Example (valued):
    ```mlir
    %result = parsed_csl.labeled_block "blk" {
      parsed_csl.if %cond {
        parsed_csl.break "blk" %val1
      }
      parsed_csl.break "blk" %val2
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$label,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let regions = (region SizedRegion<1>:$body);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{ $label $body attr-dict }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Module Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_ModuleOp : ParsedCSL_Op<"module", [
  IsolatedFromAbove,
  NoTerminator,
  SingleBlock,
  SymbolTable
]> {
  let summary = "Imported module instance";
  let description = [{
    Represents an instance of an imported module created by @import_module.
    Each call to @import_module creates a new module instance. The module is
    isolated from above and contains the parsed contents of the imported file
    with param bindings applied.

    The module body contains all declarations from the imported file (variables,
    functions, etc.). Param bindings are prepopulated as constants at the start
    of the module body.

    The static result type is !parsed_csl.unresolved. During comptime evaluation,
    the inferred type is !csl.imported_module.

    Example:
    ```mlir
    %mod = parsed_csl.module source_file="multiply.csl" instance_id=0 {
      %factor = parsed_csl.constant #csl.comptime_int<42>
      parsed_csl.func @multiply(%x: !parsed_csl.unresolved) {
        %result = parsed_csl.binary mul, %x, %factor
        parsed_csl.return %result
      }
    } : !parsed_csl.unresolved
    ```
  }];

  let arguments = (ins
    StrAttr:$source_file,
    I64Attr:$module_identifier,
    OptionalAttr<StrAttr>:$module_name
  );

  let regions = (region SizedRegion<1>:$region);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{
    `source_file` `=` $source_file
    `module_identifier` `=` $module_identifier
    (`name` `=` $module_name^)?
    $region attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    FailureOr<GlobalOp> getMemberVariable(StringRef name);
    /// Get a member function by name. Returns a variant holding either
    /// FuncOp (for instantiated functions) or GenericFuncOp (for generic
    /// functions). Returns failure if no function with that name exists.
    FailureOr<std::variant<FuncOp, GenericFuncOp>> getMemberFunc(StringRef name);
  }];
}

//===----------------------------------------------------------------------===//
// Comptime Operations
//===----------------------------------------------------------------------===//

def ParsedCSL_ComptimeExprOp : ParsedCSL_Op<"comptime_expr", [
  InferredTypeInterface,
  ParsedCSL_ComptimeScope, Pure, ParsedCSL_Expression, SingleBlock, NoRegionArguments]> {
  let summary = "Comptime expression wrapper";
  let description = [{
    Wraps an expression to indicate it should be evaluated at compile-time.

    Example:
    ```mlir
    %c = parsed_csl.comptime_expr {
      %a = parsed_csl.binary add, %x, %y
      %b = parsed_csl.binary mul, %a, %z
      parsed_csl.yield %b
    }
    ```
  }];

  let arguments = (ins OptionalAttr<TypeAttr>:$inferred_type);
  let regions = (region SizedRegion<1>:$exprRegion);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = "$exprRegion attr-dict";

  // FIXME: implement folder
  let hasFolder = 0;
}

def ParsedCSL_ComptimeBlockOp : ParsedCSL_Op<"comptime", [
  ParsedCSL_ComptimeScope, 
  ParsedCSL_Statement, 
  RecursiveMemoryEffects, 
  SingleBlock, 
  NoRegionArguments, 
  NoTerminator
  ]> {
  let summary = "Comptime block wrapper";
  let description = [{
    Wraps a block of code to indicate it should be executed at compile-time.

    Example:
    ```mlir
    parsed_csl.comptime {
      ...
    }
    ```
  }];

  let arguments = (ins);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";
}

def ParsedCSL_LayoutBlockOp : ParsedCSL_Op<"layout", [
  ParsedCSL_ComptimeScope,
  ParsedCSL_Statement,
  RecursiveMemoryEffects,
  SingleBlock,
  NoRegionArguments,
  NoTerminator
  ]> {
  let summary = "Layout block wrapper";
  let description = [{
    Wraps a block of code that configures the tile layout.
    Semantically identical to a comptime block, but emitted as `layout { }`.

    Example:
    ```mlir
    parsed_csl.layout {
      ...
    }
    ```
  }];

  let arguments = (ins);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";
}

def ParsedCSL_YieldOp : ParsedCSL_Op<"yield", [
  ParsedCSL_Statement,
  ParsedCSL_PseudoTerminator,
  Terminator,
  // Can terminate comptime expression, type region of GenericFuncOp,
  // else region of SelectOp, rhs region of ShortCircuitOp, cond region of WhileOp,
  // or case/else regions of SwitchOp
  // ParentOneOf<["ComptimeExprOp", "GenericFuncOp", "GlobalOp", "SelectOp", "ShortCircuitOp", "WhileOp", "SwitchOp"]>
  ]> {
  let summary = "Yield value from expression region";
  let description = [{
    Indicates that control flows back from this region to the parent region. 
    If a value is provided, it becomes the result of the parent operation.

    Example:
    ```mlir
    parsed_csl.yield %value
    ```
  }];

  let arguments = (ins Optional<ParsedCSL_UnresolvedType>:$value);

  let assemblyFormat = "($value^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Tensor Access Operations (for DSD creation)
//===----------------------------------------------------------------------===//

def ParsedCSL_TensorAccessOp : ParsedCSL_Op<"tensor_access", [
    InferredTypeInterface,
    RecursivelySpeculatable,
    RecursiveMemoryEffects,
    ParsedCSL_Expression
]> {
  let summary = "Tensor access expression for DSD creation";
  let description = [{
    Represents a tensor access expression like `|i,j|{10,20} -> array[i+1, j*2]`.

    The base array/pointer and extents are operands. The body region has block
    arguments for the induction variables and computes index expressions that
    are yielded by the terminator.

    Example CSL syntax: |i,j|{10,20} -> array[i+1, j*2]

    Example MLIR:
    ```mlir
    %array = parsed_csl.get_symbol @array
    %e0 = parsed_csl.constant #csl.comptime_int<10>
    %e1 = parsed_csl.constant #csl.comptime_int<20>
    %result = parsed_csl.tensor_access base(%array) extents(%e0, %e1) ["i", "j"] {
    ^bb0(%i: !parsed_csl.unresolved, %j: !parsed_csl.unresolved):
      %one = parsed_csl.constant #csl.comptime_int<1>
      %two = parsed_csl.constant #csl.comptime_int<2>
      %idx0 = parsed_csl.binary add, %i, %one
      %idx1 = parsed_csl.binary mul, %j, %two
      parsed_csl.tensor_access_yield indices(%idx0, %idx1)
    }
    ```

    Produces an anonymous struct with base_address, offset, stride, extent fields
    that can be passed to @get_dsd to create a DSD.
  }];

  let arguments = (ins
    ParsedCSL_UnresolvedType:$base,
    Variadic<ParsedCSL_UnresolvedType>:$extents,
    StrArrayAttr:$induction_var_names,
    OptionalAttr<TypeAttr>:$inferred_type
  );

  let regions = (region SizedRegion<1>:$body);
  let results = (outs ParsedCSL_UnresolvedType:$result);

  let assemblyFormat = [{
    `base` `(` $base `)` `extents` `(` $extents `)` $induction_var_names $body attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the block arguments representing induction variables.
    Block::BlockArgListType getInductionVars() {
      return getBody().front().getArguments();
    }

    /// Returns the number of dimensions (induction variables).
    unsigned getNumDimensions() {
      return getInductionVarNames().size();
    }
  }];
}

def ParsedCSL_TensorAccessYieldOp : ParsedCSL_Op<"tensor_access_yield", [
    ParsedCSL_PseudoTerminator,
    Terminator,
    HasParent<"TensorAccessOp">
]> {
  let summary = "Yield computed index expressions from tensor_access region";
  let description = [{
    Terminates a tensor_access region, yielding the computed index expressions
    for each dimension. The number of indices must match the number of
    induction variables in the parent TensorAccessOp.

    Example:
    ```mlir
    parsed_csl.tensor_access_yield indices(%idx0, %idx1)
    ```
  }];

  let arguments = (ins
    Variadic<ParsedCSL_UnresolvedType>:$indices
  );

  let assemblyFormat = [{
    `indices` `(` $indices `)` attr-dict
  }];

  let hasVerifier = 1;
}

def ParsedCSL_BuiltinCallOp : ParsedCSL_Op<"builtin_call", [
    InferredTypeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "builtin call";
  let description = [{
    Represents a call to a builtin function. May return void or a value.
    The builtin is identified by name (string) rather than enum, so that
    calls to non-existent builtins are allowed as long as they are never
    reached during comptime evaluation. Resolution to the actual builtin
    enum happens during semantic analysis (ComptimeEvalPass).

    Memory effects are determined by whether the builtin is pure (no side effects).

    Some builtins like @export_symbol take symbol references (functions, globals)
    directly without loading them. The comptime analysis tracks which arguments
    are lvalues vs rvalues.
  }];
  let arguments = (ins
    StrAttr:$builtin_name,
    Variadic<ParsedCSL_UnresolvedType>:$args,
    OptionalAttr<TypeAttr>:$inferred_type
  );
  let results = (outs ParsedCSL_UnresolvedType:$result);
  let assemblyFormat = "$builtin_name `(` $args `)` attr-dict";

  let extraClassDeclaration = [{
    /// Try to resolve the builtin name to a BuiltinKind enum value.
    /// Returns std::nullopt if the name is not a known builtin.
    std::optional<csl::BuiltinKind> resolveBuiltin() {
      return csl::symbolizeBuiltinKind(getBuiltinName());
    }
  }];
}

#endif // PARSED_CSL_OPS
