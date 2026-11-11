//===- Passes.td - ParsedCSL Transformation Passes --------*- tablegen -*-===//
//
// Definitions of transformation passes for the ParsedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_TRANSFORMS_PASSES
#define PARSED_CSL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ComptimeEvalPass : Pass<"parsed-csl-comptime-eval", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Evaluate and fold compile-time operations";
  let description = [{
    This pass evaluates compile-time operations in the ParsedCSL dialect
    and folds them to constants where possible. It:

    1. Runs CompileAnalysis to evaluate globals and function bodies
    2. Replaces comptime-known SSA values with constants
    3. Eliminates dead code after constant folding

    The pass uses a layered analysis approach:
    - **GlobalEvalContext**: Evaluates top-level constants
    - **FunctionEvalContext**: Tracks function-local comptime evaluation with
      scope management for runtime-conditional regions
    - **ComptimeFunctionEvaluator**: Handles comptime function calls with
      recursion detection

    **Runtime Conditional Scopes (Write Barriers):**
    Operations inside runtime-conditional control flow (e.g., if statements
    with non-comptime conditions) cannot modify comptime variables from outer
    scopes. This prevents undefined behavior when the scope may not execute.

    **Comptime-Required Contexts:**
    Inside comptime blocks or when evaluating comptime function calls, ALL
    operations must be comptime evaluable. Failure to evaluate is a compile
    error.

    Example:

    ```mlir
    // Before
    parsed_csl.compilation_unit {
      %c10 = parsed_csl.constant 10 : i32
      %c20 = parsed_csl.constant 20 : i32
      %sum = parsed_csl.binary add, %c10, %c20 : (i32, i32) -> i32
      parsed_csl.var "x" = %sum : i32
    }

    // After
    parsed_csl.compilation_unit {
      %c30 = parsed_csl.constant 30 : i32
      parsed_csl.var "x" = %c30 : i32
    }
    ```
  }];

  let constructor = "mlir::parsed_csl::createComptimeEvalPass()";

  let options = [
    Option<"cslFlags", "csl-flags", "::mlir::csl::CslFlags",
           /*default=*/"::mlir::csl::CslFlags{}",
           "CSL compilation flags">
  ];

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def FinalizeInstantiationPass : Pass<"parsed-csl-finalize-instantiation", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Erase GenericFuncOps and rename instantiated FuncOps";
  let description = [{
    After comptime evaluation, all function declarations exist as GenericFuncOps
    and all instantiated functions have suffixed names (e.g., `foo_inst0`).
    This pass:

    1. Erases all GenericFuncOps (they have served their purpose)
    2. Renames instantiated FuncOps back to their original names where
       unambiguous (single instantiation per scope)
    3. Updates all FuncRefAttr and FuncAddrAttr references to match

    For functions with multiple instantiations in the same scope (e.g.,
    `double_val(i32, x)` and `double_val(f32, y)`), the suffixed names
    are kept to avoid ambiguity.

    This pass must run directly after ComptimeEvalPass.
  }];

  let constructor = "mlir::parsed_csl::createFinalizeInstantiationPass()";

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def ResolveReadsPass : Pass<"parsed-csl-resolve-reads", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Turn the reads of locations into loads";
  let description = [{
    The parser emits a `parsed_csl.read` wherever the source uses an
    expression as a value, at the position where the source reads it. Once
    comptime evaluation has inferred the types, this pass resolves each read:
    a read of a location becomes a `parsed_csl.load` at the same position,
    and a read of anything else is replaced by the value it reads. A pointer
    variable that is the base of an index access is loaded at the access,
    since the pointer and not the variable is indexed.

    Example:
    ```mlir
    // Before (inferred_type on %x is !csl.lvalue<mutable si32>, on %f a
    // function type)
    %rx = parsed_csl.read %x
    %rf = parsed_csl.read %f
    parsed_csl.call %rf(%rx)

    // After
    %lx = parsed_csl.load from %x
    parsed_csl.call %f(%lx)
    ```

    Afterwards, every location that is used as a value is loaded explicitly;
    the pass reports any that is not as an internal error. It must run after
    ComptimeEvalPass, which sets the inferred types.
  }];

  let constructor = "mlir::parsed_csl::createResolveReadsPass()";

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def NormalizeCastsPass : Pass<"normalize-casts", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Validate and normalize all cast operations";
  let description = [{
    This pass validates and normalizes cast operations in two phases:

    **Phase 1: Validate Existing Casts**
    - Walks all existing parsed_csl.cast operations
    - For casts with comptime-known sources:
      - Evaluates the cast using CSL folder helpers
      - On success: replaces the cast with a constant (fold)
      - On failure: emits diagnostic and fails the pass
    - Ensures all explicit casts are valid before lowering

    **Phase 2: Insert Implicit Casts**
    - Inserts explicit cast operations where type coercion is needed
    - Processes binary operations, variables, assignments, calls, and returns
    - For comptime-known values being cast:
      - Evaluates the cast and creates a constant (fold)
      - Fails the pass if cast is invalid (e.g., overflow)
    - For runtime values: materializes cast operations

    The pass inserts casts for:
    - Binary operation operands to match the common type
    - Variable initializers to match the variable type
    - Assignment values to match the lvalue type
    - Call arguments to match parameter types
    - Return values to match the function return type

    This makes all type conversions explicit and validates them before
    lowering to TypedCSL, catching overflow and invalid casts early.

    Example:
    ```mlir
    // Before
    parsed_csl.func "add"(%arg0 : %i32) -> (%i32) {
      %c5 = parsed_csl.constant #csl.comptime_int<5>
      %result = parsed_csl.binary add, %arg0, %c5
      parsed_csl.return %result
    }

    // After (Phase 2 inserts cast, which gets folded to constant)
    %c5_i32 = parsed_csl.constant #csl.int<32, signed><5>
    %result = parsed_csl.binary add, %arg0, %c5_i32
    ```
  }];

  let constructor = "mlir::parsed_csl::createNormalizeCastsPass()";

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def LowerTensorAccessPass : Pass<"parsed-csl-lower-tensor-access", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Lower runtime TensorAccessOps to ConstructStructOps";
  let description = [{
    Converts TensorAccessOps with runtime (non-comptime) bases to explicit
    struct construction operations. Comptime TAEs are already evaluated
    by ComptimeEvalPass - this handles the remaining runtime cases.

    For each runtime TensorAccessOp, the pass:
    1. Analyzes the body region to extract affine index expressions
    2. Computes CSL-style strides using the reset-accumulate formula
    3. Creates a ConstructStructOp with fields: base_address, offset, stride, extent

    The strides are pre-computed at compile time (since index expressions must
    be affine in the induction variables), while the base address is taken
    from the runtime base operand.

    This pass must run after ComptimeEvalPass and before NormalizeCastsPass.
    Running before NormalizeCasts is required because for runtime TAEs, the
    body region is NOT evaluated during comptime eval, meaning operations
    inside have no inferred_type attributes - NormalizeCasts would crash.

    Example:
    ```mlir
    // Before
    %result = parsed_csl.tensor_access base(%arr) extents(%e) ["i"] {
    ^bb0(%i: !parsed_csl.unresolved):
      %c2 = parsed_csl.constant #csl.comptime_int<2>
      %idx = parsed_csl.binary mul, %i, %c2
      parsed_csl.tensor_access_yield indices(%idx)
    }

    // After
    %addr = parsed_csl.address_of %arr
    %offset = parsed_csl.constant #csl.int<16, signed><0>
    %stride = parsed_csl.constant #csl.int<8, signed><2>
    %result = parsed_csl.construct_struct ["base_address", "offset", "stride", "extent"]
                                          %addr, %offset, %stride, %e
    ```
  }];

  let constructor = "mlir::parsed_csl::createLowerTensorAccessPass()";

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def InlineModulesPass : Pass<"inline-modules", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Inline module contents to top level";
  let description = [{
    Flattens all parsed_csl.module operations by moving their variables
    and functions to the top-level CompilationUnitOp with prefixed names.

    After this pass:
    - All modules are removed
    - Global variables from modules are moved to top level with prefixed names
    - Functions from modules are moved to top level with prefixed names
    - All field_access operations on modules are replaced with direct SSA references
    - Nested modules are handled recursively (bottom-up processing)

    When auto-linkname is enabled (default), private (non-export, non-extern)
    functions and globals that don't already have a user-specified linkname
    get a dot-separated linkname attribute matching the official CSL compiler's
    naming convention (e.g., "mod.func").

    Example:
    ```mlir
    // Before
    csl.compilation_unit {
      %mod = parsed_csl.module name = "myModule" {
        %var = parsed_csl.var "x" ...
      }
      %ref = parsed_csl.field_access %mod["x"]
    }

    // After
    csl.compilation_unit {
      %var = parsed_csl.var "myModule$x" ...
      // field_access replaced with direct reference to %var
    }
    ```
  }];

  let options = [
    Option<"cslFlags", "csl-flags", "::mlir::csl::CslFlags",
           /*default=*/"::mlir::csl::CslFlags{}",
           "CSL compilation flags">
  ];

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def InjectCoroRuntimePass : Pass<"inject-coro-runtime", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Import the coroutine runtime into every PE program";
  let description = [{
    Adds the `__fiber_runtime` and `__alloc_runtime` module imports.

    The runtime is compiled into the binary and served from memory as the
    system modules `<coro/fiber_runtime>` and `<coro/resources>`, so this pass
    only has to emit the two `const ... = @import_module(...)` declarations a
    user would otherwise have written by hand -- including an absolute path
    into a checkout of this repository. Everything downstream (comptime
    specialization, module inlining and symbol prefixing) then treats them
    like any other import.

    Both are imported unconditionally, and neither is selected by
    `--coro-mode`:

    - Need cannot be decided here. This pass must precede comptime
      evaluation, since that is what resolves imports and specializes the
      runtime -- but comptime evaluation is also what pulls in imported
      modules, so the only thing visible at this point is the entry file.
      Constructs requiring the runtime routinely live in an imported library
      instead, so gating on the entry file's syntax misses them.
    - `wse{2,3}.csl` is not purely stackful-backend machinery. It also hosts
      `get_current_task_id` and `bind_task`, which LowerCustomBuiltinsPass
      calls for `@current_task()` and runtime `@bind_*_task` in both
      backends, and the stackless backend's generated worker uses them.
    - The fiber runtime calls `@release_local_task`, so it depends on the
      allocator; the two are never imported separately.

    An unused import costs compile time; a missing one is a hard error late
    in lowering. Unreferenced runtime symbols are removed by
    StripRuntimeExportsPass followed by SymbolDCE, so nothing unused reaches
    the output.
  }];

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

def ValidateParsedCSLPass : Pass<"validate-parsed-csl", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Validate ParsedCSL IR before lowering";
  let description = [{
    This pass performs validation checks on ParsedCSL IR before lowering to
    TypedCSL. It catches semantic errors that are difficult to detect during
    comptime evaluation or pattern-based normalization passes.

    **Current checks:**

    1. **Comptime variable check**: Verifies that no comptime VariableOps remain
       after comptime evaluation. They should have been evaluated and DCE'd by
       the canonicalizer after ComptimeEvalPass.

    This pass must run directly before lowering to TypedCSL.
  }];

  let constructor = "mlir::parsed_csl::createValidateParsedCSLPass()";

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect"];
}

#endif // PARSED_CSL_TRANSFORMS_PASSES
