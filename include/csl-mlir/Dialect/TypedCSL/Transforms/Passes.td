//===- Passes.td - TypedCSL Transformation Passes ----------*- tablegen -*-===//
//
// Definitions of transformation passes for the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_CSL_TRANSFORMS_PASSES
#define TYPED_CSL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def RollLayoutLoopsPass : Pass<"roll-layout-loops", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Compact layout builtin calls into for loops";
  let description = [{
    Compacts repeated layout builtin calls (@set_tile_code, @set_color_config)
    into for loops where possible. This is the inverse of loop unrolling that
    happens during comptime evaluation.

    For each group of calls with identical invariant parameters:
    - @set_tile_code: groups by (filename, params)
    - @set_color_config: groups by (color, config)

    The pass finds maximal axis-aligned rectangles covering each group's
    coordinates and emits:
    - Single call for 1x1 rectangle
    - Single for loop for Nx1 or 1xM strips
    - Nested for loops for NxM rectangles

    This significantly reduces output size for large PE grids while preserving
    semantics.
  }];

  let dependentDialects = ["::mlir::csl::CSLDialect"];
}

def ValidateTypedCSLPass : Pass<"validate-typed-csl", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Validate TypedCSL IR after canonicalization";
  let description = [{
    This pass performs validation checks on TypedCSL IR after the canonicalizer
    has run. It catches semantic errors that may remain after TypedCSL
    canonicalization patterns have fired.

    **Current checks:**

    1. **Comptime-only type validation**: Values with comptime-only types
       (comptime_int, comptime_float, type, etc.) must be defined by
       constant-like operations. Runtime operations cannot produce comptime-only
       values.

       This check runs in TypedCSL (rather than ParsedCSL) because some
       TypedCSL canonicalization patterns eliminate comptime-known values that
       were not yet folded during ParsedCSL processing.

    This pass must run directly after the first canonicalizer in TypedCSL.
  }];
}

def CountTasksPass : Pass<"count-tasks", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Statically count the task slots required by the program";
  let description = [{
    Walks the unit and reports the number of WSE task slots required, split
    by kind (local / control / data). The result is emitted to stderr as a
    plain text summary.

    Counting model:
    - Each `@bind_local_task` / `@bind_control_task` / `@bind_data_task`
      inside a `typed_csl.comptime` block contributes one slot of the
      corresponding kind (the slot is permanently bound).
    - For each comptime-bound async-function callee, its `footprint` is
      added to the local task count (the worker pool uses local task IDs):

        footprint(f) = max over await sites in f of:
          @await(g())                       -> footprint(g)
          @await(g(), .offload=true)        -> 1 + footprint(g)
          @await_parallel(g1, ..., gn)      -> footprint(g1) + sum_{i>=2}(1 + footprint(gi))

      The first parallel callee runs on the caller's task slot, so it does
      not claim an additional slot — only its own descendants do.

    Limitations / warnings:
    - Recursive async call graphs are not supported (footprint is ill-defined).
    - `@bind_*_task` outside a `typed_csl.comptime` block (runtime binds)
      cannot be counted; the report flags this and may undercount.
    - `await` / `await_parallel` operands must be direct `typed_csl.call`
      results, matching the existing coroutine-lowering invariant.
  }];
}

def CheckLinknameCollisionsPass : Pass<"check-linkname-collisions", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Check for symbol name collisions based on effective linkname";
  let description = [{
    Verifies that no two symbols (functions or globals) share the same effective
    name. For symbols with a linkname attribute, the linkname is the effective
    name. For symbols without a linkname, the symbol name is the effective name.

    This pass should run after module inlining and conversion to TypedCSL,
    where all symbols are at the top level.
  }];
}

def TailRecursionToLoopPass : Pass<"tail-recursion-to-loop", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Turn tail calls of a function to itself into a loop";
  let description = [{
    Rewrites every function that calls itself in tail position
    (`return f(...)`, or `f(...); return;` in a void function) into a loop:
    the body becomes a `while (true)`, each parameter a mutable variable,
    and each tail call assigns the new arguments to the parameter variables
    and continues the loop.  This removes the recursion from the call graph,
    so stack sizing needs no annotation for it.  CSL has no labeled
    `continue`, so a tail call inside another loop of the function stays a
    call.

    Async functions and tasks are left alone, as are functions that take an
    address with `&`: every iteration of the loop reuses the storage of the
    parameters and locals, while every recursive call has its own.

    `max_recursion_depth(N)` on a function that is no longer recursive
    afterwards is reported and removed.
  }];
}

//===----------------------------------------------------------------------===//
// Coroutine Pipeline
//===----------------------------------------------------------------------===//
//
// The coroutine transform is a 4-pass pipeline:
//
//   1. CoroPrepareCFPass: Lower calling convention (futures, await loops)
//   2. LowerSuspendingSCFToCFPass: Flatten suspending structured CF to CFG
//   3. CoroFrameRewritePass: Analyze frame layout and rewrite spill/reload
//   4. TrampolineEmitPass: Build structured while+switch state machine
//

def CoroPrepareCFPass : Pass<"coro-prepare-cf", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Prepare control flow for coroutine transform";
  let description = [{
    Lowers the async calling convention and control flow in preparation for
    the coroutine transform. Processes async functions in callee-first
    topological order.

    Transformations:
    - ForOp → WhileOp for loops containing suspend/await
    - CallOp returning !csl.future → allocate callee frame + init + eager call
    - AwaitOp → while(!done) { call; if(!done) suspend; } retry loop
    - Type conversion: !csl.future<@fn> → *!csl.coroframe<@fn>
    - Type replacement: !csl.coroframe<@fn> → [N]u8 (once callee sizes known)

    Stores the slot alignment as the "coro.slot_align" attribute on the
    CompilationUnitOp for downstream passes to read.
  }];

  let dependentDialects = ["::mlir::cf::ControlFlowDialect"];

  let options = [
    Option<"slotAlignment", "slot-alignment", "unsigned", "2",
           "Minimum alignment in bytes for coroutine frame slots "
           "(1 = byte [*]u8, 2 = word [*]u16)">
  ];
}

def LowerSuspendingSCFToCFPass : Pass<"lower-suspending-scf-to-cf", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Selectively lower structured CF containing suspend points to CFG";
  let description = [{
    Lowers TypedCSL structured control flow operations (if, while, switch,
    labeled block) to CFG form (cf.br, cf.cond_br) ONLY when they transitively
    contain suspend or await points. Non-suspending structured ops survive
    intact inside basic blocks.

    Also converts SuspendOp → SuspendBrOp and AwaitOp → AwaitBrOp (CFG
    terminators that split the block at each suspend/await point).

    Breaking terminators (return, break, continue) that target a decomposed
    structured op are rewritten to cf.br to the appropriate merge block.
  }];

  let dependentDialects = ["::mlir::cf::ControlFlowDialect"];
}

def InlineSmallNonResumeBlocksPass : Pass<"inline-small-non-resume-blocks", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Inline small non-resume CFG blocks into their predecessors";
  let description = [{
    Multi-predecessor blocks that are NOT resume targets (successors of
    suspend_br or await_br) survive canonicalization as separate basic blocks
    (e.g. loop headers with back-edges). Each becomes a separate trampoline
    state, adding switch dispatch overhead on every iteration.

    This pass identifies non-resume blocks with fewer than a threshold number
    of ops and duplicates their contents into each predecessor. This eliminates
    unnecessary trampoline state transitions for small control flow blocks like
    loop condition checks.

    MUST run directly after CoroFrameRewritePass: at that point all cross-block
    SSA values have been spilled to the coroutine frame, so every block loads
    its own values independently. This guarantees that inlined blocks have no
    escaping SSA definitions, making block duplication and dead-block erasure
    safe. Running before CoroFrameRewritePass would hit SSA dominance
    violations for blocks whose op results are used by successor blocks.

    MUST run before TrampolineEmitPass, which maps blocks 1:1 to switch states.
  }];

  let dependentDialects = ["::mlir::cf::ControlFlowDialect"];
}

def CoroFrameRewritePass : Pass<"coro-frame-rewrite", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Analyze coroutine frames and rewrite suspend-crossing values";
  let description = [{
    Runs the coroutine frame analysis (CoroFrameBuilder) on each async function
    to determine which values cross suspend points and need frame storage.
    Performs bin-packing layout with liveness-based overlap.

    Rewrites:
    - VariableOps crossing suspend points → frame slot accesses
    - SSA values crossing suspend points → store-after-def + reload-before-use
    - Async function parameters → stored in frame at entry

    After this pass, no SSA value crosses a suspend point — all persist
    through the coroutine frame.
  }];
}

def TrampolineEmitPass : Pass<"trampoline-emit", "::mlir::csl::CompilationUnitOp"> {
  let dependentDialects = ["::mlir::cf::ControlFlowDialect"];
  let summary = "Emit structured trampoline state machines from CFG-form coroutines";
  let description = [{
    Transforms CFG-form async functions into structured while+switch trampoline
    state machines. Each basic block becomes a switch case, with one state per
    block.

    Control flow mapping:
    - suspend_br → store state + return false (not done)
    - await_br → call callee, check done, conditional suspend
    - cf.br/cf.cond_br → store state + yield (re-enter switch)
    - return → store result + store done state + return true

    For tasks: creates a global frame variable, signature becomes () → void.
    For async fns: signature becomes ([*]u8) → i1 (frame pointer, done flag).

    Non-suspending structured ops (if, while without suspends) inside basic
    blocks are preserved in the switch cases.
  }];
}

//===----------------------------------------------------------------------===//
// Fiber-based Cooperative Multitasking
//===----------------------------------------------------------------------===//

def FiberLoweringPass : Pass<"fiber-lowering", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Lower async functions using fiber-based stack switching";
  let description = [{
    Lowers async functions using fiber-based cooperative multitasking with
    O(1) context switching. At suspend points, all registers and SP are
    saved atomically via inline assembly; at resume, they are restored
    and execution jumps directly to the resume point.

    Transformations:
    - Look up runtime worker pool globals (struct of arrays) + the
      `worker_entry` task from the imported fiber runtime CSL module.
    - Generate per-root-task fiber stacks and reset functions.
    - Rewrite the user's `@bind_*_task(async_fn, T)` to bind `worker_entry`
      to T at comptime; the per-root reset wires up `pool[T].sp/pc` and
      the completion-handler r15 slot for the first invocation.
    - Fuse async call + @await into a direct call (callee runs on caller).
    - Replace `@await_parallel` with dynamic claim/bind/activate per child
      and a runtime `suspend()` call that saves the caller's SP/PC.
    - Replace SuspendOp with a runtime `suspend()` call.
  }];

  let dependentDialects = ["::mlir::csl::CSLDialect"];

  let options = [
    Option<"targetArch", "arch", "mlir::csl::TargetArch",
           /*default=*/"mlir::csl::TargetArch::WSE2",
           "Target architecture",
           [{mlir::csl::enumValuesForTargetArch()}]>,
    Option<"outlineAwaitParallel", "outline-await-parallel", "bool",
           /*default=*/"false",
           "Outline each @await_parallel site into an ordinary async fn so a "
           "backend-inline-aware profiler can attribute its cost separately; "
           "the function is neither frontend-inline nor noinline">
  ];
}

//===----------------------------------------------------------------------===//
// Lower Custom Builtins
//===----------------------------------------------------------------------===//

def LowerCustomBuiltinsPass : Pass<"lower-custom-builtins", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Lower custom CSL builtins to hardware primitives";
  let description = [{
    Lowers custom builtins that require hardware-specific lowering:

    - @current_task() → call runtime get_current_task_id + @get_local_task_id
    - @bind_*_task (runtime) → call runtime bind_task(fn_ptr, task_id)
    - @claim_*() → call runtime claim function + wrap in typed ID
    - @release_*() → extract u16 via @get_int + call runtime release function

    Also patches resource module globals with comptime-claimed bitmap values.
  }];

  let dependentDialects = ["::mlir::csl::CSLDialect"];

  let options = [
    Option<"targetArch", "arch", "mlir::csl::TargetArch",
           /*default=*/"mlir::csl::TargetArch::WSE2",
           "Target architecture",
           [{mlir::csl::enumValuesForTargetArch()}]>
  ];
}

//===----------------------------------------------------------------------===//
// Strip Runtime Exports
//===----------------------------------------------------------------------===//

def StripRuntimeExportsPass : Pass<"strip-runtime-exports", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Strip export attributes from runtime module symbols";
  let description = [{
    Removes the export attribute and linkname from all functions and globals
    whose names start with a runtime module prefix (__fiber_runtime__,
    __alloc_runtime__). These symbols are marked export in the .csl source
    to survive DCE until the compiler passes that reference them have run.
    This pass strips the exports so they don't leak into the final output.

    With `runtime-noinline`, runtime functions are additionally marked
    noinline instead of being left for cslc to inline. That exists purely so
    a profiler can attribute cycles to them: by default every runtime
    function except `worker_entry` is inlined into its callers, so its cost
    is charged to user code. An instrumented build is not the shipped one --
    the extra calls change code layout, frame sizes and timing.
  }];

  let options = [
    Option<"runtimeNoinline", "runtime-noinline", "bool", /*default=*/"false",
           "Mark runtime functions noinline so a profiler can attribute "
           "cycles to them (changes layout, frame sizes and timing)">
  ];

  let dependentDialects = ["::mlir::csl::CSLDialect"];
}

#endif // TYPED_CSL_TRANSFORMS_PASSES
