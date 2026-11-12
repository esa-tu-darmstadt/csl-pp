//===- TypedCSLBase.td - TypedCSL Dialect Definition ------*- tablegen -*-===//
//
// TypedCSL Dialect - Fully typed CSL IR ready for lowering.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_CSL_BASE
#define TYPED_CSL_BASE

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// TypedCSL Dialect
//===----------------------------------------------------------------------===//

def TypedCSL_Dialect : Dialect {
  let name = "typed_csl";
  let summary = "Typed CSL dialect with fully resolved types";
  let description = [{
    The TypedCSL dialect represents CSL code after type resolution and comptime
    evaluation. All types are concrete runtime types, all comptime values have
    been evaluated and inlined, and functions are isolated from above.

    This dialect is the final IR representation before lowering to target code.

    Key properties:
    - All types are concrete (i32, f32, ptr<i32>, etc.) - no unresolved types
    - No comptime-only types (no comptime_int, type, etc.)
    - No comptime variables or comptime function calls
    - Functions are isolated from above (globals are referenced through
      constant #csl.global_ref values)
    - Explicit casts inserted for all type conversions
    - Global const variables have been inlined
  }];

  let cppNamespace = "::mlir::typed_csl";

  // Use generated type printer/parser
  let useDefaultTypePrinterParser = 1;

  // Use default attribute printer/parser
  let useDefaultAttributePrinterParser = 1;

  // Has constant materializer
  let hasConstantMaterializer = 1;

  let dependentDialects = ["mlir::csl::CSLDialect", "mlir::cf::ControlFlowDialect"];

  // Verifies the argument attributes of the dialect (typed_csl.noalias).
  let hasRegionArgAttrVerify = 1;

  let extraClassDeclaration = [{
    // Register all types
    void registerTypes();

    // Register all attributes
    void registerAttributes();

    /// Unit attribute of a pointer parameter of a typed_csl.func whose
    /// pointee the function accesses only through that parameter, like
    /// `llvm.noalias` (C's `restrict`): no other variable, parameter or global
    /// of the function overlaps it. The emitter then does not treat a store
    /// through the parameter as a possible write to other storage. The
    /// stackless coroutine lowering sets it on the frame parameter of the
    /// functions it generates.
    static StringRef getNoAliasAttrName() { return "typed_csl.noalias"; }
  }];
}

#endif // TYPED_CSL_BASE
