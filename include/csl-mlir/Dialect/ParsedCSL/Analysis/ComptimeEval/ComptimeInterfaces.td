//===- ComptimeInterfaces.td - Comptime Evaluation Interfaces -*- tablegen -*-===//
//
// Interface definitions for compile-time evaluation in ParsedCSL.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_COMPTIME_INTERFACES
#define PARSED_CSL_COMPTIME_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ComptimeEvaluableOpInterface
//===----------------------------------------------------------------------===//

def ParsedCSL_ComptimeEvaluableOpInterface : OpInterface<"ComptimeEvaluableOpInterface"> {
  let description = [{
    Interface for operations that can be evaluated at compile-time.

    Operations implement this interface via external models to participate in
    compile-time evaluation. The interface provides two methods:

    1. evaluate() - Unified method that validates semantics, infers the result
       type, and attempts to evaluate the operation at compile-time. Returns a
       ComptimeEvalResult containing either:
       - ComptimeEvalSuccess: Successfully evaluated at compile-time (contains
         TypedAttr for rvalues, LValueDescriptor for lvalues, or control flow)
       - Type: Type inference succeeded but comptime evaluation failed (optional
         failure). The Type represents the inferred result type.
       - ComptimeError: Semantic error or comptime failure with diagnostics.

    2. requiresComptimeBySemantics() - Indicates whether the operation must be
       evaluated at compile-time (e.g., a global const variable, or having
       comptime-only return type). If true, evaluate() must succeed; otherwise,
       a diagnostic should be emitted.
  }];

  let cppNamespace = "::mlir::parsed_csl";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Unified evaluation: validate semantics, infer result type, and optionally
        evaluate at compile-time.

        This method performs three phases:
        1. Validation: Check semantic constraints (type compatibility, implicit
           cast legality, operation-specific rules). Return ComptimeError on
           semantic violations.
        2. Type Inference: Compute the result type. For operations with results,
           this must succeed if validation passed.
        3. Comptime Evaluation: Attempt to evaluate the operation. If evaluation
           fails but type inference succeeded, return the inferred Type.

        Returns:
        - ComptimeEvalSuccess: Evaluation succeeded (with value or control flow)
        - Type: Type inference succeeded, comptime evaluation failed (optional)
        - ComptimeError: Semantic error or required comptime failure

        The types of operands can be queried via context.getInferredType(Value).
        Comptime values can be queried via context.getOrLoadComptimeRValue(Value).
      }],
      /*retTy=*/"::mlir::parsed_csl::ComptimeEvalResult",
      /*methodName=*/"evaluate",
      /*args=*/(ins "::mlir::parsed_csl::ComptimeEvalContext&":$context)
    >,

    InterfaceMethod<
      /*desc=*/[{
        Indicates whether this operation has semantics that require it to be
        evaluated at compile-time, independently of the context it is executed in.

        Evaluation may still be required even if this returns false, e.g., if
        the operation's result is used in a comptime context or if the operation
        is executed within a comptime block.

        If true, evaluate() must either succeed and optionally return a value,
        or fail with a diagnostic explaining why evaluation failed.

        Operations like comptime variable declarations or loads/stores to
        comptime lvalues should return true here. Those must be evaluated at
        compile-time even when not used in a comptime context.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"requiresComptimeBySemantics",
      /*args=*/(ins "const ::mlir::parsed_csl::ComptimeEvalContext&":$context),
      /*body=*/[{
        return false;
      }]
    >,
  ];
}

#endif // PARSED_CSL_COMPTIME_INTERFACES
