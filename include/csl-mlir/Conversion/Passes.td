//===- Passes.td - CSL Transformation Passes --------------*- tablegen -*-===//
//
// Definitions of transformation passes for CSL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_CONVERSION_PASSES
#define CSL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Conversion Passes
//===----------------------------------------------------------------------===//

def ConvertParsedCSLToTypedCSLPass : Pass<"convert-parsedcsl-to-typedcsl", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Lower ParsedCSL operations to TypedCSL operations";
  let description = [{
  }];

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect", "::mlir::typed_csl::TypedCSLDialect"];
}

//===----------------------------------------------------------------------===//
// Target Passes
//===----------------------------------------------------------------------===//

def EmitCSLPass : Pass<"emit-csl", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Emit CSL source code from TypedCSL IR";
  let description = [{
    This pass converts TypedCSL IR to CSL source code. SSA values that are
    used only once are inlined as expressions; values used multiple times
    generate named temporary variables.

    The pass outputs to the raw_ostream configured externally.
  }];

  let options = [
    Option<"varPrefix", "var-prefix", "std::string", /*default=*/"\"__csl_tmp\"",
           "Prefix for auto-generated variable names">,
    Option<"indentWidth", "indent-width", "unsigned", /*default=*/"4",
           "Spaces per indent level (default: 4)">
  ];

  let dependentDialects = ["::mlir::typed_csl::TypedCSLDialect"];
}

#endif // CSL_CONVERSION_PASSES
