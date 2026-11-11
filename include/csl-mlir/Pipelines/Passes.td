//===- Passes.td - CSL Pipeline Passes ---------------------*- tablegen -*-===//
//
// Pass definitions for the CSL pipeline library.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_PIPELINES_PASSES_TD
#define CSL_MLIR_PIPELINES_PASSES_TD

include "mlir/Pass/PassBase.td"

def ProcessTileCodePass : Pass<"process-tile-code", "::mlir::csl::CompilationUnitOp"> {
  let summary = "Process @set_tile_code calls and compile PE programs";
  let description = [{
    Processes @set_tile_code calls in layout blocks:
    1. Finds @set_rectangle to extract rectangle dimensions
    2. For each unique (filename, params_struct) combination:
       - Parses the PE program file
       - Runs the CSL pipeline with params as bindings
       - Emits the transformed file
    3. Rewrites @set_tile_code calls with new filename and empty struct

    Rectangle dimensions for PE compilation are extracted from @set_rectangle
    in the layout block, NOT from command-line options.
  }];

  let options = [
    Option<"cslFlags", "csl-flags", "::mlir::csl::CslFlags",
           /*default=*/"::mlir::csl::CslFlags{}",
           "CSL compilation flags">,
    Option<"peOutputDir", "pe-output-dir", "std::string", /*default=*/"\"\"",
           "Output directory for compiled PE programs (empty = auto temp dir)">
  ];

  let dependentDialects = ["::mlir::parsed_csl::ParsedCSLDialect", "::mlir::typed_csl::TypedCSLDialect"];
}

#endif // CSL_MLIR_PIPELINES_PASSES_TD
