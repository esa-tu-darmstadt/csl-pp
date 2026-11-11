//===- ParsedCSLBase.td - ParsedCSL Dialect Definition ----*- tablegen -*-===//
//
// ParsedCSL Dialect - Untyped CSL IR directly from parser.
//
//===----------------------------------------------------------------------===//

#ifndef PARSED_CSL_BASE
#define PARSED_CSL_BASE

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// ParsedCSL Dialect
//===----------------------------------------------------------------------===//

def ParsedCSL_Dialect : Dialect {
  let name = "parsed_csl";
  let summary = "Parsed CSL dialect with unresolved types";
  let description = [{
    The ParsedCSL dialect represents CSL code directly after parsing, before
    type resolution and comptime evaluation. All value types are unresolved,
    and type information flows through dynamic type operands and analysis.

    This dialect is the entry point for CSL source code into the compiler.

    Key properties:
    - ALL value SSA results have type !parsed_csl.unresolved
    - Type information flows through SSA values of type !parsed_csl.type
    - Comptime variables and function calls are allowed
    - Functions are NOT isolated from above (can reference outer scopes)
    - No explicit casts (implicit type coercion handled during conversion)
    - Supports comptime-only types (comptime_int, type, etc.)
    - Global consts are just variables with const qualifier
  }];

  let cppNamespace = "::mlir::parsed_csl";

  // Use generated type printer/parser
  let useDefaultTypePrinterParser = 1;

  // Use generated attribute printer/parser
  let useDefaultAttributePrinterParser = 1;

  // Has constant materializer
  let hasConstantMaterializer = 1;

  let dependentDialects = ["mlir::csl::CSLDialect"];

  let extraClassDeclaration = [{
    // Register all types
    void registerTypes();

    // Register all attributes
    void registerAttributes();
  }];
}

#endif // PARSED_CSL_BASE
