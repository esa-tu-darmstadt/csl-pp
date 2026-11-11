//===- CSLOps.td - CSL Operation Definitions --------------*- tablegen -*-===//
//
// Operation definitions for the CSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_OPS
#define CSL_OPS

include "csl-mlir/Dialect/CSL/IR/CSLBase.td"
include "csl-mlir/Dialect/CSL/IR/CSLTypes.td"
include "csl-mlir/Dialect/CSL/IR/CSLAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// CSL Operation Definitions
//===----------------------------------------------------------------------===//

class CSL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSL_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CompilationUnitOp - Top-level module
//===----------------------------------------------------------------------===//

def CSL_CompilationUnitOp : CSL_Op<"compilation_unit", [
    NoTerminator,
    SingleBlock,
    IsolatedFromAbove,
    SymbolTable,
    Symbol,
    SymbolName
]> {
  let summary = "Top-level compilation unit";
  let description = [{
    Represents a complete CSL source file. This is the unified compilation unit
    operation used by both ParsedCSL and TypedCSL stages.

    During the ParsedCSL stage, the body contains parsed operations with
    unresolved types and comptime constructs.

    After conversion to TypedCSL, the body contains fully typed operations
    with all comptime evaluation completed.

    The SymbolTable trait allows this op to contain named symbols (functions,
    globals, etc.) that can be referenced from within the compilation unit.
    A compilation unit may itself have a symbol name. This permits another IR
    construct to identify one function in a collection of compilation units
    with a nested symbol reference such as `@kernels::@update`. The name is
    optional because a standalone CSL source file needs no enclosing symbol.

    `source_path`, when present, records the source file from which the unit was
    parsed. Compiler clients can use it to recreate file-relative module
    resolution after serializing and reloading the IR.

    Examples:
    ```mlir
    // ParsedCSL stage
    csl.compilation_unit @source {
      parsed_csl.func @foo(%x: !parsed_csl.unresolved) -> !parsed_csl.unresolved {
        parsed_csl.return %x
      }
    }

    // TypedCSL stage
    csl.compilation_unit {
      typed_csl.func @foo(%x: !csl.int<32, signed>) -> !csl.int<32, signed> {
        typed_csl.return %x : !csl.int<32, signed>
      }
    }
    ```
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<StrAttr>:$source_path);
  let regions = (region SizedRegion<1>:$region);
  let builders = [
    OpBuilder<(ins CArg<"std::optional<StringRef>", "{}">:$name)>
  ];

  let assemblyFormat = "($sym_name^)? $region attr-dict";

  let extraClassDeclaration = [{
    /// Standalone compilation units are allowed to remain anonymous.
    bool isOptionalSymbol() { return true; }

    /// Compilation units are always visible to the IR that contains them.
    mlir::SymbolTable::Visibility getVisibility() {
      return mlir::SymbolTable::Visibility::Public;
    }
    void setVisibility(mlir::SymbolTable::Visibility visibility) {
      assert(visibility == mlir::SymbolTable::Visibility::Public &&
             "csl.compilation_unit symbols are always public");
    }
  }];
}

#endif // CSL_OPS
