//===- CSLBase.td - CSL Dialect Definition ----------------*- tablegen -*-===//
//
// CSL Dialect - Base dialect for Cerebras Software Language.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_BASE
#define CSL_BASE

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// CSL Dialect
//===----------------------------------------------------------------------===//

def CSL_Dialect : Dialect {
  let name = "csl";
  let summary = "Base CSL dialect with shared types and operations";
  let description = [{
    The CSL dialect is the base dialect for Cerebras Software Language,
    providing shared types, attributes, and core operations used by both
    the ParsedCSL and TypedCSL dialects.

    This dialect contains:
    - All runtime types (int, float, bool, void, ptr, array, hardware types)
    - All comptime types (comptime_int, comptime_float, comptime_string, type)
    - All value attributes (constants)
    - Unified LValueType for both parsed and typed stages
    - CompilationUnitOp (top-level container)

    The CSL dialect serves as the foundation, with ParsedCSL and TypedCSL
    building on top of it for their specific purposes.
  }];

  let cppNamespace = "::mlir::csl";

  // Use generated type printer/parser
  let useDefaultTypePrinterParser = 1;

  // Use generated attribute printer/parser
  let useDefaultAttributePrinterParser = 1;

  // Has constant materializer
  let hasConstantMaterializer = 0;

  let extraClassDeclaration = [{
    // Register all types
    void registerTypes();

    // Register all attributes
    void registerAttributes();

    std::vector<std::optional<BuiltinDescriptor>> builtinDescriptors;
    void registerBuiltinDescriptors();

    /// Import name prefix for the fiber runtime module (wse2.csl / wse3.csl).
    static StringRef getFiberRuntimePrefix() { return "__fiber_runtime__"; }

    /// Import name prefix for the resource allocator module (resources.csl).
    static StringRef getAllocRuntimePrefix() { return "__alloc_runtime__"; }

    /// Attribute name for comptime resource bitmaps on CompilationUnitOp.
    static StringRef getComptimeResourceBitmapsAttrName() {
      return "csl.comptime_resource_bitmaps";
    }
  }];
}

#endif // CSL_BASE
