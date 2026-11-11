//===- CSL.h - CSL Dialect --------------------------------------*- C++ -*-===//
//
// Main header for the CSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_IR_CSL_H
#define CSL_MLIR_DIALECT_CSL_IR_CSL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// CSL Dialect
//===----------------------------------------------------------------------===//

namespace mlir::csl {
struct BuiltinDescriptor;
enum class BuiltinKind : uint32_t;
} // namespace mlir::csl

#include "csl-mlir/Dialect/CSL/IR/CSLDialect.h.inc"

// Include enums first (FloatKind needed by FloatType)
#include "csl-mlir/Dialect/CSL/IR/CSLEnums.h.inc"
#include <variant>

//===----------------------------------------------------------------------===//
// CSL Interfaces
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSLOpInterfaces.h.inc"
#include "csl-mlir/Dialect/CSL/IR/CSLTypeInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// CSL Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "csl-mlir/Dialect/CSL/IR/CSLTypes.h.inc"

//===----------------------------------------------------------------------===//
// CSL Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "csl-mlir/Dialect/CSL/IR/CSLAttrs.h.inc"

//===----------------------------------------------------------------------===//
// CSL Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "csl-mlir/Dialect/CSL/IR/CSLOps.h.inc"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace mlir {
class AttrTypeReplacer;

namespace csl {

// Type constraint helpers
#include "csl-mlir/Dialect/CSL/IR/CSLTypeConstraints.h.inc"

//===----------------------------------------------------------------------===//
// Type Information
//===----------------------------------------------------------------------===//

/// Returns the size of a CSL type in bytes, or std::nullopt if not applicable.
///
/// Handles MLIR builtin integer and float types directly (IntegerType,
/// FloatType), then delegates to TypeInfoInterface for CSL-defined types.
/// Returns nullopt for types without a statically known size (e.g., DSD types,
/// void, lvalue references) and for types not recognized as CSL types.
std::optional<uint32_t> getSizeInBytes(Type type);

uint32_t getAlignmentForType(Type type);

/// Returns whether a type is a CSL runtime type — one that can exist as a
/// value on the hardware at runtime.
///
/// Returns true for scalar integer/float types, enums, hardware-visible types
/// (colors, DSDs, DSRs, etc.), and for aggregates of runtime types: arrays of
/// runtime elements, structs whose fields are runtime types or void, and
/// pointers to runtime types or functions.
/// Returns false for comptime-only types and meta-types (void, lvalue).
bool isRuntimeType(Type type);

/// Returns whether `type` has exactly one value that carries no information:
/// void, or a struct whose fields all have such a type.  Values of these
/// types are always comptime-known.
bool isVoidLikeType(Type type);

/// Returns the only value of a type for which isVoidLikeType() holds.
TypedAttr getVoidLikeValue(Type type);

/// Register replacements that remove the void fields of struct types and the
/// void fields of struct values.  Void fields take no memory and CSL does not
/// allow them in runtime values.  A struct left without fields becomes the
/// empty tuple.
void addVoidFieldStripping(AttrTypeReplacer &replacer);

/// Returns `type` with the void fields of all structs in it removed.
Type stripVoidFields(Type type);

/// Returns whether a type contains per-PE layout information.
/// Returns true for PerPEArrayType and for composite types (arrays, structs)
/// that recursively contain per-PE arrays.
bool hasPerPELayout(Type type);

/// Returns the size of a type in bytes for a specific PE at coordinate (x, y).
/// For types without per-PE layout, this delegates to getSizeInBytes().
std::optional<uint32_t> getSizeInBytesForPE(Type type, int64_t x, int64_t y);

//===----------------------------------------------------------------------===//
// Export Compatibility
//===----------------------------------------------------------------------===//

/// Check if a type is export compatible for variable declarations.
/// Export compatible types are:
/// - Built-in types: i8, i16, i32, i64, u8, u16, u32, u64, f16, f32,
/// bool,
/// color
/// - Array types [N]T where T is export compatible
/// - Pointer types *T or *const T where T is export compatible
/// - Function pointer types *const fn(...) where the function type is
/// export
///   compatible (for function declarations)
/// - Enum types
///
/// NOT export compatible:
/// - Comptime-only types (comptime_int, comptime_float,
/// comptime_string, type)
/// - Struct types
/// - DSD and DSR types
/// - Direction type
/// - Range type
/// - Void (except in function return type position)
/// - Function types (as opposed to function pointer types)
bool isExportCompatibleType(Type type);

/// Check if a function type is export compatible.
/// A function type fn(t1, t2, ..., tn) t is export compatible if:
/// - All argument types are export compatible for variable
/// declarations
/// - The return type is either export compatible for variable
/// declarations,
///   or is void
bool isExportCompatibleFunctionType(FunctionType funcType);

//===----------------------------------------------------------------------===//
// Pretty Printing
//===----------------------------------------------------------------------===//

/// Forward declaration for nominal type registry used during
/// emission.
class NominalTypeRegistry;

/// Pretty print a CSL type in CSL syntax (e.g., "i32", "[10]u16",
/// "*f32").
/// If a NominalTypeRegistry is provided, nominal types (enums and
/// named
/// structs) will be printed using their generated names instead of
/// inline.
void printType(llvm::raw_ostream &os, Type type,
               const NominalTypeRegistry *registry = nullptr);

/// Pretty print a typed attribute value in CSL syntax (e.g., "42",
/// "true").
/// If a NominalTypeRegistry is provided, enum values will be printed
/// as
/// TypeName.element instead of just .element.
void printValue(llvm::raw_ostream &os, TypedAttr attr,
                const NominalTypeRegistry *registry = nullptr);

/// Return a pretty-printed string for a CSL type.
std::string prettyPrintType(Type type);

/// Return a pretty-printed string for a typed attribute value.
std::string prettyPrintValue(TypedAttr attr);

/// Print a string with CSL escape sequences (\n, \t, \r, \\, \").
void printEscapedString(llvm::StringRef str, llvm::raw_ostream &os);

//===----------------------------------------------------------------------===//
// Parameter Bindings
//===----------------------------------------------------------------------===//

/// A parameter binding: value and location where it was defined
struct ParamBinding {
  TypedAttr value;   // Value the parameter is bound to
  Location location; // Location where the value was defined
};

/// A map of parameter name to parameter binding
using ParamBindingsMap = llvm::DenseMap<StringRef, ParamBinding>;

} // namespace csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_CSL_IR_CSL_H
