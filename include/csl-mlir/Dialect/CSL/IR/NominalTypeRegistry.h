//===- NominalTypeRegistry.h - CSL emitted type registry -------*- C++ -*-===//
//
// Registry for mapping concrete types to generated declarations and names for
// CSL emission.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_IR_NOMINALTYPEREGISTRY_H
#define CSL_MLIR_DIALECT_CSL_IR_NOMINALTYPEREGISTRY_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

namespace mlir {
namespace csl {

// Forward declarations
class EnumType;
class StructType;
class CompilationUnitOp;

/// Registry for mapping concrete types that need declarations during CSL
/// emission to generated names.
///
/// The complete MLIR type is the key. In particular, generic nominal type
/// definitions can produce multiple concrete types from one definition
/// location, and each concrete instantiation needs its own CSL declaration.
class NominalTypeRegistry {
public:
  /// Collect all types requiring declarations from a compilation unit.
  void collectTypes(CompilationUnitOp unit);

  /// Get the generated name for a registered type.
  std::optional<StringRef> getName(Type type) const;

  /// Emit type definitions to the output stream.
  void emitDefinitions(raw_ostream &os);

private:
  /// Generated names keyed by complete concrete type. A nominal type's
  /// definition location remains part of the MLIR type, preserving nominal
  /// identity while distinguishing generic instantiations.
  DenseMap<Type, std::string> typeToName;

  /// Ordered list of types to emit (maintains emission order).
  SmallVector<Type> typesInOrder;

  /// Counters for generating unique names.
  unsigned enumCounter = 0;
  unsigned structCounter = 0;

  void collectFromType(Type type);
  void collectFromAttr(Attribute attr);
  void registerType(Type type);
};

} // namespace csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_CSL_IR_NOMINALTYPEREGISTRY_H
