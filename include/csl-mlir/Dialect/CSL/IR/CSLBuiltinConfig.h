//===- CSLBuiltinConfig.h - Configuration structs of builtins ---*- C++ -*-===//
//
// Several builtins take a configuration struct literal, e.g.
// `@get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = n, .stride = -1 })`.
// Like cslc, the literal takes its type from the configuration: each field
// is coerced to the type of the field, so a comptime number such as `-1`
// becomes an `i8`. This file describes the fields of these configurations.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINCONFIG_H
#define CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINCONFIG_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>
#include <string>
#include <vector>

namespace mlir::csl {

enum class BuiltinKind : uint32_t;

/// The type a configuration field is coerced to.
enum class ConfigFieldType {
  Keep, ///< The given type: pointers, tasks, colors, queues, tensor accesses.
  U16,
  I8,
  I16,
  Bool,
};

/// The architectures on which a configuration field exists.
enum class ConfigArch { Both, WSE2, WSE3 };

struct ConfigSchema;

/// A field of a configuration struct.
struct ConfigField {
  const char *name;
  ConfigFieldType type = ConfigFieldType::Keep;
  /// The value is a tuple whose elements are coerced to `type`.
  bool isTuple = false;
  /// The value is itself a configuration struct, e.g. `.{ .high = true }`.
  const ConfigSchema *nested = nullptr;
  bool mustBeComptime = false;
  ConfigArch arch = ConfigArch::Both;
};

/// The fields of a configuration struct.
struct ConfigSchema {
  /// The configuration in diagnostics, e.g. "the configuration of a
  /// mem1d_dsd".
  const char *description;
  std::vector<ConfigField> fields;

  /// The field `name` that exists on the target architecture, if any.
  const ConfigField *lookup(StringRef name, bool isWse3) const;
};

/// The schema of argument `argIdx` of builtin `kind`, or null if that
/// argument is not a configuration struct with fixed fields. `args` are the
/// argument values (NoValueAttr if not comptime-known): the schema of
/// @get_dsd depends on its DSD type.
const ConfigSchema *getConfigSchema(BuiltinKind kind, unsigned argIdx,
                                    ArrayRef<TypedAttr> args);

/// The type that a configuration literal of type `literalType` coerces to:
/// the same fields, each of the type its schema gives. Fails if a field is
/// not in `schema` or its value cannot be coerced implicitly.
FailureOr<Type> coerceConfigType(Type literalType, const ConfigSchema &schema);

/// Check a configuration literal of type `literalType` against `schema` on
/// the target architecture. `isFieldComptimeKnown` tells whether the field at
/// the given path of field names is comptime-known. Returns a diagnostic if
/// a field does not exist, has a type that cannot be coerced, or must be
/// comptime-known but is not.
std::optional<std::string> checkConfig(
    Type literalType, const ConfigSchema &schema, bool isWse3,
    function_ref<bool(ArrayRef<StringRef>)> isFieldComptimeKnown);

} // namespace mlir::csl

#endif // CSL_MLIR_DIALECT_CSL_IR_CSLBUILTINCONFIG_H
