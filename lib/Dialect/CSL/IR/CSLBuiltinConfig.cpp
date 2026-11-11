//===- CSLBuiltinConfig.cpp - Configuration structs of builtins -----------===//
//
// The fields follow the CSL documentation of DSDs, DSRs and DSD operations,
// checked against cslc 2.10 where the documentation leaves details open.
// Where it is unclear whether cslc accepts a field, the field is listed, so
// a valid program is never rejected. For the same reason, a field only must
// be comptime-known where cslc was seen to require it: the documentation
// also calls e.g. the strides of a mem4d_dsd comptime-known, but cslc
// accepts runtime strides. A tensor access is comptime-only by its type.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinConfig.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>

using namespace mlir;
using namespace mlir::csl;

namespace {

using FT = ConfigFieldType;
using CA = ConfigArch;

ConfigField field(const char *name, FT type = FT::Keep) {
  return ConfigField{name, type};
}
ConfigField comptimeField(const char *name, FT type = FT::Keep) {
  ConfigField f{name, type};
  f.mustBeComptime = true;
  return f;
}
ConfigField onArch(ConfigField f, CA arch) {
  f.arch = arch;
  return f;
}
ConfigField tupleField(const char *name, FT type) {
  ConfigField f{name, type};
  f.isTuple = true;
  return f;
}
ConfigField nestedField(const char *name, const ConfigSchema *nested) {
  ConfigField f{name};
  f.nested = nested;
  return f;
}

// Nested configurations whose fields select one option, e.g.
// `.priority = .{ .high = true }`.
const ConfigSchema priority{
    "the priority",
    {field("high", FT::Bool), field("medium", FT::Bool),
     field("low", FT::Bool)}};
const ConfigSchema simdModeWse2{
    "the SIMD mode",
    {field("simd_32", FT::Bool), field("simd_64", FT::Bool),
     field("simd_32_or_64", FT::Bool), field("simd_16", FT::Bool)}};
const ConfigSchema simdModeWse3{
    "the SIMD mode",
    {field("simd_off", FT::Bool), field("simd_max", FT::Bool)}};
const ConfigSchema zeroSource{
    "the zero source",
    {field("first_source", FT::Bool), field("second_source", FT::Bool)}};
const ConfigSchema fifoAction{
    "the FIFO action",
    {field("test_or_suspend", FT::Bool), field("terminate", FT::Bool),
     field("suspend", FT::Bool), field("fault", FT::Bool)}};
const ConfigSchema onControl{
    "the action on control wavelets",
    {field("terminate", FT::Bool), field("activate"), field("unblock")}};

const ConfigSchema mem1dDsd{
    "the configuration of a mem1d_dsd",
    {field("base_address"), field("extent", FT::U16), field("stride", FT::I8),
     field("offset", FT::I16), field("tensor_access"),
     field("wavelet_index_offset", FT::Bool)}};

const ConfigSchema mem4dDsd{
    "the configuration of a mem4d_dsd",
    {field("base_address"), field("offset", FT::I16),
     tupleField("stride", FT::I16), tupleField("extent", FT::U16),
     field("tensor_access"), field("wavelet_index_offset", FT::Bool)}};

const ConfigSchema fabinDsd{
    "the configuration of a fabin_dsd",
    {comptimeField("extent", FT::U16), field("input_queue"),
     field("fabric_color"), onArch(nestedField("priority", &priority), CA::WSE3),
     onArch(nestedField("simd_mode", &simdModeWse2), CA::WSE2),
     onArch(nestedField("simd_mode", &simdModeWse3), CA::WSE3),
     field("control_transform", FT::Bool)}};

const ConfigSchema faboutDsd{
    "the configuration of a fabout_dsd",
    {comptimeField("extent", FT::U16), field("output_queue"),
     field("fabric_color"), onArch(nestedField("priority", &priority), CA::WSE3),
     field("control", FT::Bool),
     onArch(field("wavelet_index_offset", FT::Bool), CA::WSE2),
     field("advance_switch", FT::Bool),
     onArch(nestedField("simd_mode", &simdModeWse2), CA::WSE2),
     onArch(nestedField("simd_mode", &simdModeWse3), CA::WSE3),
     nestedField("zero", &zeroSource), field("control_transform", FT::Bool)}};

const ConfigSchema dsdOperation{
    "the configuration of a DSD operation",
    {field("async", FT::Bool), field("activate"), field("unblock"),
     nestedField("on_control", &onControl),
     onArch(nestedField("priority", &priority), CA::WSE2),
     field("index", FT::U16), onArch(field("ut_id"), CA::WSE3)}};

const ConfigSchema loadToDsr{
    "the configuration of @load_to_dsr",
    {field("async", FT::Bool), field("activate"), field("unblock"),
     nestedField("on_control", &onControl), nestedField("priority", &priority),
     field("save_address", FT::Bool), field("single_step", FT::Bool),
     onArch(field("ut_id"), CA::WSE3)}};

const ConfigSchema loadToDsrXdsr{
    "the configuration of @load_to_dsr_xdsr_sr",
    {field("save_address", FT::Bool), field("single_step", FT::Bool)}};

const ConfigSchema allocateFifo{
    "the configuration of @allocate_fifo",
    {nestedField("empty_action", &fifoAction),
     nestedField("full_action", &fifoAction), field("activate_pop"),
     field("activate_push"), nestedField("priority", &priority),
     onArch(nestedField("simd_mode", &simdModeWse2), CA::WSE2),
     field("dest"), field("src"), field("xdsr")}};

const ConfigSchema dfilt{
    "the configuration of @dfilt",
    {field("async", FT::Bool), field("activate"), field("unblock"),
     nestedField("priority", &priority), onArch(field("ut_id"), CA::WSE3)}};

const ConfigSchema initializeQueue{
    "the configuration of @initialize_queue",
    {field("color"), nestedField("priority", &priority),
     onArch(field("ctrl_table_id"), CA::WSE3),
     onArch(field("dense_mode", FT::Bool), CA::WSE3)}};

/// The builtins whose last argument is an optional DSD operation
/// configuration.
bool isDsdOperation(BuiltinKind kind) {
  switch (kind) {
  case BuiltinKind::Faddhs:
  case BuiltinKind::Fmachs:
  case BuiltinKind::Fmacs:
  case BuiltinKind::Fmaxh:
  case BuiltinKind::Fmaxs:
  case BuiltinKind::Fmovh:
  case BuiltinKind::Fmovs:
  case BuiltinKind::Mov16:
  case BuiltinKind::Mov32:
  case BuiltinKind::Slr16:
  case BuiltinKind::Xp162fs:
    return true;
  default:
    return false;
  }
}

Type getFieldType(FT type, MLIRContext *ctx) {
  switch (type) {
  case FT::Keep:
    return Type();
  case FT::U16:
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
  case FT::I8:
    return IntegerType::get(ctx, 8, IntegerType::Signed);
  case FT::I16:
    return IntegerType::get(ctx, 16, IntegerType::Signed);
  case FT::Bool:
    return IntegerType::get(ctx, 1);
  }
  llvm_unreachable("unknown configuration field type");
}

/// The fields of `schema` that exist on the target architecture.
std::string listFields(const ConfigSchema &schema, bool isWse3) {
  std::string list;
  llvm::raw_string_ostream os(list);
  bool first = true;
  for (const ConfigField &f : schema.fields) {
    if (f.arch == (isWse3 ? CA::WSE2 : CA::WSE3))
      continue;
    if (!first)
      os << ", ";
    os << f.name;
    first = false;
  }
  return list;
}

/// The type that a value of type `valueType` has as a field of type
/// `target`, or failure. Only a comptime number gets the field type; like
/// cslc, a value of an integer type is taken as it is if it is comptime-known
/// or not wider than the field, e.g. a u16 for an i16 stride.
FailureOr<Type> coerceScalar(Type valueType, Type target, bool comptimeKnown) {
  if (valueType == target)
    return target;
  if (mlir::isa<ComptimeIntType, ComptimeFloatType>(valueType))
    return isImplicitCastAllowed(valueType, target, /*comptimeKnown=*/true)
               ? FailureOr<Type>(target)
               : FailureOr<Type>(failure());
  auto valueInt = mlir::dyn_cast<IntegerType>(valueType);
  auto targetInt = mlir::dyn_cast<IntegerType>(target);
  if (valueInt && targetInt && valueInt.getWidth() > 1 &&
      targetInt.getWidth() > 1 &&
      (comptimeKnown || valueInt.getWidth() <= targetInt.getWidth()))
    return valueType;
  return failure();
}

/// The type `valueType` of `field` coerces to, or failure.
FailureOr<Type> coerceField(Type valueType, const ConfigField &f,
                            bool comptimeKnown) {
  if (f.nested)
    return coerceConfigType(valueType, *f.nested);
  Type target = getFieldType(f.type, valueType.getContext());
  if (!target)
    return valueType;
  if (!f.isTuple)
    return coerceScalar(valueType, target, comptimeKnown);
  auto tuple = mlir::dyn_cast<StructType>(valueType);
  if (!tuple || !tuple.isTuple())
    return failure();
  SmallVector<Type> elementTypes;
  for (Type element : tuple.getFieldTypes()) {
    FailureOr<Type> coerced = coerceScalar(element, target, comptimeKnown);
    if (failed(coerced))
      return failure();
    elementTypes.push_back(*coerced);
  }
  return (Type)StructType::get(valueType.getContext(), elementTypes);
}

} // namespace

const ConfigField *ConfigSchema::lookup(StringRef name, bool isWse3) const {
  for (const ConfigField &f : fields)
    if (name == f.name && f.arch != (isWse3 ? CA::WSE2 : CA::WSE3))
      return &f;
  return nullptr;
}

const ConfigSchema *mlir::csl::getConfigSchema(BuiltinKind kind,
                                               unsigned argIdx,
                                               ArrayRef<TypedAttr> args) {
  switch (kind) {
  case BuiltinKind::GetDsd: {
    if (argIdx != 1)
      return nullptr;
    auto dsdType = mlir::dyn_cast<csl::TypeAttr>(args[0]);
    if (!dsdType)
      return nullptr;
    Type type = dsdType.getValue();
    if (mlir::isa<Mem1dDsdType>(type))
      return &mem1dDsd;
    if (mlir::isa<Mem4dDsdType>(type))
      return &mem4dDsd;
    if (mlir::isa<FabinDsdType>(type))
      return &fabinDsd;
    if (mlir::isa<FaboutDsdType>(type))
      return &faboutDsd;
    return nullptr;
  }
  case BuiltinKind::LoadToDsr:
    return argIdx == 2 ? &loadToDsr : nullptr;
  case BuiltinKind::LoadToDsrXdsrSr:
    return argIdx == 4 ? &loadToDsrXdsr : nullptr;
  case BuiltinKind::AllocateFifo:
    return argIdx == 1 ? &allocateFifo : nullptr;
  case BuiltinKind::Dfilt:
    return argIdx == 1 ? &dfilt : nullptr;
  case BuiltinKind::InitializeQueue:
    return argIdx == 1 ? &initializeQueue : nullptr;
  default:
    // The configuration of a DSD operation is its optional last argument.
    if (isDsdOperation(kind) && argIdx + 1 == args.size() &&
        mlir::isa<StructType>(args[argIdx].getType()))
      return &dsdOperation;
    return nullptr;
  }
}

FailureOr<Type> mlir::csl::coerceConfigType(Type literalType,
                                            const ConfigSchema &schema) {
  auto structType = mlir::dyn_cast<StructType>(literalType);
  if (!structType)
    return failure();
  // An empty literal has no fields to coerce; a named struct already has its
  // field types.
  if (structType.getNumFields() == 0 || !structType.isAnonymous())
    return literalType;
  if (structType.isTuple())
    return failure();
  // checkConfig has reported anything invalid: the architecture of a field,
  // and a runtime value that does not coerce. So any variant of a field that
  // coerces gives its type, and every value is treated as comptime-known.
  SmallVector<Type> fieldTypes;
  for (auto [name, type] :
       llvm::zip(structType.getFieldNames(), structType.getFieldTypes())) {
    FailureOr<Type> coerced = failure();
    for (const ConfigField &f : schema.fields) {
      if (name != f.name)
        continue;
      coerced = coerceField(type, f, /*comptimeKnown=*/true);
      if (succeeded(coerced))
        break;
    }
    if (failed(coerced))
      return failure();
    fieldTypes.push_back(*coerced);
  }
  return (Type)StructType::get(literalType.getContext(),
                               structType.getFieldNames(), fieldTypes);
}

std::optional<std::string> mlir::csl::checkConfig(
    Type literalType, const ConfigSchema &schema, bool isWse3,
    function_ref<bool(ArrayRef<StringRef>)> isFieldComptimeKnown) {
  const char *arch = isWse3 ? "WSE-3" : "WSE-2";
  std::string message;
  llvm::raw_string_ostream os(message);

  SmallVector<StringRef> path;
  // `otherArch` is the variant of `current` on the other architecture, to
  // tell a field that only exists there.
  std::function<bool(Type, const ConfigSchema &, const ConfigSchema &)> check =
      [&](Type type, const ConfigSchema &current,
          const ConfigSchema &otherArch) -> bool {
    auto structType = mlir::dyn_cast<StructType>(type);
    if (!structType || !structType.isAnonymous() ||
        structType.getNumFields() == 0)
      return true;
    if (structType.isTuple()) {
      os << current.description
         << " must be given with field names, e.g. `.{ .name = value }`";
      return false;
    }
    for (auto [name, fieldType] :
         llvm::zip(structType.getFieldNames(), structType.getFieldTypes())) {
      const ConfigField *f = current.lookup(name, isWse3);
      if (!f) {
        os << "'" << name.getValue() << "' is not a field of "
           << current.description;
        if (otherArch.lookup(name, !isWse3))
          os << " on " << arch;
        os << "; its fields on " << arch << " are "
           << listFields(current, isWse3);
        return false;
      }
      path.push_back(name.getValue());
      auto popPath = llvm::scope_exit([&] { path.pop_back(); });
      if (f->nested) {
        if (!mlir::isa<StructType>(fieldType)) {
          os << "field '" << name.getValue() << "' of " << current.description
             << " must be a struct literal, e.g. `." << name.getValue()
             << " = .{ ." << f->nested->fields.front().name << " = true }`";
          return false;
        }
        const ConfigField *otherField = otherArch.lookup(name, !isWse3);
        if (!check(fieldType, *f->nested,
                   otherField && otherField->nested ? *otherField->nested
                                                    : *f->nested))
          return false;
        continue;
      }
      if (failed(coerceField(fieldType, *f, isFieldComptimeKnown(path)))) {
        os << "field '" << name.getValue() << "' of " << current.description
           << " has type '" << prettyPrintType(fieldType)
           << "', which does not coerce to '";
        Type target = getFieldType(f->type, type.getContext());
        if (f->isTuple)
          os << "a tuple of " << prettyPrintType(target) << "'";
        else
          os << prettyPrintType(target) << "'; convert it with `@as("
             << prettyPrintType(target) << ", ...)`";
        return false;
      }
      if (f->mustBeComptime && !isFieldComptimeKnown(path)) {
        os << "field '" << name.getValue() << "' of " << current.description
           << " must be known at compile time";
        return false;
      }
    }
    return true;
  };
  if (check(literalType, schema, schema))
    return std::nullopt;
  return message;
}
