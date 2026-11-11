//===- CSLBuiltinTypes.cpp - Builtin Type Registry ------------------------===//
//
// Single source of truth for CSL builtin type names (i32, f16, bool, etc.).
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinTypes.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

struct BuiltinTypeDef {
  llvm::StringLiteral name;
  Type (*getType)(MLIRContext *);
  bool (*isType)(Type);
};

// clang-format off
const BuiltinTypeDef builtinTypes[] = {
    {"i8",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 8, IntegerType::Signed);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 8 && i.isSigned();
     }},
    {"i16",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 16, IntegerType::Signed);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 16 && i.isSigned();
     }},
    {"i32",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 32, IntegerType::Signed);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 32 && i.isSigned();
     }},
      {"i64",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 64, IntegerType::Signed);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 64 && i.isSigned();
     }},
    {"u8",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 8, IntegerType::Unsigned);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 8 && i.isUnsigned();
     }},
    {"u16",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 16, IntegerType::Unsigned);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 16 && i.isUnsigned();
     }},
    {"u32",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 32, IntegerType::Unsigned);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 32 && i.isUnsigned();
     }},
     {"u64",
     +[](MLIRContext *ctx) -> Type {
       return IntegerType::get(ctx, 64, IntegerType::Unsigned);
     },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 64 && i.isUnsigned();
     }},
    {"f16",
     +[](MLIRContext *ctx) -> Type { return Float16Type::get(ctx); },
     +[](Type t) -> bool { return isa<Float16Type>(t); }},
    {"bf16",
     +[](MLIRContext *ctx) -> Type { return BFloat16Type::get(ctx); },
     +[](Type t) -> bool { return isa<BFloat16Type>(t); }},
    {"cb16",
     +[](MLIRContext *ctx) -> Type { return csl::CB16Type::get(ctx); },
     +[](Type t) -> bool { return isa<csl::CB16Type>(t); }},
    {"f32",
     +[](MLIRContext *ctx) -> Type { return Float32Type::get(ctx); },
     +[](Type t) -> bool { return isa<Float32Type>(t); }},
    {"bool",
     +[](MLIRContext *ctx) -> Type { return IntegerType::get(ctx, 1); },
     +[](Type t) -> bool {
       auto i = dyn_cast<IntegerType>(t);
       return i && i.getWidth() == 1;
     }},
    {"void",
     +[](MLIRContext *ctx) -> Type { return csl::VoidType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::VoidType>(t); }},
    {"type",
     +[](MLIRContext *ctx) -> Type { return csl::TypeType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::TypeType>(t); }},
    {"comptime_int",
     +[](MLIRContext *ctx) -> Type { return csl::ComptimeIntType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::ComptimeIntType>(t); }},
    {"comptime_float",
     +[](MLIRContext *ctx) -> Type { return csl::ComptimeFloatType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::ComptimeFloatType>(t); }},
    {"comptime_string",
     +[](MLIRContext *ctx) -> Type {
       return csl::ComptimeStringType::get(ctx);
     },
     +[](Type t) -> bool { return isa<csl::ComptimeStringType>(t); }},
    {"imported_module",
     +[](MLIRContext *ctx) -> Type {
       return csl::ImportedModuleType::get(ctx);
     },
     +[](Type t) -> bool { return isa<csl::ImportedModuleType>(t); }},
    {"dsr_dest",
     +[](MLIRContext *ctx) -> Type { return csl::DsrDestType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DsrDestType>(t); }},
    {"dsr_src0",
     +[](MLIRContext *ctx) -> Type { return csl::DsrSrc0Type::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DsrSrc0Type>(t); }},
    {"dsr_src1",
     +[](MLIRContext *ctx) -> Type { return csl::DsrSrc1Type::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DsrSrc1Type>(t); }},
    {"dsr_fifo_dest",
     +[](MLIRContext *ctx) -> Type { return csl::DsrFifoDestType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DsrFifoDestType>(t); }},
    {"dsr_fifo_src1",
     +[](MLIRContext *ctx) -> Type { return csl::DsrFifoSrc1Type::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DsrFifoSrc1Type>(t); }},
    {"xdsr",
     +[](MLIRContext *ctx) -> Type { return csl::XdsrType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::XdsrType>(t); }},
    {"sr", +[](MLIRContext *ctx) -> Type { return csl::SrType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::SrType>(t); }},
    {"mem1d_dsd",
     +[](MLIRContext *ctx) -> Type { return csl::Mem1dDsdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::Mem1dDsdType>(t); }},
    {"mem4d_dsd",
     +[](MLIRContext *ctx) -> Type { return csl::Mem4dDsdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::Mem4dDsdType>(t); }},
    {"fabin_dsd",
     +[](MLIRContext *ctx) -> Type { return csl::FabinDsdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::FabinDsdType>(t); }},
    {"fabout_dsd",
     +[](MLIRContext *ctx) -> Type { return csl::FaboutDsdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::FaboutDsdType>(t); }},
    {"fifo_handle",
     +[](MLIRContext *ctx) -> Type { return csl::FifoDsdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::FifoDsdType>(t); }},
    {"color",
     +[](MLIRContext *ctx) -> Type { return csl::ColorType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::ColorType>(t); }},
    {"local_task_id",
     +[](MLIRContext *ctx) -> Type { return csl::LocalTaskIdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::LocalTaskIdType>(t); }},
    {"control_task_id",
     +[](MLIRContext *ctx) -> Type { return csl::ControlTaskIdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::ControlTaskIdType>(t); }},
    {"ut_id",
     +[](MLIRContext *ctx) -> Type { return csl::UtIdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::UtIdType>(t); }},
    {"data_task_id",
     +[](MLIRContext *ctx) -> Type { return csl::DataTaskIdType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DataTaskIdType>(t); }},
    {"input_queue",
     +[](MLIRContext *ctx) -> Type { return csl::InputQueueType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::InputQueueType>(t); }},
    {"output_queue",
     +[](MLIRContext *ctx) -> Type { return csl::OutputQueueType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::OutputQueueType>(t); }},
    {"direction",
     +[](MLIRContext *ctx) -> Type { return csl::DirectionType::get(ctx); },
     +[](Type t) -> bool { return isa<csl::DirectionType>(t); }},
};
// clang-format on

} // namespace

FailureOr<StringRef> csl::getBuiltinTypeName(Type type) {
  for (const auto &def : builtinTypes)
    if (def.isType(type))
      return StringRef(def.name);
  return failure();
}

bool csl::isBuiltinTypeName(StringRef name) {
  for (const auto &def : builtinTypes)
    if (def.name == name)
      return true;
  return false;
}

FailureOr<Type> csl::getBuiltinType(StringRef name, MLIRContext *ctx) {
  for (const auto &def : builtinTypes)
    if (def.name == name)
      return def.getType(ctx);
  return failure();
}

void csl::getAllBuiltinTypeNames(SmallVectorImpl<StringRef> &names) {
  for (const auto &def : builtinTypes)
    names.push_back(def.name);
}
