//===- CSLPrettyPrint.cpp - Pretty printing for CSL types/values ----------===//
//
// Implementation of pretty printing utilities for CSL types and attributes.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinTypes.h"
#include "csl-mlir/Dialect/CSL/IR/NominalTypeRegistry.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Zero Value Printing (for @type_of pattern)
//===----------------------------------------------------------------------===//

/// Prints a zero/default value for the given type, suitable for use in
/// @type_of(.{...}) patterns. Returns true if successful, false if the type
/// doesn't support zero value printing.
///
/// For primitives: prints @as(type, zero)
/// For structs: prints .{ ... } with recursive calls for fields
static bool printZeroValue(llvm::raw_ostream &os, Type type,
                           const NominalTypeRegistry *registry);

//===----------------------------------------------------------------------===//
// Type Pretty Printing
//===----------------------------------------------------------------------===//

void csl::printType(llvm::raw_ostream &os, Type type,
                    const NominalTypeRegistry *registry) {
  // Try the builtin type table first (handles all simple named types).
  if (auto name = getBuiltinTypeName(type); succeeded(name)) {
    os << *name;
    return;
  }

  llvm::TypeSwitch<Type>(type)
      .Case<ArrayType>([&](ArrayType arrayType) {
        // Print dimensions in CSL syntax: [10, 20]elementType
        os << "[";
        llvm::interleaveComma(arrayType.getShape(), os);
        os << "]";
        // Workaround for a cslc parser bug: `[N][*]T`, `[N]*T`, and `[N][M]T`
        // ambiguate with the array's own dimension list. Wrap a pointer- or
        // array-element type in parens so emission round-trips through cslc.
        Type elemTy = arrayType.getElementType();
        bool needsParens =
            mlir::isa<PointerType>(elemTy) || mlir::isa<ArrayType>(elemTy);
        if (needsParens)
          os << "(";
        printType(os, elemTy, registry);
        if (needsParens)
          os << ")";
      })
      .Case<PerPEArrayType>([&](PerPEArrayType ppType) {
        // Print as [dim0, per_pe(...), dim1]elemType
        os << "[";
        for (unsigned i = 0; i < ppType.getRank(); ++i) {
          if (i > 0)
            os << ", ";
          if (ppType.isDimPerPE(i)) {
            os << "per_pe(";
            llvm::interleaveComma(ppType.getPerPEDimSizes(i), os);
            os << ")";
          } else {
            os << ppType.getStaticDimSize(i);
          }
        }
        os << "]";
        printType(os, ppType.getElementType(), registry);
      })
      .Case<PointerType>([&](PointerType ptrType) {
        if (ptrType.getIsManyPtr()) {
          os << "[*]";
        } else {
          os << "*";
        }
        if (ptrType.getIsConst()) {
          os << "const ";
        }
        printType(os, ptrType.getPointeeType(), registry);
      })
      .Case<StructType>([&](StructType structType) {
        // Named structs and structural anonymous structs which require a
        // generated declaration are both recorded by the emission registry.
        if (registry) {
          if (auto name = registry->getName(structType)) {
            os << *name;
            return;
          }
        }

        // CSL supports anonymous struct values, but cannot spell anonymous
        // struct types. Derive the type from a representative value instead.
        std::string zeroValueStr;
        llvm::raw_string_ostream zeroOs(zeroValueStr);
        if (printZeroValue(zeroOs, structType, registry)) {
          os << "@type_of(" << zeroValueStr << ")";
          return;
        }

        // Fallback to current syntax for diagnostics or unsupported types
        if (structType.isTuple()) {
          // Tuple syntax: .{ type1, type2, ... }
          os << (structType.getNumFields() == 0 ? ".{" : ".{ ");
          for (size_t i = 0; i < structType.getNumFields(); ++i) {
            if (i > 0)
              os << ", ";
            printType(os, structType.getFieldType(i), registry);
          }
          os << (structType.getNumFields() == 0 ? "}" : " }");
        } else {
          // Struct syntax: .{ name1: type1, name2: type2, ... }
          os << ".{ ";
          for (size_t i = 0; i < structType.getNumFields(); ++i) {
            if (i > 0)
              os << ", ";
            auto fieldName = structType.getFieldName(i);
            if (succeeded(fieldName)) {
              os << *fieldName << ": ";
            }
            printType(os, structType.getFieldType(i), registry);
          }
          os << " }";
        }
      })
      .Case<LValueType>([&](LValueType lvalueType) {
        os << "lvalue<";
        // Print the underlying value type
        printType(os, lvalueType.getValueType(), registry);
        os << ">";
      })
      .Case<EnumType>([&](EnumType enumType) {
        // Check registry for enum types
        if (registry) {
          if (auto name = registry->getName(enumType)) {
            os << *name;
            return;
          }
        }
        // Print enum type definition: enum(underlying_type) { elem1, elem2 =
        // val, ... }
        os << "enum(";
        printType(os, enumType.getUnderlyingType(), registry);
        os << ") { ";
        auto names = enumType.getElementNames();
        auto values = enumType.getElementValues();
        for (size_t i = 0; i < names.size(); ++i) {
          if (i > 0)
            os << ", ";
          os << names[i].getValue();
          // Only print explicit value if it's not the auto-assigned default
          // (i.e., if it doesn't follow from the previous value)
          int64_t expectedValue = (i == 0) ? 0 : values[i - 1] + 1;
          if (values[i] != expectedValue) {
            os << " = " << values[i];
          }
        }
        os << " }";
      })
      .Case<FunctionType>([&](FunctionType funcType) {
        os << "fn(";
        for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
          if (i > 0)
            os << ", ";
          printType(os, funcType.getInput(i), registry);
        }
        os << ") ";
        if (funcType.getNumResults() == 0) {
          os << "void";
        } else if (funcType.getNumResults() == 1) {
          printType(os, funcType.getResult(0), registry);
        } else {
          os << "(";
          for (size_t i = 0; i < funcType.getNumResults(); ++i) {
            if (i > 0)
              os << ", ";
            printType(os, funcType.getResult(i), registry);
          }
          os << ")";
        }
      })
      // Types not in the builtin table (compound or rare types)
      .Case<FifoDsdType>([&](FifoDsdType) { os << "fifo_dsd"; })
      .Case<RangeType>([&](RangeType rangeType) {
        os << "range<";
        printType(os, rangeType.getElementType(), registry);
        os << ">";
      })
      .Default([&](Type t) {
        // Fallback to MLIR printing for unknown types
        t.print(os);
      });
}

//===----------------------------------------------------------------------===//
// Zero Value Printing Implementation
//===----------------------------------------------------------------------===//

static bool printZeroValue(llvm::raw_ostream &os, Type type,
                           const NominalTypeRegistry *registry) {
  return llvm::TypeSwitch<Type, bool>(type)
      .Case<IntegerType>([&](IntegerType intType) {
        os << "@as(";
        csl::printType(os, type, registry);
        os << ", ";
        if (intType.getWidth() == 1)
          os << "false";
        else
          os << "0";
        os << ")";
        return true;
      })
      .Case<Float16Type, Float32Type, Float64Type, BFloat16Type, CB16Type>(
          [&](auto) {
            os << "@as(";
            csl::printType(os, type, registry);
            os << ", 0.0)";
            return true;
          })
      .Case<StructType>([&](StructType structType) {
        // Recursively print zero values for all fields
        os << (structType.getNumFields() == 0 ? ".{" : ".{ ");
        for (size_t i = 0; i < structType.getNumFields(); ++i) {
          if (i > 0)
            os << ", ";

          // Named field: .fieldName = value
          if (!structType.isTuple()) {
            auto fieldName = structType.getFieldName(i);
            if (succeeded(fieldName))
              os << "." << *fieldName << " = ";
          }

          // Recursive call for field type
          if (!printZeroValue(os, structType.getFieldType(i), registry))
            return false;
        }
        os << (structType.getNumFields() == 0 ? "}" : " }");
        return true;
      })
      .Default([](Type) { return false; });
}

//===----------------------------------------------------------------------===//
// Value Pretty Printing
//===----------------------------------------------------------------------===//

void csl::printValue(llvm::raw_ostream &os, TypedAttr attr,
                     const NominalTypeRegistry *registry) {
  llvm::TypeSwitch<Attribute>(attr)
      // Boolean - special case of IntegerAttr with i1 type
      .Case<BoolAttr>([&](BoolAttr boolAttr) {
        os << (boolAttr.getValue() ? "true" : "false");
      })
      // Runtime integers
      .Case<IntegerAttr>([&](IntegerAttr intAttr) {
        Type type = intAttr.getType();
        if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
          if (intType.getWidth() == 1) {
            // Bool
            os << (intAttr.getValue().getBoolValue() ? "true" : "false");
          } else if (intType.isSigned() || intType.isSignless()) {
            intAttr.getValue().print(os, /*isSigned=*/true);
          } else {
            intAttr.getValue().print(os, /*isSigned=*/false);
          }
        } else {
          intAttr.getValue().print(os, /*isSigned=*/true);
        }
      })
      // Runtime floats
      .Case<FloatAttr>([&](FloatAttr floatAttr) {
        llvm::SmallString<32> str;
        floatAttr.getValue().toString(str);
        if (str.find('.') == llvm::StringRef::npos)
          str.append(".0");
        os << str;
      })
      // Comptime integers
      .Case<ComptimeIntAttr>([&](ComptimeIntAttr comptimeInt) {
        comptimeInt.getValue().print(os, /*isSigned=*/true);
      })
      // Comptime floats
      .Case<ComptimeFloatAttr>([&](ComptimeFloatAttr comptimeFloat) {
        llvm::SmallString<32> str;
        comptimeFloat.getValue().toString(str);
        if (str.find('.') == llvm::StringRef::npos)
          str.append(".0");
        os << str;
      })
      // Comptime strings
      .Case<ComptimeStringAttr>([&](ComptimeStringAttr comptimeStr) {
        os << "\"";
        csl::printEscapedString(comptimeStr.getValue(), os);
        os << "\"";
      })
      // Type values
      .Case<csl::TypeAttr>([&](csl::TypeAttr typeAttr) {
        printType(os, typeAttr.getValue(), registry);
      })
      // Void
      .Case<VoidAttr>([&](VoidAttr) { os << "{}"; })
      // Enum values
      .Case<EnumValueAttr>([&](EnumValueAttr enumAttr) {
        // Print as TypeName.element (CSL doesn't support standalone .element)
        if (registry) {
          if (auto name = registry->getName(enumAttr.getEnumType())) {
            os << *name << "." << enumAttr.getElementName().getValue();
            return;
          }
        }
        // Fallback: print full enum type + element
        printType(os, enumAttr.getEnumType(), registry);
        os << "." << enumAttr.getElementName().getValue();
      })
      // Struct values
      .Case<StructValueAttr>([&](StructValueAttr structAttr) {
        // For named structs, print TypeName { ... } instead of .{ ... }
        auto structType = mlir::dyn_cast<StructType>(structAttr.getType());
        if (registry && structType && structType.getDefLocation()) {
          if (auto name = registry->getName(structType)) {
            os << *name << " ";
          }
        } else {
          os << ".";
        }
        os << (structAttr.getNumFields() == 0 ? "{" : "{ ");
        for (size_t i = 0; i < structAttr.getNumFields(); ++i) {
          if (i > 0)
            os << ", ";
          auto fieldName = structAttr.getFieldName(i);
          if (succeeded(fieldName)) {
            os << "." << *fieldName << " = ";
          }
          printValue(os, structAttr.getField(i), registry);
        }
        os << (structAttr.getNumFields() == 0 ? "}" : " }");
      })
      // Array values
      .Case<ArrayValueAttr>([&](ArrayValueAttr arrayAttr) {
        ArrayRef<TypedAttr> elements = arrayAttr.getElements();

        // Print type followed by flattened values in CSL format: [dims]Type{v1,
        // v2, ...}
        printType(os, arrayAttr.getType(), registry);
        os << "{";
        for (size_t i = 0; i < elements.size(); ++i) {
          if (i > 0)
            os << ", ";
          printValue(os, elements[i], registry);
        }
        os << "}";
      })
      // LValue
      .Case<LValueAttr>([&](LValueAttr lvalueAttr) {
        printValue(os, lvalueAttr.getRvalue(), registry);
      })
      // Range values
      .Case<RangeAttr>([&](RangeAttr rangeAttr) {
        os << "@range(";
        printType(os, rangeAttr.getElementType(), registry);
        os << ", " << rangeAttr.getStart() << ", " << rangeAttr.getStop()
           << ", " << rangeAttr.getStep() << ")";
      })
      // Color, TaskId, and Queue values
      .Case<ColorAttr>([&](ColorAttr colorAttr) {
        os << "@get_color(" << colorAttr.getValue() << ")";
      })
      .Case<LocalTaskIdAttr>([&](LocalTaskIdAttr taskIdAttr) {
        os << "@get_local_task_id(" << taskIdAttr.getValue() << ")";
      })
      .Case<DataTaskIdAttr>([&](DataTaskIdAttr taskIdAttr) {
        // DataTaskId is created from a color (WSE-2) or input_queue (WSE-3)
        os << "@get_data_task_id(";
        printValue(os, taskIdAttr.getSource(), registry);
        os << ")";
      })
      .Case<ControlTaskIdAttr>([&](ControlTaskIdAttr taskIdAttr) {
        os << "@get_control_task_id(" << taskIdAttr.getValue() << ")";
      })
      .Case<InputQueueAttr>([&](InputQueueAttr queueAttr) {
        os << "@get_input_queue(" << static_cast<int>(queueAttr.getValue())
           << ")";
      })
      .Case<OutputQueueAttr>([&](OutputQueueAttr queueAttr) {
        os << "@get_output_queue(" << static_cast<int>(queueAttr.getValue())
           << ")";
      })
      // Direction values
      .Case<DirectionAttr>([&](DirectionAttr dirAttr) {
        os << stringifyDirection(dirAttr.getValue());
      })
      // DSR values
      .Case<DsrDestAttr>([&](DsrDestAttr dsrAttr) {
        os << "@get_dsr(dsr_dest, " << static_cast<int>(dsrAttr.getValue())
           << ")";
      })
      .Case<DsrSrc0Attr>([&](DsrSrc0Attr dsrAttr) {
        os << "@get_dsr(dsr_src0, " << static_cast<int>(dsrAttr.getValue())
           << ")";
      })
      .Case<DsrSrc1Attr>([&](DsrSrc1Attr dsrAttr) {
        os << "@get_dsr(dsr_src1, " << static_cast<int>(dsrAttr.getValue())
           << ")";
      })
      .Case<DsrFifoDestAttr>([&](DsrFifoDestAttr dsrAttr) {
        os << "@get_dsr(dsr_fifo_dest, " << static_cast<int>(dsrAttr.getValue())
           << ")";
      })
      .Case<DsrFifoSrc1Attr>([&](DsrFifoSrc1Attr dsrAttr) {
        os << "@get_dsr(dsr_fifo_src1, " << static_cast<int>(dsrAttr.getValue())
           << ")";
      })
      .Case<XdsrAttr>([&](XdsrAttr xdsrAttr) {
        os << "@get_xdsr(" << static_cast<int>(xdsrAttr.getValue()) << ")";
      })
      .Case<SrAttr>([&](SrAttr srAttr) {
        os << "@get_sr(" << static_cast<int>(srAttr.getValue()) << ")";
      })
      // DSD values
      .Case<Mem1dDsdAttr>([&](Mem1dDsdAttr dsdAttr) {
        // Use top-level properties (not .tensor_access wrapper) so that
        // stride/extent are scalars, matching cslc's expected syntax.
        os << "@get_dsd(mem1d_dsd, .{ .base_address = ";
        printValue(os, dsdAttr.getBaseAddress(), registry);
        os << ", .offset = " << dsdAttr.getOffset();
        os << ", .stride = " << static_cast<int>(dsdAttr.getStride());
        os << ", .extent = " << dsdAttr.getExtent();
        if (dsdAttr.getWaveletIndexOffset())
          os << ", .wavelet_index_offset = true";
        os << " })";
      })
      .Case<Mem4dDsdAttr>([&](Mem4dDsdAttr dsdAttr) {
        // Use top-level properties. Stride and extent are tuples for mem4d.
        os << "@get_dsd(mem4d_dsd, .{ .base_address = ";
        printValue(os, dsdAttr.getBaseAddress(), registry);
        os << ", .offset = " << dsdAttr.getOffset();
        os << ", .stride = .{ ";
        llvm::interleaveComma(dsdAttr.getStrides(), os,
                              [&](int16_t s) { os << s; });
        os << " }, .extent = .{ ";
        llvm::interleaveComma(dsdAttr.getExtents(), os,
                              [&](uint16_t e) { os << e; });
        os << " }";
        if (dsdAttr.getWaveletIndexOffset())
          os << ", .wavelet_index_offset = true";
        os << " })";
      })
      .Case<FabinDsdAttr>([&](FabinDsdAttr dsdAttr) {
        os << "@get_dsd(fabin_dsd, .{ .extent = " << dsdAttr.getExtent();
        if (dsdAttr.getFabricColor()) {
          os << ", .fabric_color = ";
          printValue(os, dsdAttr.getFabricColor(), registry);
        }
        if (dsdAttr.getInputQueue()) {
          os << ", .input_queue = ";
          printValue(os, dsdAttr.getInputQueue(), registry);
        }
        if (dsdAttr.getPriority()) {
          os << ", .priority = ";
          printValue(os, dsdAttr.getPriority(), registry);
        }
        if (dsdAttr.getControlTransform())
          os << ", .control_transform = true";
        if (dsdAttr.getSimdMode()) {
          os << ", .simd_mode = ";
          printValue(os, dsdAttr.getSimdMode(), registry);
        }
        os << " })";
      })
      .Case<FaboutDsdAttr>([&](FaboutDsdAttr dsdAttr) {
        os << "@get_dsd(fabout_dsd, .{ .extent = " << dsdAttr.getExtent();
        if (dsdAttr.getFabricColor()) {
          os << ", .fabric_color = ";
          printValue(os, dsdAttr.getFabricColor(), registry);
        }
        if (dsdAttr.getOutputQueue()) {
          os << ", .output_queue = ";
          printValue(os, dsdAttr.getOutputQueue(), registry);
        }
        if (dsdAttr.getControl())
          os << ", .control = true";
        if (dsdAttr.getWaveletIndexOffset())
          os << ", .wavelet_index_offset = true";
        if (dsdAttr.getAdvanceSwitch())
          os << ", .advance_switch = true";
        if (dsdAttr.getControlTransform())
          os << ", .control_transform = true";
        if (dsdAttr.getSimdMode()) {
          os << ", .simd_mode = ";
          printValue(os, dsdAttr.getSimdMode(), registry);
        }
        if (dsdAttr.getZero()) {
          os << ", .zero = ";
          printValue(os, dsdAttr.getZero(), registry);
        }
        os << " })";
      })
      .Case<FifoDsdAttr>([&](FifoDsdAttr dsdAttr) {
        os << "@allocate_fifo("
           << dsdAttr.getBuffer().getSymbolName().getValue() << ", .{";
        if (dsdAttr.getActivatePop()) {
          os << " .activate_pop = ";
          printValue(os, dsdAttr.getActivatePop(), registry);
        }
        if (dsdAttr.getActivatePush()) {
          os << " .activate_push = ";
          printValue(os, dsdAttr.getActivatePush(), registry);
        }
        if (dsdAttr.getPriority()) {
          os << " .priority = ";
          printValue(os, dsdAttr.getPriority(), registry);
        }
        if (dsdAttr.getSimdMode()) {
          os << " .simd_mode = ";
          printValue(os, dsdAttr.getSimdMode(), registry);
        }
        os << " })";
      })
      .Case<SimdModeAttr>([&](SimdModeAttr simdAttr) {
        switch (simdAttr.getMode()) {
        case SimdModeKind::Simd32:
          os << ".{ .simd_32 = true }";
          break;
        case SimdModeKind::Simd64:
          os << ".{ .simd_64 = true }";
          break;
        case SimdModeKind::Simd32Or64:
          os << ".{ .simd_32_or_64 = true }";
          break;
        case SimdModeKind::Simd16:
          os << ".{ .simd_16 = true }";
          break;
        case SimdModeKind::SimdOff:
          os << ".{ .simd_off = true }";
          break;
        case SimdModeKind::SimdMax:
          os << ".{ .simd_max = true }";
          break;
        }
      })
      .Case<PriorityAttr>([&](PriorityAttr priorityAttr) {
        switch (priorityAttr.getLevel()) {
        case PriorityLevel::Low:
          os << ".{ .low = true }";
          break;
        case PriorityLevel::Medium:
          os << ".{ .medium = true }";
          break;
        case PriorityLevel::High:
          os << ".{ .high = true }";
          break;
        }
      })
      .Case<ZeroAttr>([&](ZeroAttr zeroAttr) {
        switch (zeroAttr.getSource()) {
        case ZeroSource::FirstSource:
          os << ".{ .first_source = true }";
          break;
        case ZeroSource::SecondSource:
          os << ".{ .second_source = true }";
          break;
        }
      })
      .Case<UtIdAttr>([&](UtIdAttr mtidAttr) {
        os << "@get_ut_id(" << mtidAttr.getValue() << ")";
      })
      // GlobalAddr - emit as &symbol_name
      .Case<GlobalAddrAttr>([&](GlobalAddrAttr addrAttr) {
        os << "&" << addrAttr.getGlobalRef().getSymbolName().getValue();
      })
      // FuncAddr - emit as &symbol_name
      .Case<FuncAddrAttr>([&](FuncAddrAttr addrAttr) {
        os << "&" << addrAttr.getFuncRef().getSymbolName().getValue();
      })
      // FuncRef - emit as symbol_name
      .Case<FuncRefAttr>([&](FuncRefAttr refAttr) {
        os << refAttr.getSymbolName().getValue();
      })
      // GlobalRef - emit as symbol_name
      .Case<GlobalRefAttr>([&](GlobalRefAttr refAttr) {
        os << refAttr.getSymbolName().getValue();
      })
      .Default([&](Attribute a) {
        // Fallback to MLIR printing for unknown attributes
        a.print(os);
      });
}

//===----------------------------------------------------------------------===//
// String Conversion Helpers
//===----------------------------------------------------------------------===//

std::string csl::prettyPrintType(Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printType(os, type);
  return result;
}

std::string csl::prettyPrintValue(TypedAttr attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printValue(os, attr);
  return result;
}

//===----------------------------------------------------------------------===//
// String Escaping
//===----------------------------------------------------------------------===//

void csl::printEscapedString(llvm::StringRef str, llvm::raw_ostream &os) {
  for (unsigned char c : str) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\t':
      os << "\\t";
      break;
    case '\r':
      os << "\\r";
      break;
    default:
      os << c;
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Export Compatibility
//===----------------------------------------------------------------------===//

bool csl::isExportCompatibleType(Type type) {
  return llvm::TypeSwitch<Type, bool>(type)
      // Integer types: i8, i16, i32, i64, u8, u16, u32, u64, bool
      .Case<IntegerType>([](IntegerType intType) {
        unsigned width = intType.getWidth();
        // bool (i1), i8/u8, i16/u16, i32/u32, i64/u64 are export compatible
        return width == 1 || width == 8 || width == 16 || width == 32 ||
               width == 64;
      })
      // Float types: f16, f32, bf16, cb16 are export compatible
      .Case<Float16Type>([](Float16Type) { return true; })
      .Case<BFloat16Type>([](BFloat16Type) { return true; })
      .Case<Float32Type>([](Float32Type) { return true; })
      .Case<CB16Type>([](CB16Type) { return true; })
      // color is export compatible
      .Case<ColorType>([](ColorType) { return true; })
      // Array types: export compatible if element type is
      .Case<ArrayType>([](ArrayType arrayType) {
        return isExportCompatibleType(arrayType.getElementType());
      })
      // Pointer types: export compatible if pointee type is
      // For function pointers (*const fn), the function type must be export
      // compatible
      .Case<PointerType>([](PointerType ptrType) {
        Type pointee = ptrType.getPointeeType();
        // Function pointers must be const and the function type must be export
        // compatible
        if (auto funcType = mlir::dyn_cast<FunctionType>(pointee)) {
          // Function pointers must be const
          if (!ptrType.getIsConst())
            return false;
          return isExportCompatibleFunctionType(funcType);
        }
        // Regular pointers: pointee type must be export compatible
        return isExportCompatibleType(pointee);
      })
      // Enum types are export compatible
      .Case<EnumType>([](EnumType) { return true; })
      // Everything else is NOT export compatible:
      // - comptime_int, comptime_float, comptime_string, type
      // - struct types
      // - DSD and DSR types
      // - direction type
      // - range type
      // - void
      // - function types (not pointers)
      .Default([](Type) { return false; });
}

bool csl::isExportCompatibleFunctionType(FunctionType funcType) {
  // All argument types must be export compatible for variable declarations
  for (Type argType : funcType.getInputs()) {
    if (!isExportCompatibleType(argType))
      return false;
  }

  // Return type must be export compatible or void
  if (funcType.getNumResults() == 0)
    return true; // void return

  if (funcType.getNumResults() != 1)
    return false; // Multiple returns not supported

  Type returnType = funcType.getResult(0);
  if (mlir::isa<VoidType>(returnType))
    return true;

  return isExportCompatibleType(returnType);
}
