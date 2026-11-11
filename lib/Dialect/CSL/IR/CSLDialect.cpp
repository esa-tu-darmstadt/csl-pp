//===- CSLDialect.cpp - CSL Dialect ----------------------------*- C++ -*-===//
//
// Implementation of the CSL dialect.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::csl;

#define DEBUG_TYPE "csl-dialect"

#include "csl-mlir/Dialect/CSL/IR/CSLOpInterfaces.cpp.inc"
#include "csl-mlir/Dialect/CSL/IR/CSLTypeInterfaces.cpp.inc"

namespace {
ParseResult parseMutability(AsmParser &parser, bool &isConst) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (!parser.parseOptionalKeyword("const")) {
    isConst = true;
    return success();
  } else if (!parser.parseOptionalKeyword("mutable")) {
    isConst = false;
    return success();
  } else {
    parser.emitError(loc, "expected 'const' or 'mutable'");
    return failure();
  }
}
void printMutability(AsmPrinter &printer, bool isConst) {
  printer << (isConst ? "const" : "mutable");
}
} // namespace

//===----------------------------------------------------------------------===//
// CSL Dialect
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSLDialect.cpp.inc"

// Include enum definitions
#include "csl-mlir/Dialect/CSL/IR/CSLEnums.cpp.inc"

void CSLDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "csl-mlir/Dialect/CSL/IR/CSLOps.cpp.inc"
      >();
  registerBuiltinDescriptors();
}

void CSLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "csl-mlir/Dialect/CSL/IR/CSLTypes.cpp.inc"
      >();
}

void CSLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "csl-mlir/Dialect/CSL/IR/CSLAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom Format Directives
//===----------------------------------------------------------------------===//

namespace mlir {
namespace csl {

/// Parse array shape: "10x20x30" -> [10, 20, 30]
static ParseResult parseArrayShape(AsmParser &parser,
                                   SmallVectorImpl<int64_t> &shape) {
  do {
    int64_t dim;
    if (parser.parseInteger(dim))
      return failure();
    shape.push_back(dim);

    // Check for 'x' separator
    if (failed(parser.parseOptionalKeyword("x")))
      break;
  } while (true);

  return success();
}

/// Print array shape: [10, 20, 30] -> "10x20x30"
static void printArrayShape(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  bool first = true;
  for (int64_t dim : shape) {
    if (!first)
      printer << "x";
    printer << dim;
    first = false;
  }
}

} // namespace csl
} // namespace mlir

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "csl-mlir/Dialect/CSL/IR/CSLTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// CB16Type (Cerebras Float 16) - FloatTypeInterface
//===----------------------------------------------------------------------===//

// CB16: 1 sign, 6 exponent (bias 42), 9 mantissa bits
// maxExponent = 20  (stored exp 62 → 62-42 = 20)
// minExponent = -41 (stored exp 1  → 1-42  = -41)
// precision   = 10  (9 mantissa bits + 1 implicit bit)
// sizeInBits  = 16
static const llvm::fltSemantics semCB16 = {20, -41, 10, 16};

const llvm::fltSemantics &csl::CB16Type::getFloatSemantics() const {
  return semCB16;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CoroFrameType
//===----------------------------------------------------------------------===//

// Format: !csl.coroframe<@symbol> or !csl.coroframe<123, @symbol>
Type csl::CoroFrameType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Try to parse an optional module identifier (integer).
  std::optional<uint32_t> moduleId;
  uint32_t id;
  auto intResult = parser.parseOptionalInteger(id);
  if (intResult.has_value()) {
    if (intResult.value()) // parse error
      return Type();
    moduleId = id;
    if (parser.parseComma())
      return Type();
  }

  // Parse symbol name.
  FlatSymbolRefAttr symbolName;
  if (parser.parseAttribute(symbolName))
    return Type();

  if (parser.parseGreater())
    return Type();

  return get(parser.getContext(), moduleId, symbolName);
}

void csl::CoroFrameType::print(AsmPrinter &printer) const {
  printer << "<";
  if (getModuleIdentifier())
    printer << *getModuleIdentifier() << ", ";
  printer.printAttribute(getSymbolName());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// FutureType
//===----------------------------------------------------------------------===//

Type csl::FutureType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  std::optional<uint32_t> moduleId;
  uint32_t id;
  auto intResult = parser.parseOptionalInteger(id);
  if (intResult.has_value()) {
    if (intResult.value())
      return Type();
    moduleId = id;
    if (parser.parseComma())
      return Type();
  }

  FlatSymbolRefAttr symbolName;
  if (parser.parseAttribute(symbolName))
    return Type();

  // Optional result type: , result_type
  Type resultType;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(resultType))
      return Type();
  }

  if (parser.parseGreater())
    return Type();

  return get(parser.getContext(), moduleId, symbolName, resultType);
}

void csl::FutureType::print(AsmPrinter &printer) const {
  printer << "<";
  if (getModuleIdentifier())
    printer << *getModuleIdentifier() << ", ";
  printer.printAttribute(getSymbolName());
  if (getResultType())
    printer << ", " << getResultType();
  printer << ">";
}

//===----------------------------------------------------------------------===//
// PerPEArrayType
//===----------------------------------------------------------------------===//

unsigned csl::PerPEArrayType::getDimDataOffset(unsigned dimIdx) const {
  assert(dimIdx < getRank() && "dimension index out of range");
  unsigned offset = 0;
  auto isPerPE = getDimIsPerPe();
  for (unsigned i = 0; i < dimIdx; ++i) {
    offset += isPerPE[i] ? (getDataWidth() * getDataHeight()) : 1;
  }
  return offset;
}

int64_t csl::PerPEArrayType::getStaticDimSize(unsigned dimIdx) const {
  assert(!isDimPerPE(dimIdx) && "dimension is per-PE, not static");
  return getDimData()[getDimDataOffset(dimIdx)];
}

ArrayRef<int64_t> csl::PerPEArrayType::getPerPEDimSizes(unsigned dimIdx) const {
  assert(isDimPerPE(dimIdx) && "dimension is static, not per-PE");
  unsigned offset = getDimDataOffset(dimIdx);
  int64_t count = getDataWidth() * getDataHeight();
  return getDimData().slice(offset, count);
}

int64_t csl::PerPEArrayType::getDimSizeForPE(unsigned dimIdx, int64_t x,
                                             int64_t y) const {
  if (!isDimPerPE(dimIdx))
    return getStaticDimSize(dimIdx);
  // Out-of-bounds PEs get 0
  if (x >= getDataWidth() || y >= getDataHeight())
    return 0;
  // Same layout as [W, H] ArrayValueAttr: values[x * H + y]
  auto sizes = getPerPEDimSizes(dimIdx);
  return sizes[x * getDataHeight() + y];
}

SmallVector<int64_t> csl::PerPEArrayType::getShapeForPE(int64_t x,
                                                        int64_t y) const {
  SmallVector<int64_t> shape;
  for (unsigned i = 0; i < getRank(); ++i)
    shape.push_back(getDimSizeForPE(i, x, y));
  return shape;
}

ArrayType csl::PerPEArrayType::getArrayTypeForPE(int64_t x, int64_t y) const {
  return ArrayType::get(getShapeForPE(x, y), getElementType());
}

int64_t csl::PerPEArrayType::getNumElementsForPE(int64_t x, int64_t y) const {
  int64_t num = 1;
  for (unsigned i = 0; i < getRank(); ++i)
    num *= getDimSizeForPE(i, x, y);
  return num;
}

std::optional<uint32_t> csl::PerPEArrayType::getSizeInBytes() const {
  return std::nullopt; // No single static size
}

bool csl::PerPEArrayType::isRuntimeType() const {
  return csl::isRuntimeType(getElementType());
}

bool csl::PerPEArrayType::hasPerPELayout() const { return true; }

std::optional<uint32_t>
csl::PerPEArrayType::getSizeInBytesForPE(int64_t x, int64_t y) const {
  auto elemSize = csl::getSizeInBytes(getElementType());
  if (!elemSize)
    return std::nullopt;
  return *elemSize * static_cast<uint32_t>(getNumElementsForPE(x, y));
}

// PerPEArrayType custom parse/print
// Format: !csl.per_pe_array<grid WxH, [offset OxP,] dim1 x dim2 x ... x type>
// where each dim is either an integer (static) or pp<v1,v2,...> (per-PE)
Type csl::PerPEArrayType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse "grid WxH"
  if (parser.parseKeyword("grid"))
    return Type();

  int64_t dataWidth, dataHeight;
  if (parser.parseInteger(dataWidth) || parser.parseKeyword("x") ||
      parser.parseInteger(dataHeight))
    return Type();

  // Parse optional "offset OxP"
  int64_t offsetX = 0, offsetY = 0;
  if (!parser.parseOptionalComma()) {
    if (!parser.parseOptionalKeyword("offset")) {
      if (parser.parseInteger(offsetX) || parser.parseKeyword("x") ||
          parser.parseInteger(offsetY))
        return Type();
      if (parser.parseComma())
        return Type();
    } else {
      // Wasn't "offset", backtrack — the comma was for the first dimension
      // We need to parse dimensions starting from here
      // Actually, since we consumed the comma, we need to parse the first
      // dimension now
      goto parse_dims;
    }
  } else {
    // Need comma after grid spec
    parser.emitError(parser.getCurrentLocation(),
                     "expected ',' after grid dimensions");
    return Type();
  }

parse_dims:
  // Parse dimensions: each is either an integer or pp<v1,v2,...>
  // separated by 'x', ending with 'x type'
  SmallVector<bool> dimIsPerPE;
  SmallVector<int64_t> dimData;

  while (true) {
    // Try to parse 'pp<...>'
    if (!parser.parseOptionalKeyword("pp")) {
      dimIsPerPE.push_back(true);
      if (parser.parseLess())
        return Type();
      int64_t val;
      if (parser.parseInteger(val))
        return Type();
      dimData.push_back(val);
      while (!parser.parseOptionalComma()) {
        if (parser.parseInteger(val))
          return Type();
        dimData.push_back(val);
      }
      if (parser.parseGreater())
        return Type();
    } else {
      // Try to parse an integer dimension
      int64_t dim;
      OptionalParseResult intResult = parser.parseOptionalInteger(dim);
      if (!intResult.has_value()) {
        // Not an integer and not 'pp' — must be the element type
        break;
      }
      if (failed(*intResult))
        return Type();
      dimIsPerPE.push_back(false);
      dimData.push_back(dim);
    }

    // Expect 'x' separator between dimensions and before type
    if (parser.parseKeyword("x"))
      return Type();
  }

  if (dimIsPerPE.empty()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected at least one dimension");
    return Type();
  }

  // Parse element type
  Type elemType;
  if (parser.parseType(elemType))
    return Type();

  if (parser.parseGreater())
    return Type();

  return get(parser.getContext(), dataWidth, dataHeight, offsetX, offsetY,
             dimIsPerPE, dimData, elemType);
}

void csl::PerPEArrayType::print(AsmPrinter &printer) const {
  printer << "<grid " << getDataWidth() << "x" << getDataHeight();

  if (getOffsetX() != 0 || getOffsetY() != 0)
    printer << ", offset " << getOffsetX() << "x" << getOffsetY();

  printer << ", ";

  for (unsigned i = 0; i < getRank(); ++i) {
    if (isDimPerPE(i)) {
      printer << "pp<";
      auto sizes = getPerPEDimSizes(i);
      llvm::interleaveComma(sizes, printer);
      printer << ">";
    } else {
      printer << getStaticDimSize(i);
    }
    printer << " x ";
  }

  printer << getElementType() << ">";
}

//===----------------------------------------------------------------------===//
// ArrayType interface implementations
//===----------------------------------------------------------------------===//

bool csl::ArrayType::isRuntimeType() const {
  return csl::isRuntimeType(getElementType());
}

bool csl::ArrayType::hasPerPELayout() const {
  return csl::hasPerPELayout(getElementType());
}

std::optional<uint32_t> csl::ArrayType::getSizeInBytesForPE(int64_t x,
                                                            int64_t y) const {
  auto elemSize = csl::getSizeInBytesForPE(getElementType(), x, y);
  if (!elemSize)
    return std::nullopt;
  return *elemSize * static_cast<uint32_t>(getNumElements());
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

int64_t csl::ArrayType::getNumElements() const {
  int64_t num = 1;
  for (int64_t dim : getShape())
    num *= dim;
  return num;
}

// StructType helper methods
Type csl::StructType::getFieldType(StringRef name) const {
  auto names = getFieldNames();
  if (names.empty())
    return Type(); // Tuple - no field names

  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name)
      return getFieldTypes()[i];
  }
  return Type(); // Not found
}

FailureOr<size_t> csl::StructType::getFieldIndex(StringRef name) const {
  auto names = getFieldNames();
  if (names.empty())
    return failure(); // Tuple - no field names

  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name)
      return i;
  }
  return failure(); // Not found
}

bool csl::StructType::isRuntimeType() const {
  // A void field takes no memory and is removed from runtime structs, so it
  // does not make the struct comptime-only.
  return llvm::all_of(getFieldTypes(), [](Type fieldType) {
    return mlir::isa<VoidType>(fieldType) || csl::isRuntimeType(fieldType);
  });
}

bool csl::StructType::hasPerPELayout() const {
  for (Type ft : getFieldTypes()) {
    if (csl::hasPerPELayout(ft))
      return true;
  }
  return false;
}

std::optional<uint32_t> csl::StructType::getSizeInBytesForPE(int64_t x,
                                                             int64_t y) const {
  uint32_t total = 0;
  uint32_t maxAlign = 1;
  for (Type ft : getFieldTypes()) {
    // A void field takes no memory.
    if (mlir::isa<VoidType>(ft))
      continue;
    auto fieldSize = csl::getSizeInBytesForPE(ft, x, y);
    if (!fieldSize)
      return std::nullopt;
    // Align the current offset to the field's natural alignment.
    uint32_t align = csl::getAlignmentForType(ft);
    total = (total + align - 1) & ~(align - 1);
    total += *fieldSize;
    if (align > maxAlign)
      maxAlign = align;
  }
  // Pad to struct alignment so arrays of structs stay aligned.
  total = (total + maxAlign - 1) & ~(maxAlign - 1);
  return total;
}

// StructType custom print/parse
Type csl::StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  LocationAttr defLocation;
  Type firstType;

  Attribute attr;
  if (parser.parseAttribute(attr))
    return Type();

  if (auto locAttr = mlir::dyn_cast<LocationAttr>(attr)) {
    // Got location - expect comma, then parse first field type
    defLocation = locAttr;
    if (parser.parseComma() || parser.parseType(firstType))
      return Type();
  } else if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    // Got type wrapped as attribute - this is the first field type
    firstType = typeAttr.getValue();
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected location attribute or field type");
    return Type();
  }

  // Parse fields with syntax: type:name (named) or just type (tuple)
  SmallVector<Type> fieldTypes;
  SmallVector<StringAttr> fieldNames;
  fieldTypes.push_back(firstType);

  // Check if first field has a name - this determines the pattern for all
  // fields
  bool isNamedFields = parser.parseOptionalColon().succeeded();
  if (isNamedFields) {
    StringRef firstName;
    if (parser.parseKeyword(&firstName))
      return Type();
    fieldNames.push_back(StringAttr::get(parser.getContext(), firstName));
  }

  // Parse remaining fields - must follow same pattern
  while (parser.parseOptionalComma().succeeded()) {
    Type type;
    if (parser.parseType(type))
      return Type();
    fieldTypes.push_back(type);

    if (isNamedFields) {
      // Expect :name for this field
      StringRef name;
      if (parser.parseColon() || parser.parseKeyword(&name))
        return Type();
      fieldNames.push_back(StringAttr::get(parser.getContext(), name));
    }
  }

  if (parser.parseGreater())
    return Type();

  return get(parser.getContext(),
             isNamedFields ? fieldNames : ArrayRef<StringAttr>(), fieldTypes,
             defLocation);
}

void csl::StructType::print(AsmPrinter &printer) const {
  printer << "<";

  // Print nominal location if present (for named structs)
  if (getDefLocation()) {
    printer.printAttribute(getDefLocation());
    printer << ",";
  }

  // Print fields
  auto fieldTypes = getFieldTypes();
  auto fieldNames = getFieldNames();

  if (!fieldNames.empty()) {
    // Named fields: type:name format
    for (size_t i = 0; i < fieldTypes.size(); ++i) {
      if (i > 0)
        printer << ",";
      printer << fieldTypes[i] << ":" << fieldNames[i].getValue();
    }
  } else {
    // Tuple: just types
    llvm::interleaveComma(fieldTypes, printer);
  }

  printer << ">";
}

// EnumType helper methods
FailureOr<int64_t> csl::EnumType::getElementValue(StringRef name) const {
  auto names = getElementNames();
  auto values = getElementValues();

  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name)
      return values[i];
  }
  return failure(); // Not found
}

FailureOr<StringRef> csl::EnumType::getElementName(int64_t value) const {
  auto names = getElementNames();
  auto values = getElementValues();

  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] == value)
      return names[i].getValue();
  }
  return failure(); // Not found
}

bool csl::EnumType::isValidValue(int64_t value) const {
  return succeeded(getElementName(value));
}

FailureOr<size_t> csl::EnumType::getElementIndex(StringRef name) const {
  auto names = getElementNames();

  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name)
      return i;
  }
  return failure(); // Not found
}

// EnumType custom print/parse
// Format: !csl.enum<loc(...), si16, "red":0, "white":1, "blue":2>
Type csl::EnumType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse location (required for nominal typing)
  LocationAttr defLocation;
  if (parser.parseAttribute(defLocation))
    return Type();

  if (parser.parseComma())
    return Type();

  // Parse underlying integer type
  IntegerType underlyingType;
  if (parser.parseType(underlyingType))
    return Type();

  // Parse elements: "name":value pairs
  SmallVector<StringAttr> elementNames;
  SmallVector<int64_t> elementValues;

  while (parser.parseOptionalComma().succeeded()) {
    std::string name;
    if (parser.parseString(&name))
      return Type();

    if (parser.parseColon())
      return Type();

    int64_t value;
    if (parser.parseInteger(value))
      return Type();

    elementNames.push_back(StringAttr::get(parser.getContext(), name));
    elementValues.push_back(value);
  }

  if (parser.parseGreater())
    return Type();

  return get(parser.getContext(), underlyingType, elementNames, elementValues,
             defLocation);
}

void csl::EnumType::print(AsmPrinter &printer) const {
  printer << "<";

  // Print location for nominal typing
  printer.printAttribute(getDefLocation());
  printer << ", ";

  // Print underlying type
  printer << getUnderlyingType();

  // Print elements
  auto names = getElementNames();
  auto values = getElementValues();
  for (size_t i = 0; i < names.size(); ++i) {
    printer << ", \"" << names[i].getValue() << "\":" << values[i];
  }

  printer << ">";
}

//===----------------------------------------------------------------------===//
// Attribute Definitions
//===----------------------------------------------------------------------===//

// StructValueAttr helper methods
TypedAttr csl::StructValueAttr::getField(StringRef name) const {
  FailureOr<size_t> idx = getFieldIndex(name);
  if (failed(idx))
    return TypedAttr();
  return getFieldValues()[*idx];
}

Type csl::StructValueAttr::getType() {
  llvm::SmallVector<Type> fieldTypes;
  for (TypedAttr attr : getFieldValues()) {
    fieldTypes.push_back(attr.getType());
  }

  return csl::StructType::get(getContext(), getFieldNames(), fieldTypes,
                              getDefLocation());
}

Type csl::PriorityAttr::getType() const {
  StringRef fieldName;
  switch (getLevel()) {
  case PriorityLevel::Low:
    fieldName = "low";
    break;
  case PriorityLevel::Medium:
    fieldName = "medium";
    break;
  case PriorityLevel::High:
    fieldName = "high";
    break;
  }
  return StructType::get(getContext(),
                         {StringAttr::get(getContext(), fieldName)},
                         {IntegerType::get(getContext(), 1)});
}

Type csl::SimdModeAttr::getType() const {
  StringRef fieldName;
  switch (getMode()) {
  case SimdModeKind::Simd32:
    fieldName = "simd_32";
    break;
  case SimdModeKind::Simd64:
    fieldName = "simd_64";
    break;
  case SimdModeKind::Simd32Or64:
    fieldName = "simd_32_or_64";
    break;
  case SimdModeKind::Simd16:
    fieldName = "simd_16";
    break;
  case SimdModeKind::SimdOff:
    fieldName = "simd_off";
    break;
  case SimdModeKind::SimdMax:
    fieldName = "simd_max";
    break;
  }
  return StructType::get(getContext(),
                         {StringAttr::get(getContext(), fieldName)},
                         {IntegerType::get(getContext(), 1)});
}

Type csl::ZeroAttr::getType() const {
  StringRef fieldName;
  switch (getSource()) {
  case ZeroSource::FirstSource:
    fieldName = "first_source";
    break;
  case ZeroSource::SecondSource:
    fieldName = "second_source";
    break;
  }
  return StructType::get(getContext(),
                         {StringAttr::get(getContext(), fieldName)},
                         {IntegerType::get(getContext(), 1)});
}

FailureOr<size_t> csl::StructValueAttr::getFieldIndex(StringRef name) const {
  auto names = getFieldNames();
  if (names.empty())
    return failure(); // Tuple - no field names

  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name)
      return i;
  }
  return failure(); // Not found
}
csl::StructValueAttr csl::StructValueAttr::updateField(StructValueAttr base,
                                                       StringRef fieldName,
                                                       TypedAttr newValue) {
  FailureOr<size_t> idx = base.getFieldIndex(fieldName);
  assert(succeeded(idx) && "field not found");
  return updateField(base, *idx, newValue);
}

csl::StructValueAttr csl::StructValueAttr::updateField(StructValueAttr base,
                                                       size_t index,
                                                       TypedAttr newValue) {
  assert(index < base.getNumFields() && "index out of range");

  SmallVector<TypedAttr> newValues(base.getFieldValues());
  newValues[index] = newValue;

  return get(base.getContext(), newValues, base.getDefLocation(),
             base.getFieldNames());
}

// StructValueAttr custom print/parse
Attribute csl::StructValueAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();

  LocationAttr defLocation;
  TypedAttr firstValue;

  Attribute attr;
  if (parser.parseAttribute(attr))
    return Attribute();

  if (auto locAttr = mlir::dyn_cast<LocationAttr>(attr)) {
    // Got location - expect comma, then parse first field type
    defLocation = locAttr;
    if (parser.parseComma() || parser.parseAttribute(firstValue))
      return Attribute();
  } else if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attr)) {
    // Got type wrapped as attribute - this is the first field type
    firstValue = typedAttr;
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected location attribute or field value");
    return Attribute();
  }

  // Parse fields with syntax: type:name (named) or just type (tuple)
  SmallVector<TypedAttr> fieldValues;
  SmallVector<StringAttr> fieldNames;
  fieldValues.push_back(firstValue);

  // Check if first field has a name - this determines the pattern for all
  // fields
  bool isNamedFields = parser.parseOptionalColon().succeeded();
  if (isNamedFields) {
    StringRef firstName;
    if (parser.parseKeyword(&firstName))
      return Attribute();
    fieldNames.push_back(StringAttr::get(parser.getContext(), firstName));
  }

  // Parse remaining fields - must follow same pattern
  while (parser.parseOptionalComma().succeeded()) {
    TypedAttr value;
    if (parser.parseAttribute(value))
      return Attribute();
    fieldValues.push_back(value);

    if (isNamedFields) {
      // Expect :name for this field
      StringRef name;
      if (parser.parseColon() || parser.parseKeyword(&name))
        return Attribute();
      fieldNames.push_back(StringAttr::get(parser.getContext(), name));
    }
  }

  if (parser.parseGreater())
    return Attribute();

  return get(parser.getContext(), fieldValues, defLocation,
             isNamedFields ? fieldNames : ArrayRef<StringAttr>());
}

void csl::StructValueAttr::print(AsmPrinter &printer) const {
  printer << "<";

  // Print nominal location if present (for named structs)
  if (getDefLocation()) {
    printer.printAttribute(getDefLocation());
    printer << ",";
  }

  // Print fields
  auto fieldValues = getFieldValues();
  auto fieldNames = getFieldNames();

  if (!fieldNames.empty()) {
    // Named fields: value:name format
    for (size_t i = 0; i < fieldValues.size(); ++i) {
      if (i > 0)
        printer << ",";
      printer << fieldValues[i] << ":" << fieldNames[i].getValue();
    }
  } else {
    // Tuple: just types
    llvm::interleaveComma(fieldValues, printer);
  }

  printer << ">";
}

// ArrayValueAttr helper methods
ArrayType csl::ArrayValueAttr::getType() const {
  return ArrayType::get(getContext(), getShape(), getElementType());
}

int64_t csl::ArrayValueAttr::getNumElements() const {
  int64_t num = 1;
  for (int64_t dim : getShape())
    num *= dim;
  return num;
}

TypedAttr csl::ArrayValueAttr::getElement(int64_t linearIndex) const {
  assert(linearIndex >= 0 && linearIndex < getNumElements() &&
         "linear index out of bounds");
  return getElements()[linearIndex];
}

int64_t csl::ArrayValueAttr::linearizeIndices(ArrayRef<int64_t> indices) const {
  ArrayRef<int64_t> shape = getShape();
  assert(indices.size() == shape.size() && "index rank mismatch");

  int64_t linearIndex = 0;
  int64_t stride = 1;
  // Row-major: rightmost dimension varies fastest
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    assert(indices[i] >= 0 && indices[i] < shape[i] && "index out of bounds");
    linearIndex += indices[i] * stride;
    stride *= shape[i];
  }
  return linearIndex;
}

TypedAttr csl::ArrayValueAttr::getElement(ArrayRef<int64_t> indices) const {
  return getElement(linearizeIndices(indices));
}

csl::ArrayValueAttr csl::ArrayValueAttr::updateElement(ArrayValueAttr base,
                                                       int64_t linearIndex,
                                                       TypedAttr newValue) {
  assert(linearIndex >= 0 && linearIndex < base.getNumElements() &&
         "index out of bounds");

  SmallVector<TypedAttr> newElements(base.getElements());
  newElements[linearIndex] = newValue;

  return get(base.getContext(), base.getShape(), base.getElementType(),
             newElements);
}

// ArrayValueAttr custom print/parse
// Format: #csl.array_value<[dim1, dim2, ...] x elemType, [elem1, elem2, ...]>
Attribute csl::ArrayValueAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();

  // Parse shape: [dim1, dim2, ...]
  if (parser.parseLSquare())
    return Attribute();

  SmallVector<int64_t> shape;
  int64_t dim;
  if (parser.parseInteger(dim))
    return Attribute();
  shape.push_back(dim);

  while (parser.parseOptionalComma().succeeded()) {
    if (parser.parseInteger(dim))
      return Attribute();
    shape.push_back(dim);
  }

  if (parser.parseRSquare())
    return Attribute();

  // Parse 'x' and element type
  if (parser.parseKeyword("x"))
    return Attribute();

  Type elemType;
  if (parser.parseType(elemType))
    return Attribute();

  // Parse comma and element list
  if (parser.parseComma() || parser.parseLSquare())
    return Attribute();

  SmallVector<TypedAttr> elements;
  if (parser.parseOptionalRSquare().failed()) {
    // Non-empty list
    TypedAttr elem;
    if (parser.parseAttribute(elem))
      return Attribute();
    elements.push_back(elem);

    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseAttribute(elem))
        return Attribute();
      elements.push_back(elem);
    }

    if (parser.parseRSquare())
      return Attribute();
  }

  if (parser.parseGreater())
    return Attribute();

  return get(parser.getContext(), shape, elemType, elements);
}

void csl::ArrayValueAttr::print(AsmPrinter &printer) const {
  printer << "<[";

  // Print shape
  llvm::interleaveComma(getShape(), printer);
  printer << "] x " << getElementType() << ", [";

  // Print elements with nested brackets for multi-dimensional arrays
  ArrayRef<int64_t> shape = getShape();
  ArrayRef<TypedAttr> elements = getElements();

  if (shape.size() == 1) {
    // 1D array: simple flat list
    llvm::interleaveComma(elements, printer);
  } else if (shape.size() == 2) {
    // 2D array: nested brackets [[row1], [row2], ...]
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    for (int64_t row = 0; row < rows; ++row) {
      if (row > 0)
        printer << ", ";
      printer << "[";
      for (int64_t col = 0; col < cols; ++col) {
        if (col > 0)
          printer << ", ";
        printer << elements[row * cols + col];
      }
      printer << "]";
    }
  } else {
    // Higher dimensions: flat list with comment
    llvm::interleaveComma(elements, printer);
  }

  printer << "]>";
}

// EnumValueAttr custom print/parse
// Format: #csl.enum_value<"element_name", int_value : !csl.enum<...>>
Attribute csl::EnumValueAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();

  // Parse element name
  std::string elementName;
  if (parser.parseString(&elementName))
    return Attribute();

  if (parser.parseComma())
    return Attribute();

  // Parse integer value
  int64_t intValue;
  if (parser.parseInteger(intValue))
    return Attribute();

  if (parser.parseColon())
    return Attribute();

  // Parse enum type
  EnumType enumType;
  if (parser.parseType(enumType))
    return Attribute();

  if (parser.parseGreater())
    return Attribute();

  return get(parser.getContext(), enumType,
             StringAttr::get(parser.getContext(), elementName), intValue);
}

void csl::EnumValueAttr::print(AsmPrinter &printer) const {
  printer << "<\"" << getElementName().getValue() << "\", " << getIntValue()
          << " : " << getEnumType() << ">";
}

//===----------------------------------------------------------------------===//
// Attribute Definitions (TableGen)
//===----------------------------------------------------------------------===//

/// Parses a double (for ComptimeFloatAttr)
namespace mlir {
template <>
struct FieldParser<llvm::APFloat> {
  static FailureOr<llvm::APFloat> parse(AsmParser &parser) {
    double value;
    if (parser.parseFloat(value))
      return failure();
    return llvm::APFloat(value);
  }
};
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "csl-mlir/Dialect/CSL/IR/CSLAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute Custom Parsers/Printers
//===----------------------------------------------------------------------===//

// Type LValueType::parse(AsmParser &parser) {

//   // Parse literal '<'
//   if (parser.parseLess())
//     return {};

//   // Parse mutability
//   bool isConst;
//   if (!parser.parseOptionalKeyword("const")) {
//     isConst = true;
//   } else if (!parser.parseOptionalKeyword("mutable")) {
//     isConst = false;
//   } else {
//     parser.emitError(parser.getCurrentLocation(),
//                      "expected 'const' or 'mutable'");
//     return {};
//   }

//   // Parse optional valueType
//   Type elementType;
//   OptionalParseResult parseResult = parser.parseOptionalType(elementType);
//   if (parseResult.has_value()) {
//     if (failed(*parseResult))
//       return {};
//   } else {
//     elementType = UnresolvedType::get(parser.getContext());
//   }

//   // Parse literal '>'
//   if (parser.parseGreater())
//     return {};

//   return LValueType::get(parser.getContext(), isConst, elementType);
// }

// void LValueType::print(AsmPrinter &printer) const {
//   printer << "<";
//   printer << (getIsConst() ? "const" : "mutable");
//   if (!(getValueType() == UnresolvedType::get(getContext()))) {
//     printer << ' ';
//     printer.printStrippedAttrOrType(getValueType());
//   }
//   printer << ">";
// }

namespace mlir {
namespace csl {

// Type constraint implementations
#include "csl-mlir/Dialect/CSL/IR/CSLTypeConstraints.cpp.inc"

//===----------------------------------------------------------------------===//
// Type Information Free Functions
//===----------------------------------------------------------------------===//

std::optional<uint32_t> getSizeInBytes(Type type) {
  // MLIR builtin integer types: si8/ui8 → 1, si16/ui16 → 2, si32/ui32 → 4
  if (auto t = mlir::dyn_cast<IntegerType>(type))
    return (t.getWidth() + 7) / 8;

  // MLIR builtin float types: f16/bf16 → 2, f32 → 4
  if (auto t = mlir::dyn_cast<FloatType>(type))
    return t.getWidth() / 8;

  // CSL types implementing TypeInfoInterface (CB16, Array, Struct, Enum)
  if (auto iface = mlir::dyn_cast<TypeInfoInterface>(type))
    return iface.getSizeInBytes();

  // DSD descriptor types — when spilled to a coroutine frame, conservatively
  // reserve 32 bytes per DSD. The real on-chip representation lives in DSR
  // registers; this is a stack-spill size large enough for the descriptor.
  // FIXME: use the actual descriptor size once known.
  if (mlir::isa<Mem1dDsdType, Mem4dDsdType, FabinDsdType, FaboutDsdType,
                FifoDsdType, DsrDestType, DsrSrc0Type, DsrSrc1Type,
                DsrFifoDestType, DsrFifoSrc1Type, XdsrType, SrType>(type))
    return 32;

  return std::nullopt;
}

uint32_t getAlignmentForType(Type type) {
  if (auto arrTy = mlir::dyn_cast<ArrayType>(type)) {
    if (auto intElemTy = mlir::dyn_cast<IntegerType>(arrTy.getElementType());
        intElemTy /* && intElemTy.getWidth() == 8 */) {
      // Workaround: Coro frames lowers to arrays of i8, but they need to be
      // aligned to the alignment of the largest frame slot they contain. For
      // now, just special case i8 arrays to have 4-byte alignment.
      // FIXME: We need 8 byte alignment to workaround a cslc compiler bug,
      // which seems to ignore consecutive stores to the state slots in
      // parallel_await when the slots are just 4 bytes apart. Dont ask me. I
      // debugged this for 2 days, and I have no idea what is happening in
      // cslc..
      return 8;
    }
    return getAlignmentForType(arrTy.getElementType());
  }
  if (auto structTy = mlir::dyn_cast<StructType>(type)) {
    uint32_t maxAlign = 1;
    for (Type ft : structTy.getFieldTypes())
      maxAlign = std::max(maxAlign, getAlignmentForType(ft));
    return maxAlign;
  }
  auto size = getSizeInBytes(type);
  if (!size) {
    llvm::errs() << "type has unknown size, cannot determine alignment: "
                 << type << "\n";
  }
  assert(size && "type has unknown size, cannot determine alignment");
  return std::min(8u, *size);
}

bool hasPerPELayout(Type type) {
  if (auto iface = mlir::dyn_cast<TypeInfoInterface>(type))
    return iface.hasPerPELayout();
  return false;
}

std::optional<uint32_t> getSizeInBytesForPE(Type type, int64_t x, int64_t y) {
  // MLIR builtin types: no per-PE layout
  if (auto t = mlir::dyn_cast<IntegerType>(type))
    return (t.getWidth() + 7) / 8;
  if (auto t = mlir::dyn_cast<FloatType>(type))
    return t.getWidth() / 8;

  if (auto iface = mlir::dyn_cast<TypeInfoInterface>(type))
    return iface.getSizeInBytesForPE(x, y);

  return std::nullopt;
}

bool isRuntimeType(Type type) {
  // MLIR builtin integer and float types (including bool/i1) are all runtime
  if (mlir::isa<IntegerType, FloatType>(type))
    return true;

  // CSL types implementing TypeInfoInterface: delegate to the type
  if (auto iface = mlir::dyn_cast<TypeInfoInterface>(type))
    return iface.isRuntimeType();

  // Hardware-visible types not implementing the interface
  if (mlir::isa<ColorType, DirectionType, InputQueueType, OutputQueueType,
                DataTaskIdType, LocalTaskIdType, ControlTaskIdType, UtIdType,
                Mem1dDsdType, Mem4dDsdType, FabinDsdType, FaboutDsdType,
                FifoDsdType, DsrDestType, DsrSrc0Type, DsrSrc1Type,
                DsrFifoDestType, DsrFifoSrc1Type, XdsrType, SrType, RangeType>(
          type))
    return true;

  return false;
}

bool isVoidLikeType(Type type) {
  if (mlir::isa<VoidType>(type))
    return true;
  auto structType = mlir::dyn_cast<StructType>(type);
  return structType && llvm::all_of(structType.getFieldTypes(), isVoidLikeType);
}

TypedAttr getVoidLikeValue(Type type) {
  assert(isVoidLikeType(type) && "type has more than one value");
  if (mlir::isa<VoidType>(type))
    return VoidAttr::get(type.getContext());
  auto structType = mlir::cast<StructType>(type);
  SmallVector<TypedAttr> fieldValues;
  for (Type fieldType : structType.getFieldTypes())
    fieldValues.push_back(getVoidLikeValue(fieldType));
  return StructValueAttr::get(type.getContext(), fieldValues,
                              structType.getDefLocation(),
                              structType.getFieldNames());
}

void addVoidFieldStripping(AttrTypeReplacer &replacer) {
  // A struct left without fields becomes the empty tuple: CSL has no empty
  // named structs.
  replacer.addReplacement([](StructType structType) -> std::optional<Type> {
    if (!llvm::any_of(structType.getFieldTypes(), llvm::IsaPred<VoidType>))
      return std::nullopt;
    SmallVector<StringAttr> names;
    SmallVector<Type> types;
    for (auto [index, fieldType] : llvm::enumerate(structType.getFieldTypes())) {
      if (mlir::isa<VoidType>(fieldType))
        continue;
      if (!structType.isTuple())
        names.push_back(structType.getFieldNames()[index]);
      types.push_back(fieldType);
    }
    if (types.empty())
      return StructType::get(structType.getContext(), ArrayRef<Type>());
    return StructType::get(structType.getContext(), names, types,
                           structType.getDefLocation());
  });
  replacer.addReplacement(
      [](StructValueAttr value) -> std::optional<Attribute> {
        ArrayRef<TypedAttr> fieldValues = value.getFieldValues();
        if (!llvm::any_of(fieldValues, llvm::IsaPred<VoidAttr>))
          return std::nullopt;
        SmallVector<StringAttr> names;
        SmallVector<TypedAttr> values;
        for (auto [index, fieldValue] : llvm::enumerate(fieldValues)) {
          if (mlir::isa<VoidAttr>(fieldValue))
            continue;
          if (!value.isTuple())
            names.push_back(value.getFieldNames()[index]);
          values.push_back(fieldValue);
        }
        if (values.empty())
          return StructValueAttr::get(value.getContext(), {}, LocationAttr(),
                                      {});
        return StructValueAttr::get(value.getContext(), values,
                                    value.getDefLocation(), names);
      });
}

Type stripVoidFields(Type type) {
  AttrTypeReplacer replacer;
  addVoidFieldStripping(replacer);
  return replacer.replace(type);
}

} // namespace csl
} // namespace mlir
