//===- CSLAttrs.cpp - CSL Attribute Custom Parsers/Printers -----*- C++ -*-===//
//
// Custom parse and print implementations for CSL attributes.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Helper functions for parsing/printing
//===----------------------------------------------------------------------===//

/// Parse a keyword-value pair like "extent=10"
static ParseResult parseKeywordValue(AsmParser &parser, StringRef keyword,
                                     uint16_t &value) {
  if (parser.parseKeyword(keyword) || parser.parseEqual() ||
      parser.parseInteger(value))
    return failure();
  return success();
}

static ParseResult parseKeywordValue(AsmParser &parser, StringRef keyword,
                                     int8_t &value) {
  int64_t val;
  if (parser.parseKeyword(keyword) || parser.parseEqual() ||
      parser.parseInteger(val))
    return failure();
  value = static_cast<int8_t>(val);
  return success();
}

static ParseResult parseKeywordValue(AsmParser &parser, StringRef keyword,
                                     int16_t &value) {
  int64_t val;
  if (parser.parseKeyword(keyword) || parser.parseEqual() ||
      parser.parseInteger(val))
    return failure();
  value = static_cast<int16_t>(val);
  return success();
}

static ParseResult parseKeywordBool(AsmParser &parser, StringRef keyword,
                                    bool &value) {
  if (parser.parseKeyword(keyword) || parser.parseEqual())
    return failure();

  StringRef boolStr;
  if (parser.parseKeyword(&boolStr))
    return failure();

  if (boolStr == "true")
    value = true;
  else if (boolStr == "false")
    value = false;
  else
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'true' or 'false'");
  return success();
}

//===----------------------------------------------------------------------===//
// Mem1dDsdAttr
//===----------------------------------------------------------------------===//

// Format: #csl.mem1d_dsd<base_addr, extent=N, stride=N, offset=N,
// wavelet_index_offset=bool>
Attribute Mem1dDsdAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse base_address (any TypedAttr)
  TypedAttr baseAddress;
  if (parser.parseAttribute(baseAddress))
    return {};

  if (parser.parseComma())
    return {};

  // Parse extent=N
  uint16_t extent;
  if (parseKeywordValue(parser, "extent", extent))
    return {};

  if (parser.parseComma())
    return {};

  // Parse stride=N
  int8_t stride;
  if (parseKeywordValue(parser, "stride", stride))
    return {};

  if (parser.parseComma())
    return {};

  // Parse offset=N
  int16_t offset;
  if (parseKeywordValue(parser, "offset", offset))
    return {};

  if (parser.parseComma())
    return {};

  // Parse wavelet_index_offset=bool
  bool waveletIndexOffset;
  if (parseKeywordBool(parser, "wavelet_index_offset", waveletIndexOffset))
    return {};

  if (parser.parseGreater())
    return {};

  return Mem1dDsdAttr::get(parser.getContext(), baseAddress, extent, stride,
                           offset, waveletIndexOffset);
}

void Mem1dDsdAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getBaseAddress());
  printer << ", extent=" << getExtent();
  printer << ", stride=" << static_cast<int>(getStride());
  printer << ", offset=" << getOffset();
  printer << ", wavelet_index_offset="
          << (getWaveletIndexOffset() ? "true" : "false");
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Mem4dDsdAttr
//===----------------------------------------------------------------------===//

// Format: #csl.mem4d_dsd<base_addr, extents=[N,...], strides=[N,...], offset=N,
// wavelet_index_offset=bool>
Attribute Mem4dDsdAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse base_address
  TypedAttr baseAddress;
  if (parser.parseAttribute(baseAddress))
    return {};

  if (parser.parseComma())
    return {};

  // Parse extents=[N, ...]
  if (parser.parseKeyword("extents") || parser.parseEqual() ||
      parser.parseLSquare())
    return {};

  SmallVector<uint16_t> extents;
  uint16_t ext;
  if (parser.parseInteger(ext))
    return {};
  extents.push_back(ext);

  while (parser.parseOptionalComma().succeeded()) {
    if (parser.parseInteger(ext))
      return {};
    extents.push_back(ext);
  }

  if (parser.parseRSquare() || parser.parseComma())
    return {};

  // Parse strides=[N, ...]
  if (parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare())
    return {};

  SmallVector<int16_t> strides;
  int64_t strVal;
  if (parser.parseInteger(strVal))
    return {};
  strides.push_back(static_cast<int16_t>(strVal));

  while (parser.parseOptionalComma().succeeded()) {
    if (parser.parseInteger(strVal))
      return {};
    strides.push_back(static_cast<int16_t>(strVal));
  }

  if (parser.parseRSquare() || parser.parseComma())
    return {};

  // Parse offset=N
  int16_t offset;
  if (parseKeywordValue(parser, "offset", offset))
    return {};

  if (parser.parseComma())
    return {};

  // Parse wavelet_index_offset=bool
  bool waveletIndexOffset;
  if (parseKeywordBool(parser, "wavelet_index_offset", waveletIndexOffset))
    return {};

  if (parser.parseGreater())
    return {};

  return Mem4dDsdAttr::get(parser.getContext(), baseAddress, extents, strides,
                           offset, waveletIndexOffset);
}

void Mem4dDsdAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getBaseAddress());

  printer << ", extents=[";
  llvm::interleaveComma(getExtents(), printer);
  printer << "]";

  printer << ", strides=[";
  llvm::interleaveComma(getStrides(), printer);
  printer << "]";

  printer << ", offset=" << getOffset();
  printer << ", wavelet_index_offset="
          << (getWaveletIndexOffset() ? "true" : "false");
  printer << ">";
}

//===----------------------------------------------------------------------===//
// FabinDsdAttr
//===----------------------------------------------------------------------===//

// Format: #csl.fabin_dsd<extent=N[, fabric_color=attr][, input_queue=attr]
//                        [, priority=attr][, control_transform=bool]
//                        [, simd_mode=attr]>
Attribute FabinDsdAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse extent=N
  uint16_t extent;
  if (parseKeywordValue(parser, "extent", extent))
    return {};

  // Optional parameters
  ColorAttr fabricColor;
  InputQueueAttr inputQueue;
  PriorityAttr priority;
  bool controlTransform = false;
  SimdModeAttr simdMode;

  while (parser.parseOptionalComma().succeeded()) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return {};

    if (parser.parseEqual())
      return {};

    if (keyword == "fabric_color") {
      if (parser.parseAttribute(fabricColor))
        return {};
    } else if (keyword == "input_queue") {
      if (parser.parseAttribute(inputQueue))
        return {};
    } else if (keyword == "priority") {
      if (parser.parseAttribute(priority))
        return {};
    } else if (keyword == "control_transform") {
      StringRef boolStr;
      if (parser.parseKeyword(&boolStr))
        return {};
      controlTransform = (boolStr == "true");
    } else if (keyword == "simd_mode") {
      if (parser.parseAttribute(simdMode))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown keyword: " + keyword);
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return FabinDsdAttr::get(parser.getContext(), fabricColor, extent, inputQueue,
                           priority, controlTransform, simdMode);
}

void FabinDsdAttr::print(AsmPrinter &printer) const {
  printer << "<extent=" << getExtent();

  if (getFabricColor()) {
    printer << ", fabric_color=";
    printer.printAttribute(getFabricColor());
  }
  if (getInputQueue()) {
    printer << ", input_queue=";
    printer.printAttribute(getInputQueue());
  }
  if (getPriority()) {
    printer << ", priority=";
    printer.printAttribute(getPriority());
  }
  if (getControlTransform()) {
    printer << ", control_transform=true";
  }
  if (getSimdMode()) {
    printer << ", simd_mode=";
    printer.printAttribute(getSimdMode());
  }

  printer << ">";
}

//===----------------------------------------------------------------------===//
// FaboutDsdAttr
//===----------------------------------------------------------------------===//

// Format: #csl.fabout_dsd<extent=N[, fabric_color=attr][, output_queue=attr]
//                         [, control=bool][, wavelet_index_offset=bool]
//                         [, advance_switch=bool][, control_transform=bool]
//                         [, simd_mode=attr][, zero=attr]>
Attribute FaboutDsdAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse extent=N
  uint16_t extent;
  if (parseKeywordValue(parser, "extent", extent))
    return {};

  // Optional parameters with defaults
  ColorAttr fabricColor;
  OutputQueueAttr outputQueue;
  bool control = false;
  bool waveletIndexOffset = false;
  bool advanceSwitch = false;
  bool controlTransform = false;
  SimdModeAttr simdMode;
  ZeroAttr zero;

  while (parser.parseOptionalComma().succeeded()) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return {};

    if (parser.parseEqual())
      return {};

    if (keyword == "fabric_color") {
      if (parser.parseAttribute(fabricColor))
        return {};
    } else if (keyword == "output_queue") {
      if (parser.parseAttribute(outputQueue))
        return {};
    } else if (keyword == "control") {
      StringRef boolStr;
      if (parser.parseKeyword(&boolStr))
        return {};
      control = (boolStr == "true");
    } else if (keyword == "wavelet_index_offset") {
      StringRef boolStr;
      if (parser.parseKeyword(&boolStr))
        return {};
      waveletIndexOffset = (boolStr == "true");
    } else if (keyword == "advance_switch") {
      StringRef boolStr;
      if (parser.parseKeyword(&boolStr))
        return {};
      advanceSwitch = (boolStr == "true");
    } else if (keyword == "control_transform") {
      StringRef boolStr;
      if (parser.parseKeyword(&boolStr))
        return {};
      controlTransform = (boolStr == "true");
    } else if (keyword == "simd_mode") {
      if (parser.parseAttribute(simdMode))
        return {};
    } else if (keyword == "zero") {
      if (parser.parseAttribute(zero))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown keyword: " + keyword);
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return FaboutDsdAttr::get(parser.getContext(), fabricColor, extent,
                            outputQueue, control, waveletIndexOffset,
                            advanceSwitch, controlTransform, simdMode, zero);
}

void FaboutDsdAttr::print(AsmPrinter &printer) const {
  printer << "<extent=" << getExtent();

  if (getFabricColor()) {
    printer << ", fabric_color=";
    printer.printAttribute(getFabricColor());
  }
  if (getOutputQueue()) {
    printer << ", output_queue=";
    printer.printAttribute(getOutputQueue());
  }
  if (getControl()) {
    printer << ", control=true";
  }
  if (getWaveletIndexOffset()) {
    printer << ", wavelet_index_offset=true";
  }
  if (getAdvanceSwitch()) {
    printer << ", advance_switch=true";
  }
  if (getControlTransform()) {
    printer << ", control_transform=true";
  }
  if (getSimdMode()) {
    printer << ", simd_mode=";
    printer.printAttribute(getSimdMode());
  }
  if (getZero()) {
    printer << ", zero=";
    printer.printAttribute(getZero());
  }

  printer << ">";
}

//===----------------------------------------------------------------------===//
// FifoDsdAttr
//===----------------------------------------------------------------------===//

// Format: #csl.fifo_dsd<buffer_ref[, activate_pop=attr][, activate_push=attr]
//                       [, priority=attr][, simd_mode=attr]>
Attribute FifoDsdAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse buffer (GlobalRefAttr)
  GlobalRefAttr buffer;
  if (parser.parseAttribute(buffer))
    return {};

  // Optional parameters
  FuncRefAttr activatePop;
  FuncRefAttr activatePush;
  PriorityAttr priority;
  SimdModeAttr simdMode;

  while (parser.parseOptionalComma().succeeded()) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return {};

    if (parser.parseEqual())
      return {};

    if (keyword == "activate_pop") {
      if (parser.parseAttribute(activatePop))
        return {};
    } else if (keyword == "activate_push") {
      if (parser.parseAttribute(activatePush))
        return {};
    } else if (keyword == "priority") {
      if (parser.parseAttribute(priority))
        return {};
    } else if (keyword == "simd_mode") {
      if (parser.parseAttribute(simdMode))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown keyword: " + keyword);
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return FifoDsdAttr::get(parser.getContext(), buffer, activatePop,
                          activatePush, priority, simdMode);
}

void FifoDsdAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getBuffer());

  if (getActivatePop()) {
    printer << ", activate_pop=";
    printer.printAttribute(getActivatePop());
  }
  if (getActivatePush()) {
    printer << ", activate_push=";
    printer.printAttribute(getActivatePush());
  }
  if (getPriority()) {
    printer << ", priority=";
    printer.printAttribute(getPriority());
  }
  if (getSimdMode()) {
    printer << ", simd_mode=";
    printer.printAttribute(getSimdMode());
  }

  printer << ">";
}

//===----------------------------------------------------------------------===//
// DataTaskIdAttr
//===----------------------------------------------------------------------===//

bool DataTaskIdAttr::isFromColor() const {
  return mlir::isa<ColorAttr>(getSource());
}

bool DataTaskIdAttr::isFromInputQueue() const {
  return mlir::isa<InputQueueAttr>(getSource());
}

ColorAttr DataTaskIdAttr::getColor() const {
  return mlir::dyn_cast<ColorAttr>(getSource());
}

InputQueueAttr DataTaskIdAttr::getInputQueue() const {
  return mlir::dyn_cast<InputQueueAttr>(getSource());
}

//===----------------------------------------------------------------------===//
// FuncRefAttr
//===----------------------------------------------------------------------===//

bool FuncRefAttr::isGeneric() const {
  return !mlir::isa<FunctionType>(getType());
}

FunctionType FuncRefAttr::getFunctionType() const {
  return mlir::cast<FunctionType>(getType());
}
