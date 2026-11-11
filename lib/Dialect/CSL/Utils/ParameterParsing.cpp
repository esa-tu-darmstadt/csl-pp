//===- ParameterParsing.cpp - Parameter Parsing Utilities
//---------------------===//
//
// Implementation of utilities for parsing CL parameter strings to attributes.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/Utils/ParameterParsing.h"

using namespace mlir;
using namespace mlir::csl;

namespace {
FailureOr<ParsedParams> parseImpl(StringRef params, bool isColor) {
  ParsedParams result;

  if (params.empty())
    return result;

  llvm::SmallVector<StringRef> pairs;
  params.split(pairs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  for (StringRef pair : pairs) {
    // Split by ':'
    auto colonPos = pair.find(':');
    if (colonPos == StringRef::npos) {
      llvm::errs() << "Invalid param format '" << pair
                   << "', expected NAME:VALUE\n";
      return failure();
    }

    StringRef name = pair.substr(0, colonPos).trim();
    StringRef valueStr = pair.substr(colonPos + 1).trim();

    if (name.empty() || valueStr.empty()) {
      llvm::errs() << "Invalid param format '" << pair
                   << "', name and value cannot be empty\n";
      return failure();
    }

    if (isColor) {
      // For colors, parse value as uint16_t and create a ColorAttr
      unsigned colorVal;
      if (valueStr.getAsInteger(10, colorVal) || colorVal > UINT16_MAX) {
        llvm::errs() << "Invalid color value '" << valueStr
                     << "', must be an integer in [0, 65535]\n";
        return failure();
      }
      ParamValue pv = ParamValue::fromColor(static_cast<uint16_t>(colorVal));
      result.values.push_back({name.str(), pv});
      continue;
    }

    // Determine if value is float or int based on format
    // Float if contains '.' or 'e/E' (scientific notation)
    bool isFloat = valueStr.contains('.') || valueStr.contains_insensitive('e');

    ParamValue pv;
    if (isFloat) {
      double doubleVal;
      if (valueStr.getAsDouble(doubleVal)) {
        llvm::errs() << "Invalid float value '" << valueStr << "'\n";
        return failure();
      }
      pv = ParamValue::fromFloat(doubleVal);
    } else {
      // Parse as integer
      long long intVal;
      if (valueStr.getAsInteger(10, intVal)) {
        llvm::errs() << "Invalid integer value '" << valueStr << "'\n";
        return failure();
      }
      pv = ParamValue::fromInt(intVal);
    }

    result.values.push_back({name.str(), pv});
  }

  return result;
}
} // namespace

FailureOr<ParsedParams> ParsedParams::parseParams(StringRef params) {
  return parseImpl(params, /*isColor=*/false);
}

FailureOr<ParsedParams> ParsedParams::parseColors(StringRef params) {
  return parseImpl(params, /*isColor=*/true);
}

ParamBindingsMap ParsedParams::toBindings(MLIRContext *ctx) const {
  ParamBindingsMap bindings;
  Location unknownLoc = UnknownLoc::get(ctx);
  for (const auto &[name, value] : values) {
    bindings.insert({name, {value.toAttr(ctx), unknownLoc}});
  }
  return bindings;
}

std::string ParsedParams::toString() const {
  std::string result;
  for (const auto &[name, val] : values) {
    if (!result.empty())
      result += ',';
    result += name + ':';
    switch (val.type) {
    case ParamValue::Type::Int:
      result += std::to_string(val.intVal);
      break;
    case ParamValue::Type::Float:
      result += std::to_string(val.floatVal);
      break;
    case ParamValue::Type::Color:
      result += 'c' + std::to_string(val.colorVal);
      break;
    }
  }
  return result;
}