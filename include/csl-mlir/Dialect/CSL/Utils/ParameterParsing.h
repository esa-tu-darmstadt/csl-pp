#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/MLIRContext.h"
namespace mlir {
namespace csl {

//===----------------------------------------------------------------------===//
// Dimensions
//===----------------------------------------------------------------------===//

/// A width/height pair used for rectangle, fabric-dims, and fabric-offsets.
struct Dimensions {
  uint16_t width = 0;
  uint16_t height = 0;

  bool operator==(const Dimensions &rhs) const {
    return width == rhs.width && height == rhs.height;
  }

  /// Returns true if both dimensions are non-zero (i.e. configured).
  explicit operator bool() const { return width != 0 && height != 0; }
};

//===----------------------------------------------------------------------===//
// Parameter parsing
//===----------------------------------------------------------------------===//

/// A parsed parameter value (either int or float)
struct ParamValue {
  enum class Type { Int, Float, Color } type;
  union {
    int64_t intVal;
    double floatVal;
    uint16_t colorVal;
  };

  static ParamValue fromInt(int64_t val) {
    ParamValue pv;
    pv.type = ParamValue::Type::Int;
    pv.intVal = val;
    return pv;
  }

  static ParamValue fromFloat(double val) {
    ParamValue pv;
    pv.type = ParamValue::Type::Float;
    pv.floatVal = val;
    return pv;
  }

  static ParamValue fromColor(uint16_t val) {
    ParamValue pv;
    pv.type = ParamValue::Type::Color;
    pv.colorVal = val;
    return pv;
  }

  bool operator==(const ParamValue &other) const {
    if (type != other.type)
      return false;
    switch (type) {
    case Type::Int:
      return intVal == other.intVal;
    case Type::Float:
      return floatVal == other.floatVal;
    case Type::Color:
      return colorVal == other.colorVal;
    }
    llvm_unreachable("unknown ParamValue::Type");
  }

  /// Create a TypedAttr for this value in the given context
  TypedAttr toAttr(MLIRContext *ctx) const {
    switch (type) {
    case Type::Int:
      return csl::ComptimeIntAttr::get(
          ctx, llvm::APInt(64, intVal, /*isSigned=*/true));
    case Type::Float:
      return csl::ComptimeFloatAttr::get(ctx, llvm::APFloat(floatVal));
    case Type::Color:
      return csl::ColorAttr::get(ctx, colorVal);
    }
  }
};

/// Holds parsed parameter bindings with their string storage.
/// Values are stored as raw values, not MLIR attributes, to avoid context
/// lifetime issues.
struct ParsedParams {
  llvm::SmallVector<std::pair<std::string, ParamValue>> values;

  bool operator==(const ParsedParams &other) const {
    return values == other.values;
  }

  /// Parse a params string like "DIM:2,Nz:4,FP:1.5" given by the --params CLI
  /// option. Returns failure if parsing fails.
  static FailureOr<ParsedParams> parseParams(StringRef params);

  /// Parse a params string like "x_in:1,ax_out:3" given by the --colors CLI
  /// option. Returns failure if parsing fails.
  static FailureOr<ParsedParams> parseColors(StringRef params);

  /// Create a bindings map with attributes for the given context.
  /// CLI params use UnknownLoc since they don't have a source location.
  ParamBindingsMap toBindings(MLIRContext *ctx) const;

  /// Serialize to a string like "DIM:2,Nz:4,FP:1.5,x_in:c1".
  std::string toString() const;
};

} // namespace csl
} // namespace mlir