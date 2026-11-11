//===- RollLayoutLoopsPass.cpp - Compact layout calls into loops ----------===//
//
// This pass compacts repeated layout builtin calls (@set_tile_code,
// @set_color_config) into for loops where possible.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <set>

#define DEBUG_TYPE "roll-layout-loops"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_ROLLLAYOUTLOOPSPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::typed_csl;

namespace {

/// A coordinate in the PE grid.
struct Coord {
  int16_t x;
  int16_t y;

  bool operator==(const Coord &other) const {
    return x == other.x && y == other.y;
  }
  bool operator<(const Coord &other) const {
    if (y != other.y)
      return y < other.y;
    return x < other.x;
  }
};

/// An axis-aligned rectangle defined by its corners.
struct Rectangle {
  int16_t x0, y0; // top-left corner (inclusive)
  int16_t x1, y1; // bottom-right corner (inclusive)

  int width() const { return x1 - x0 + 1; }
  int height() const { return y1 - y0 + 1; }
  int area() const { return width() * height(); }

  bool isSinglePoint() const { return x0 == x1 && y0 == y1; }
  bool isHorizontalStrip() const { return y0 == y1 && x0 != x1; }
  bool isVerticalStrip() const { return x0 == x1 && y0 != y1; }
};

/// Information about a group of builtin calls with the same invariant args.
struct CallGroup {
  /// The invariant arguments (everything except x, y coordinates).
  SmallVector<Value> invariantArgs;
  /// All coordinates where this call appears.
  std::set<Coord> coords;
  /// Original builtin kind.
  csl::BuiltinKind builtin;
  /// Location for diagnostics.
  std::optional<Location> loc;
};

/// Decompose a set of coordinates into minimal axis-aligned rectangles.
/// Uses a greedy algorithm that prioritizes larger rectangles.
static SmallVector<Rectangle> decomposeIntoRectangles(std::set<Coord> coords) {
  SmallVector<Rectangle> result;

  while (!coords.empty()) {
    // Pick the first remaining coordinate (sorted by y, then x)
    Coord seed = *coords.begin();

    // Extend horizontally as far as possible
    int16_t x1 = seed.x;
    while (coords.count({static_cast<int16_t>(x1 + 1), seed.y}))
      ++x1;

    // Extend vertically as far as possible (checking entire row exists)
    int16_t y1 = seed.y;
    while (true) {
      bool rowExists = true;
      for (int16_t x = seed.x; x <= x1; ++x) {
        if (!coords.count({x, static_cast<int16_t>(y1 + 1)})) {
          rowExists = false;
          break;
        }
      }
      if (!rowExists)
        break;
      ++y1;
    }

    // Record rectangle and remove covered coordinates
    result.push_back({seed.x, seed.y, x1, y1});
    for (int16_t y = seed.y; y <= y1; ++y) {
      for (int16_t x = seed.x; x <= x1; ++x) {
        coords.erase({x, y});
      }
    }
  }

  return result;
}

/// Extract constant integer value from a Value, if possible.
static std::optional<int64_t> getConstantInt(Value value) {
  auto constOp = value.getDefiningOp<typed_csl::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  if (auto intAttr = mlir::dyn_cast<csl::ComptimeIntAttr>(constOp.getValue()))
    return intAttr.getInt();
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getInt();

  return std::nullopt;
}

/// Generate a key string for grouping calls by their invariant arguments.
static std::string generateGroupKey(csl::BuiltinKind builtin,
                                    ArrayRef<Value> invariantArgs) {
  std::string key;
  llvm::raw_string_ostream os(key);

  // @set_tile_code calls must be emitted before @set_color_config calls (for
  // whatever reason..). So we prefix the key for those two kinds to ensure
  // correct ordering.

  if (builtin == csl::BuiltinKind::SetTileCode)
    os << "0:";
  else if (builtin == csl::BuiltinKind::SetColorConfig)
    os << "1:";
  else

    os << static_cast<int>(builtin) << ":";
  for (Value arg : invariantArgs) {
    if (auto constOp = arg.getDefiningOp<typed_csl::ConstantOp>()) {
      constOp.getValue().print(os);
    } else {
      os << arg.getAsOpaquePointer();
    }
    os << ",";
  }
  return key;
}

/// Helper to create a comptime int constant.
static Value createComptimeInt(OpBuilder &builder, Location loc,
                               int64_t value) {
  auto attr = csl::ComptimeIntAttr::get(
      builder.getContext(), llvm::APInt(64, value, /*isSigned=*/true));
  return typed_csl::ConstantOp::create(builder, loc, attr.getType(), attr);
}

/// Helper to create a signed 16-bit integer constant.
static Value createSI16(OpBuilder &builder, Location loc, int16_t value) {
  auto elemType =
      IntegerType::get(builder.getContext(), 16, IntegerType::Signed);
  auto attr = IntegerAttr::get(elemType, value);
  return typed_csl::ConstantOp::create(builder, loc, elemType, attr);
}

struct RollLayoutLoopsPass
    : public typed_csl::impl::RollLayoutLoopsPassBase<RollLayoutLoopsPass> {

  void runOnOperation() override {
    auto unit = getOperation();

    // Find layout block
    typed_csl::LayoutOp layout;
    unit.walk([&](typed_csl::LayoutOp op) {
      layout = op;
      return WalkResult::interrupt();
    });

    if (!layout)
      return; // No layout block - nothing to do

    // Collect rollable builtin calls
    // Must be a ordered map to ensure that we emit @set_tile_code before
    // @set_color_config (prefixed via generateGroupKey)
    std::map<std::string, CallGroup> groups;
    SmallVector<typed_csl::BuiltinCallOp> callsToRemove;

    layout.walk([&](typed_csl::BuiltinCallOp call) {
      auto builtin = call.getBuiltin();

      // Only handle @set_tile_code and @set_color_config
      if (builtin != csl::BuiltinKind::SetTileCode &&
          builtin != csl::BuiltinKind::SetColorConfig)
        return;

      auto args = call.getArgs();
      if (args.size() < 2)
        return;

      // Extract x, y coordinates - must be constants
      auto xOpt = getConstantInt(args[0]);
      auto yOpt = getConstantInt(args[1]);
      if (!xOpt || !yOpt)
        return;

      // Invariant args are everything after x, y
      SmallVector<Value> invariantArgs(args.begin() + 2, args.end());
      std::string key = generateGroupKey(builtin, invariantArgs);

      auto &group = groups[key];
      if (group.coords.empty()) {
        group.invariantArgs = std::move(invariantArgs);
        group.builtin = builtin;
        group.loc = call.getLoc();
      }
      group.coords.insert(
          {static_cast<int16_t>(*xOpt), static_cast<int16_t>(*yOpt)});
      callsToRemove.push_back(call);
    });

    // If no rollable calls found, nothing to do
    if (groups.empty())
      return;

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << groups.size() << " groups of rollable calls\n");

    // Find insertion point: after the last call we're removing
    // (to maintain relative order with other layout ops)
    Operation *insertionPoint = nullptr;
    for (auto call : callsToRemove) {
      if (!insertionPoint || insertionPoint->isBeforeInBlock(call))
        insertionPoint = call;
    }

    OpBuilder builder(insertionPoint);
    builder.setInsertionPointAfter(insertionPoint);
    MLIRContext *ctx = &getContext();

    // Process each group
    for (auto &entry : groups) {
      CallGroup &group = entry.second;
      auto rectangles = decomposeIntoRectangles(group.coords);

      LLVM_DEBUG(llvm::dbgs()
                 << "Group with " << group.coords.size() << " coords -> "
                 << rectangles.size() << " rectangles\n");

      for (const Rectangle &rect : rectangles) {
        emitRectangle(builder, ctx, group, rect);
      }
    }

    // Remove original calls
    for (auto call : callsToRemove) {
      call.erase();
    }

    // Cleanup unused ops in layout block
    {
      IRRewriter rewriter(builder);
      (void)simplifyRegions(rewriter, layout->getRegions());
    }
  }

private:
  /// Emit code for a single rectangle (single call, 1D loop, or 2D nested
  /// loop).
  void emitRectangle(OpBuilder &builder, MLIRContext *ctx, CallGroup &group,
                     const Rectangle &rect) {
    Location loc = *group.loc;

    if (rect.isSinglePoint()) {
      // Single call
      emitBuiltinCall(builder, loc, group, rect.x0, rect.y0);
    } else if (rect.isHorizontalStrip()) {
      // Single for loop over x
      emitForLoop(builder, ctx, loc, group, rect, /*outerIsY=*/false);
    } else if (rect.isVerticalStrip()) {
      // Single for loop over y
      emitForLoop(builder, ctx, loc, group, rect, /*outerIsY=*/true);
    } else {
      // Nested for loops: outer y, inner x
      emitNestedForLoops(builder, ctx, loc, group, rect);
    }
  }

  /// Emit a single builtin call with constant x, y.
  void emitBuiltinCall(OpBuilder &builder, Location loc, CallGroup &group,
                       int16_t x, int16_t y) {
    Value xConst = createComptimeInt(builder, loc, x);
    Value yConst = createComptimeInt(builder, loc, y);

    SmallVector<Value> args;
    args.push_back(xConst);
    args.push_back(yConst);
    args.append(group.invariantArgs.begin(), group.invariantArgs.end());

    // Void-returning builtin, pass empty result type
    typed_csl::BuiltinCallOp::create(builder, loc, /*result=*/Type{},
                                     group.builtin, args);
  }

  /// Emit a single for loop.
  void emitForLoop(OpBuilder &builder, MLIRContext *ctx, Location loc,
                   CallGroup &group, const Rectangle &rect, bool outerIsY) {
    int16_t start = outerIsY ? rect.y0 : rect.x0;
    int16_t stop = outerIsY ? rect.y1 + 1 : rect.x1 + 1;
    int16_t fixedCoord = outerIsY ? rect.x0 : rect.y0;
    StringRef varName = outerIsY ? "y" : "x";

    // Create range: @range(i16, start, stop)
    auto rangeValue = createRange(builder, ctx, loc, start, stop);

    // Create for loop with proper attributes
    auto elemNameAttr = builder.getStringAttr(varName);
    auto isInlineAttr = builder.getBoolAttr(true); // inline for to unroll
    auto forOp =
        typed_csl::ForOp::create(builder, loc, rangeValue, elemNameAttr,
                                 /*index_name=*/StringAttr(),
                                 /*label=*/StringAttr(), isInlineAttr);
    forOp.getBody().emplaceBlock();
    Block &body = forOp.getBody().front();

    // Add block argument for loop variable (si16 - signed 16-bit integer)
    auto elemType = IntegerType::get(ctx, 16, IntegerType::Signed);
    body.addArgument(elemType, loc);
    Value loopVar = body.getArgument(0);

    // Build the call inside the loop
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&body);

    Value fixedConst = createComptimeInt(builder, loc, fixedCoord);

    SmallVector<Value> args;
    if (outerIsY) {
      args.push_back(fixedConst); // x is fixed
      args.push_back(loopVar);    // y varies
    } else {
      args.push_back(loopVar);    // x varies
      args.push_back(fixedConst); // y is fixed
    }
    args.append(group.invariantArgs.begin(), group.invariantArgs.end());

    // Void-returning builtin, pass empty result type
    typed_csl::BuiltinCallOp::create(builder, loc, /*result=*/Type{},
                                     group.builtin, args);

    // Add void yield terminator for the for-loop body
    typed_csl::YieldOp::create(builder, loc, Value());
  }

  /// Emit nested for loops (outer y, inner x).
  void emitNestedForLoops(OpBuilder &builder, MLIRContext *ctx, Location loc,
                          CallGroup &group, const Rectangle &rect) {
    auto elemType = IntegerType::get(ctx, 16, IntegerType::Signed);
    auto isInlineAttr = builder.getBoolAttr(true);

    // Outer loop over y
    auto yRange = createRange(builder, ctx, loc, rect.y0, rect.y1 + 1);
    auto yNameAttr = builder.getStringAttr("y");
    auto outerFor =
        typed_csl::ForOp::create(builder, loc, yRange, yNameAttr,
                                 /*index_name=*/StringAttr(),
                                 /*label=*/StringAttr(), isInlineAttr);
    outerFor.getBody().emplaceBlock();
    Block &outerBody = outerFor.getBody().front();
    outerBody.addArgument(elemType, loc);
    Value yVar = outerBody.getArgument(0);

    // Inner loop over x
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&outerBody);

    auto xRange = createRange(builder, ctx, loc, rect.x0, rect.x1 + 1);
    auto xNameAttr = builder.getStringAttr("x");
    auto innerFor =
        typed_csl::ForOp::create(builder, loc, xRange, xNameAttr,
                                 /*index_name=*/StringAttr(),
                                 /*label=*/StringAttr(), isInlineAttr);
    innerFor.getBody().emplaceBlock();
    Block &innerBody = innerFor.getBody().front();
    innerBody.addArgument(elemType, loc);
    Value xVar = innerBody.getArgument(0);

    // Build the call inside the inner loop
    builder.setInsertionPointToStart(&innerBody);

    SmallVector<Value> args;
    args.push_back(xVar);
    args.push_back(yVar);
    args.append(group.invariantArgs.begin(), group.invariantArgs.end());

    // Void-returning builtin, pass empty result type
    typed_csl::BuiltinCallOp::create(builder, loc, /*result=*/Type{},
                                     group.builtin, args);

    // Add void yield terminator for the inner for-loop body
    typed_csl::YieldOp::create(builder, loc, Value());

    // Add void yield terminator for the outer for-loop body
    builder.setInsertionPointToEnd(&outerBody);
    typed_csl::YieldOp::create(builder, loc, Value());
  }

  /// Create a range value: @range(si16, start, stop, step).
  Value createRange(OpBuilder &builder, MLIRContext *ctx, Location loc,
                    int16_t start, int16_t stop) {
    // Create type constant for si16
    auto elemType = IntegerType::get(ctx, 16, IntegerType::Signed);
    auto typeAttr = csl::TypeAttr::get(elemType);
    auto typeConst = typed_csl::ConstantOp::create(
        builder, loc, typeAttr.getType(), typeAttr);

    // Create start, stop, step constants as si16
    auto startConst = createSI16(builder, loc, start);
    auto stopConst = createSI16(builder, loc, stop);
    auto stepConst = createSI16(builder, loc, 1);

    // Create @range(si16, start, stop, step) call - 4 argument form
    auto rangeType = csl::RangeType::get(elemType);
    auto rangeOp = typed_csl::BuiltinCallOp::create(
        builder, loc, rangeType, csl::BuiltinKind::Range,
        ValueRange{typeConst, startConst, stopConst, stepConst});
    return rangeOp.getResult();
  }
};

} // namespace
