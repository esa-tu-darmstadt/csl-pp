//===- CoroFrameBuilder.h - Coroutine frame layout and access ---*- C++ -*-===//
//
// Coroutine frame analysis, layout computation, and IR emission for the
// byte-precise frame design.
//
// === Design Overview ===
//
// Every coroutine frame is a flat byte array (`[N]u8`). Fields are laid out
// at computed byte offsets, and accessed via pointer arithmetic + @ptrcast.
// This replaces the old struct-based frame approach and enables two key
// capabilities:
//
//   1. **Liveness-based overlap**: Fields with non-overlapping lifetimes
//      can share the same byte range, reducing frame size. The bin-packing
//      algorithm naturally achieves this — no separate coalescing step.
//
//   2. **Per-PE dynamic sizing** (Step 5+): Fields whose types contain
//      PerPEArrayType have sizes that vary across PEs. These go into a
//      "dynamic region" at the end of the frame, with per-PE offsets stored
//      in a descriptor struct. The "static region" (constant-size fields)
//      uses compile-time constant offsets baked into the code.
//
// === Frame Memory Layout ===
//
//   Frame base ([*]u8 pointer)
//   │
//   ├── Static region (offset 0, constant size across all PEs)
//   │   Fields: state, result, params, VariableOps, SSA spills
//   │   All byte offsets are compile-time constants.
//   │   Size: STATIC_SIZE (same on every PE)
//   │
//   └── Dynamic region (starts at STATIC_SIZE, per-PE sizing)
//       [Step 5+, not yet implemented]
//
// === Frame Access Pattern ===
//
// All field access goes through a uniform 4-step IR pattern:
//
//   1. Index into the frame byte array at the field's byte offset
//   2. Take the address of that byte (&frame[offset] → *u8)
//   3. @ptrcast the byte pointer to the field's type (*u8 → *T)
//   4. Dereference the typed pointer to get a mutable lvalue
//
// This emits correct CSL: @ptrcast(*T, &frame[offset]).*
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_COROFRAMEBUILDER_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_COROFRAMEBUILDER_H

#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace mlir {
namespace typed_csl {

//===----------------------------------------------------------------------===//
// Coroutine Frame Calling Convention
//===----------------------------------------------------------------------===//
//
// The caller and callee agree on a fixed layout prefix for the coroutine
// frame so that the caller can initialize parameters and read the result
// without knowing the callee's full frame layout.
//
//   Offset 0:  state (u16, 2 bytes)
//   Offset 2+: params placed consecutively with natural alignment.
//              The result shares the same starting region (offset 2,
//              aligned) because it is written only after the callee
//              finishes, when parameters are dead.
//

/// All convention offsets are in bytes. They are converted to element offsets
/// (byte or word) at the point of IR emission using byteToElementOffset().
static constexpr uint32_t kCoroStateOffset = 0;
static constexpr uint32_t kCoroStateSize = 2;

/// Size of the fixed convention prefix: just the state field. Parameters
/// and result are placed after this prefix with natural alignment.
static constexpr uint32_t kCoroConventionPrefixSize = 2;

/// Attribute name for coroutine frame slot alignment on CompilationUnitOp.
/// Set by CoroPrepareCFPass, read by downstream passes.
static constexpr llvm::StringLiteral kCoroSlotAlignAttr = "coro.slot_align";

/// Read the coroutine frame slot alignment from the compilation unit.
/// Returns 2 (word-aligned) if the attribute is not set.
inline unsigned getCoroSlotAlignment(Operation *unitOp) {
  if (auto attr = unitOp->getAttrOfType<IntegerAttr>(kCoroSlotAlignAttr))
    return attr.getValue().getZExtValue();
  return 2;
}

/// Get the frame array element type for the given slot alignment.
/// align=1 → u8, align=2 → u16.
inline Type getCoroFrameElementType(MLIRContext *ctx, unsigned slotAlign) {
  return IntegerType::get(ctx, slotAlign * 8, IntegerType::Unsigned);
}

/// Get the many-pointer type for the frame: [*]u8 or [*]u16.
inline csl::PointerType getCoroFramePtrType(MLIRContext *ctx,
                                            unsigned slotAlign) {
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  return csl::PointerType::get(elemTy, /*isConst=*/false, /*isArray=*/true);
}

/// Convert a byte offset to an element offset for the frame pointer type.
/// For slotAlign=1 (u8): returns byteOffset as-is.
/// For slotAlign=2 (u16): returns byteOffset / 2, asserting it's even.
inline uint32_t byteToElementOffset(uint32_t byteOffset, unsigned slotAlign) {
  assert(byteOffset % slotAlign == 0 &&
         "byte offset not aligned to slot alignment");
  return byteOffset / slotAlign;
}

/// Create a frame array type [N x elemTy] for the given byte size.
/// For slotAlign=1: [byteSize x u8]. For slotAlign=2: [ceil(byteSize/2) x u16].
inline csl::ArrayType getCoroFrameArrayType(MLIRContext *ctx, uint32_t byteSize,
                                            unsigned slotAlign) {
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  uint32_t elemCount = (byteSize + slotAlign - 1) / slotAlign;
  return csl::ArrayType::get({static_cast<int64_t>(elemCount)}, elemTy);
}

/// Compute the aligned byte offset for a field starting at `currentOffset`.
/// When slotAlign > 1, the result is guaranteed to be a multiple of slotAlign
/// so that `byteToElementOffset()` never asserts on a misaligned offset.
inline uint32_t coroConventionOffset(uint32_t currentOffset, Type fieldType,
                                     unsigned slotAlign = 1) {
  uint32_t align = std::max(csl::getAlignmentForType(fieldType), slotAlign);
  return (currentOffset + align - 1) & ~(align - 1);
}

//===----------------------------------------------------------------------===//
// FrameFieldInfo
//===----------------------------------------------------------------------===//

/// Classification of frame fields.
enum class FieldKind {
  State,      ///< u16 state at offset 0
  Result,     ///< async result at convention offset
  Param,      ///< function parameter at convention offset
  Variable,   ///< VariableOp moved to frame
  Spill,      ///< SSA value spilled across suspend
  CalleeFrame ///< embedded callee coroutine frame
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, FieldKind kind) {
  switch (kind) {
  case FieldKind::State:
    return os << "state";
  case FieldKind::Result:
    return os << "result";
  case FieldKind::Param:
    return os << "param";
  case FieldKind::Variable:
    return os << "variable";
  case FieldKind::Spill:
    return os << "spill";
  case FieldKind::CalleeFrame:
    return os << "callee_frame";
  }
  llvm_unreachable("unknown FieldKind");
}

/// Information about a single value that needs storage in the coroutine frame.
struct FrameFieldInfo {
  /// What kind of field this is.
  FieldKind kind;

  /// The SSA value this slot stores (VariableOp result, op result, etc.).
  /// For synthetic fields (state, result, params), this is a null Value.
  Value value;

  /// The type to store in the frame. For SSA values, this is the value's
  /// type. For lvalue-typed values (VariableOps, parameters), this is the
  /// underlying value type (stripped of LValueType).
  Type storageType;

  /// Natural alignment of storageType in bytes.
  uint32_t alignment;

  /// Block-level liveness bitset. Bit i is set if this value is live at
  /// the entry of block i. Two fields can share the same byte range iff
  /// their liveness bitsets don't overlap (no block where both are live).
  llvm::BitVector liveness;

  /// Whether this field has per-PE sizing (csl::hasPerPELayout(storageType)).
  /// Determines region assignment: static (false) or dynamic (true).
  bool isPerPE = false;

  /// A human-readable name for debug output.
  std::string debugName;

  /// Fixed convention offset, if this field must be placed at a specific
  /// byte offset (state, result, params). nullopt for packer-placed fields.
  std::optional<uint32_t> conventionOffset;

  /// For SSA spills: the specific uses that cross a suspend point.
  /// Empty for non-spill fields.
  SmallVector<OpOperand *, 4> spillUses;
};

//===----------------------------------------------------------------------===//
// CoroFrameAnalysis
//===----------------------------------------------------------------------===//

/// Shared analysis result for a coroutine's frame — the set of fields that
/// need frame storage, partitioned into static and dynamic regions.
///
/// This is computed once per async function and reused for all subsequent
/// operations (packing, access, descriptor generation).
struct CoroFrameAnalysis {
  /// Fields with constant size (placed in the static region).
  SmallVector<FrameFieldInfo> staticFields;

  /// Fields with per-PE size (placed in the dynamic region).
  /// Empty until Step 5 is implemented.
  SmallVector<FrameFieldInfo> dynamicFields;

  /// Whether any field has per-PE sizing. When false, the dynamic region
  /// is empty and no descriptor is needed.
  bool hasDynamicRegion = false;

  /// Grid dimensions from PerPEArrayType (only valid if hasDynamicRegion).
  int64_t gridWidth = 0, gridHeight = 0;
  int64_t offsetX = 0, offsetY = 0;

  /// Find the static field index for a given SSA value.
  /// Returns nullopt if no field stores this value.
  std::optional<unsigned> findStaticFieldByValue(Value value) const {
    for (auto [idx, field] : llvm::enumerate(staticFields)) {
      if (field.value == value)
        return idx;
    }
    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// CoroFrameLayout
//===----------------------------------------------------------------------===//

/// Complete frame layout including computed byte offsets for all fields.
///
/// For the static region, offsets are compile-time constants baked into code.
/// For the dynamic region (Step 5+), offsets are per-PE and stored in a
/// descriptor struct.
struct CoroFrameLayout {
  /// A placed frame field: byte offset and storage type.
  struct Slot {
    uint32_t offset = 0;
    Type storageType;
  };

  /// Static region slots, indexed by field position in
  /// CoroFrameAnalysis::staticFields.
  SmallVector<Slot> staticSlots;

  /// Total size of the static region in bytes.
  uint32_t staticSize = 0;

  /// Maximum total frame size across all PEs.
  /// For Step 4 (no dynamic region), this equals staticSize.
  uint32_t maxTotalSize = 0;
};

//===----------------------------------------------------------------------===//
// CoroFrameBuilder
//===----------------------------------------------------------------------===//

/// Analyzes an async function to determine its frame fields, computes
/// liveness, runs bin-packing, and produces a complete frame layout.
///
/// Usage:
///   CoroFrameBuilder builder(funcOp);
///   auto analysis = builder.analyze();     // → CoroFrameAnalysis
///   auto layout = builder.packLayout();    // → CoroFrameLayout
///
/// The builder also integrates rematerialization: before collecting spill
/// candidates, it clones cheap ops (constants, pure ops) near their uses
/// to reduce the number of values that cross suspend points.
/// Bundles the analysis and layout results for one async function so they
/// can be looked up by callers that need to create callee frames.
struct AsyncFuncInfo {
  CoroFrameAnalysis analysis;
  CoroFrameLayout layout;
};

class CoroFrameBuilder {
public:
  /// Construct a builder for the given async function.
  ///
  /// \p calleeInfos provides frame layout information for already-analyzed
  /// async callees. Required for embedding callee frames in the caller's
  /// frame. Callers must be analyzed after their callees (topological order).
  /// This ordering is always possible because transitive recursion is
  /// forbidden — a requirement shared with the frame-in-frame embedding
  /// design (a recursive callee frame would be infinitely sized).
  /// \p slotAlign controls minimum alignment for convention offsets during
  /// analysis (1 = byte, 2 = word). Must match the value used by
  /// CoroPrepareCFPass.
  explicit CoroFrameBuilder(
      FuncOp funcOp,
      const DenseMap<FuncOp, std::unique_ptr<AsyncFuncInfo>> *calleeInfos =
          nullptr,
      unsigned slotAlign = 1);

  /// Run the analysis: collect suspend points, rematerialize cheap ops,
  /// collect frame fields, compute liveness, partition into regions.
  ///
  /// Returns failure() if the function cannot be analyzed (e.g., async task
  /// with parameters).
  LogicalResult analyze();

  /// Pack the analyzed fields into a byte layout. Must be called after
  /// analyze(). Runs liveness-aware bin packing for the static region.
  ///
  /// \p slotAlign is the minimum alignment for all frame fields (1 or 2).
  /// With slotAlign=2, all offsets are even and the total size is even,
  /// enabling [*]u16 word-pointer access.
  ///
  /// Returns the complete frame layout with byte offsets, or failure if
  /// any field has a type whose size cannot be determined statically.
  FailureOr<CoroFrameLayout> packLayout(unsigned slotAlign = 1);

  /// Get the analysis result. Valid after analyze() succeeds.
  const CoroFrameAnalysis &getAnalysis() const { return analysis; }

  /// Get the analysis result (mutable, for the pass to move ownership).
  CoroFrameAnalysis &getAnalysis() { return analysis; }

private:
  /// Collect all values that need frame storage and build FrameFieldInfos.
  void collectFrameFields();

  /// Perform rematerialization: clone cheap ops near their uses to reduce
  /// the number of values that cross suspend points.
  void rematerialize();

  /// Compute block-level liveness bitsets for all frame fields.
  void computeLiveness();

  /// Compute natural alignment for a CSL type.
  static uint32_t getAlignmentForType(Type type);

  FuncOp funcOp;
  MLIRContext *ctx;
  const DenseMap<FuncOp, std::unique_ptr<AsyncFuncInfo>> *calleeInfos;
  unsigned slotAlign;
  CoroFrameAnalysis analysis;
};

//===----------------------------------------------------------------------===//
// CoroutineFrameAccess
//===----------------------------------------------------------------------===//

/// Emits IR for accessing fields of a coroutine frame via pointer
/// arithmetic on a `[*]u8` or `[*]u16` frame pointer (configurable via
/// slotAlign).
///
/// All constructors converge to a frame element pointer (`[*]u8` or
/// `[*]u16`) internally. Access goes through `ptr_index_access` with
/// element offsets (byte offsets / slotAlign).
///
/// The access pattern emits this IR (for a field of type T at elem offset N):
///
///   %elem_lval = ptr_index_access %ptr[%N]
///   %elem_ptr = address_of %elem_lval
///   %typed_ptr = ptrcast(%elem_ptr → *T)
///   %lval = dereference %typed_ptr
///
/// Which the emitter outputs as CSL:
///   @ptrcast(*T, &ptr[N]).*
class CoroutineFrameAccess {
public:
  /// Factory that emits (or returns) the frame pointer at the
  /// builder's current insertion point. Called once per block, cached.
  using FramePtrFactory = std::function<Value(OpBuilder &, Location)>;

  /// Construct from a pointer to the frame. If the pointer is not already
  /// the frame element pointer type, emits a ptrcast. The factory always
  /// returns this same pointer (no per-block re-derivation).
  CoroutineFrameAccess(OpBuilder &b, Location loc,
                       const CoroFrameLayout &layout,
                       TypedValue<csl::PointerType> framePtr,
                       unsigned slotAlign);

  /// Construct for an async function's own frame. The factory re-derives
  /// the frame pointer per block: load + ptrcast for fns, global reference +
  /// address_of + ptrcast for tasks.
  CoroutineFrameAccess(OpBuilder &b, Location loc,
                       const CoroFrameLayout &layout, FuncOp funcOp,
                       unsigned slotAlign);

  /// Construct for a callee's frame stored in a local variable.
  /// Takes the address of the variable and ptrcasts to the frame element
  /// pointer type. The factory always returns this same pointer.
  CoroutineFrameAccess(OpBuilder &b, Location loc,
                       const CoroFrameLayout &layout, VariableOp varOp,
                       unsigned slotAlign);

  /// Emit IR to get an lvalue to a frame field by its slot index.
  /// If \p isConst is true, emits a ConstCastOp to return a const lvalue.
  Value getSlotLValue(OpBuilder &b, Location loc, unsigned fieldIdx,
                      bool isConst = false);

  /// Emit IR to load a value from a frame field.
  Value loadSlot(OpBuilder &b, Location loc, unsigned fieldIdx);

  /// Emit IR to store a value to a frame field.
  void storeSlot(OpBuilder &b, Location loc, unsigned fieldIdx, Value data);

  /// Convenience: get lvalue for the state field (always index 0).
  Value getStateLValue(OpBuilder &b, Location loc) {
    return getSlotLValue(b, loc, /*fieldIdx=*/0);
  }

  /// Convenience: load the state field.
  Value loadState(OpBuilder &b, Location loc) {
    return loadSlot(b, loc, /*fieldIdx=*/0);
  }

  /// Convenience: store to the state field.
  void storeState(OpBuilder &b, Location loc, Value stateVal) {
    storeSlot(b, loc, /*fieldIdx=*/0, stateVal);
  }

  /// Ensure the frame pointer chain exists at the top of the given
  /// block and position the builder after it. Use before inserting
  /// ops at the "top" of a block (e.g., reloads) that must come after
  /// the frame pointer derivation.
  void positionAfterFramePtr(OpBuilder &b, Location loc, Block *block);

  /// Emit ptrcast to the frame element pointer type ([*]u8 or [*]u16)
  /// if needed, otherwise return the pointer as-is.
  static Value ensureFramePtr(OpBuilder &b, Location loc,
                              TypedValue<csl::PointerType> ptr,
                              unsigned slotAlign);

  unsigned getSlotAlign() const { return slotAlign; }

private:
  /// Get the frame pointer for the builder's current insertion block.
  /// Calls the factory and caches the result on first access per block.
  Value getOrCreateFramePtr(OpBuilder &b, Location loc);

  /// Emit IR to get an element-level lvalue at a given element offset.
  Value getElementLValueAtOffset(OpBuilder &b, Location loc, Value offsetVal);

  /// Emit IR to cast an element pointer to a typed pointer and dereference.
  Value castAndDeref(OpBuilder &b, Location loc, Value elemPtr,
                     Type targetType);

  const CoroFrameLayout &layout;
  unsigned slotAlign;
  FramePtrFactory framePtrFactory;
  DenseMap<Block *, Value> framePtrPerBlock;
  MLIRContext *ctx;
};

} // namespace typed_csl
} // namespace mlir

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_COROFRAMEBUILDER_H
