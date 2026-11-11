//===- CoroFrameBuilder.cpp - Coroutine frame layout and access -----------===//
//
// Implements the coroutine frame analysis, bin-packing layout algorithm,
// and byte-level frame access IR emission.
//
// === Analysis Flow ===
//
// The CoroFrameBuilder processes an async function through these steps:
//
//   1. Collect suspend points (SuspendBrOp, AwaitBrOp).
//   2. Rematerialize cheap ops: for values with cross-block uses,
//      if the defining op is pure and cheap (constants, side-effect-free),
//      clone it before each such use. This eliminates the need to spill
//      those values to the frame.
//   3. Collect FrameFieldInfos for all values that still need frame storage:
//      a. State field (u16, live everywhere, always at index 0)
//      b. Result field (async fn with non-void return)
//      c. Parameter fields (from CoroParamsOp)
//      d. VariableOps with cross-block uses
//      e. SSA values with cross-block uses
//   4. Compute block-level liveness BitVectors for all fields.
//   5. Partition fields into static (constant-size) and dynamic (per-PE).
//
// === Bin-Packing Algorithm ===
//
// The packing algorithm assigns byte offsets to fields using liveness-aware
// first-fit-decreasing placement:
//
//   Sort fields by alignment desc, then size desc.
//   For each field:
//     Try candidate offsets starting from 0 (aligned up to field alignment).
//     At each candidate offset:
//       Compute the byte range [offset, offset + size).
//       Check all already-placed fields: if any overlaps in BOTH byte range
//       AND liveness (BitVector AND is non-zero), skip this candidate.
//       If no conflict: place the field here.
//
//
// === Frame Access IR Emission ===
//
// All frame access goes through pointer arithmetic on a byte array:
//
//   // For a field of type T at constant byte offset N:
//   %frame_lval = constant #csl.global_ref<@frame>  // or use [*]u8 arg
//   %byte_lval  = array_index_access %frame_lval[%N]
//   %byte_ptr   = address_of %byte_lval    // → *u8
//   %type_const = constant #csl.type<*T>
//   %typed_ptr  = builtin_call ptrcast(%type_const, %byte_ptr)  // → *T
//   %lval       = dereference %typed_ptr   // → lvalue<mutable T>
//
// The CSL emitter produces: @ptrcast(*T, &frame[N]).*
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/Coroutine/CoroFrameBuilder.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "coro-frame-builder"

using namespace mlir;
using namespace mlir::typed_csl;

/// Returns true if a value's definition is in a different block than a use.
/// After the trampoline, every block becomes an independent switch case,
/// so any cross-block value must go through the frame.
static bool isCrossBlockUse(Value def, OpOperand &use) {
  Block *defBlock = def.getParentBlock();
  Block *useBlock = use.getOwner()->getBlock();
  return defBlock && useBlock && defBlock != useBlock;
}

//===----------------------------------------------------------------------===//
// Utility: Rematerialization
//===----------------------------------------------------------------------===//

/// Returns true if the given operation can be cheaply rematerialized
/// (cloned near its uses instead of spilled to the frame).
///
/// Rematerializable ops include:
///   - Constants (ConstantLike trait)
///   - Pure side-effect-free ops without regions (cheap to recompute)
static bool isRematerializable(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (isMemoryEffectFree(op) && op->getNumRegions() == 0)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Utility: Alignment
//===----------------------------------------------------------------------===//

uint32_t CoroFrameBuilder::getAlignmentForType(Type type) {
  return csl::getAlignmentForType(type);
}

//===----------------------------------------------------------------------===//
// CoroFrameBuilder
//===----------------------------------------------------------------------===//

CoroFrameBuilder::CoroFrameBuilder(
    FuncOp funcOp,
    const DenseMap<FuncOp, std::unique_ptr<AsyncFuncInfo>> *calleeInfos,
    unsigned slotAlign)
    : funcOp(funcOp), ctx(funcOp.getContext()), calleeInfos(calleeInfos),
      slotAlign(slotAlign) {}

LogicalResult CoroFrameBuilder::analyze() {
  // Step 1: Rematerialize cheap ops to reduce spill count.
  LLVM_DEBUG(llvm::dbgs() << "  Rematerializing cheap ops...\n");
  rematerialize();

  // Step 4: Collect frame fields.
  LLVM_DEBUG(llvm::dbgs() << "  Collecting frame fields...\n");
  collectFrameFields();

  // Step 5: Compute liveness bitsets.
  LLVM_DEBUG(llvm::dbgs() << "  Computing liveness...\n");
  computeLiveness();

  // Step 6: Partition into static/dynamic by isPerPE.
  // (For Step 4 of the plan, all fields are static.)
  // Fields are already in analysis.staticFields from collectFrameFields().

  LLVM_DEBUG({
    llvm::dbgs() << "CoroFrameAnalysis for " << funcOp.getSymName() << ":\n";
    llvm::dbgs() << "  Blocks: " << funcOp.getBody().getBlocks().size() << "\n";
    llvm::dbgs() << "  Static fields: " << analysis.staticFields.size() << "\n";
    llvm::dbgs() << "  Dynamic fields: " << analysis.dynamicFields.size()
                 << "\n";
    for (auto [i, field] : llvm::enumerate(analysis.staticFields)) {
      llvm::dbgs() << "  [" << i << "] " << field.kind << " '"
                   << field.debugName << "' : " << field.storageType
                   << " (align=" << field.alignment
                   << ", perPE=" << field.isPerPE;
      if (field.conventionOffset)
        llvm::dbgs() << ", convention@" << *field.conventionOffset;
      llvm::dbgs() << ")\n";
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Rematerialization
//===----------------------------------------------------------------------===//

void CoroFrameBuilder::rematerialize() {
// Use different DEBUG_TYPE for the rematerialization step because it can be
// very verbose.
#undef DEBUG_TYPE
#define DEBUG_TYPE "coro-frame-builder-remat"
  DominanceInfo domInfo(funcOp);

  // Iterate to a fixed point: cloning a value near its use might make
  // its operands no longer live across the suspend, enabling further
  // cloning. In practice, 2-3 iterations suffice.
  bool changed = true;
  unsigned iterCount = 0;
  while (changed) {
    changed = false;
    iterCount++;

    // Collect all values whose definitions are across a suspend from at
    // least one of their uses.
    SmallVector<std::pair<Value, OpOperand *>> candidates;

    funcOp.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (isCrossBlockUse(result, use)) {
            candidates.push_back({result, &use});
          }
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "    [iter " << iterCount << "] " << candidates.size()
               << " cross-block candidates\n");

    for (auto [val, use] : candidates) {
      Operation *defOp = val.getDefiningOp();
      if (!defOp || !isRematerializable(defOp)) {
        LLVM_DEBUG({
          llvm::dbgs() << "    [iter " << iterCount
                       << "] SKIP (not rematerializable): "
                       << (defOp ? defOp->getName().getStringRef()
                                 : "block-arg")
                       << " : " << val.getType();
          if (defOp) {
            llvm::dbgs() << " [ConstantLike="
                         << defOp->hasTrait<OpTrait::ConstantLike>()
                         << " MemFree=" << isMemoryEffectFree(defOp)
                         << " Regions=" << defOp->getNumRegions() << "]";
          }
          llvm::dbgs() << "\n";
        });
        continue;
      }

      Block *useBlock = use->getOwner()->getBlock();

      // Skip if def and use are in the same block. Cloning within the same
      // block doesn't help: the clone would also be in this block, and if the
      // block has a KillLoop (loop body containing a suspend), the clone's
      // result would also be flagged as crossing, leading to infinite cloning.
      if (defOp->getBlock() == useBlock)
        continue;

      // Check that all operands of the defining op are available at the
      // use point (i.e., they dominate the use block).
      bool allAvailable = true;
      for (Value operand : defOp->getOperands()) {
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
          if (!domInfo.dominates(blockArg.getOwner(), useBlock)) {
            allAvailable = false;
            break;
          }
        } else {
          Block *operandDefBlock = operand.getDefiningOp()->getBlock();
          if (!domInfo.dominates(operandDefBlock, useBlock)) {
            allAvailable = false;
            break;
          }
        }
      }
      if (!allAvailable)
        continue;

      // Clone the op before the use and replace this specific use.
      // Use the same result index as the original value (important for
      // multi-result ops like CoroParamsOp).
      OpBuilder builder(ctx);
      builder.setInsertionPoint(use->getOwner());
      Operation *clone = builder.clone(*defOp);
      unsigned resultIdx = mlir::cast<OpResult>(val).getResultNumber();
      Value clonedVal = clone->getResult(resultIdx);
      use->set(clonedVal);
      changed = true;

      LLVM_DEBUG(llvm::dbgs() << "    Rematerialized: " << *defOp
                              << " before use in block " << useBlock << "\n");
    }
  }
#undef DEBUG_TYPE
#define DEBUG_TYPE "coro-frame-builder"
}

//===----------------------------------------------------------------------===//
// Frame field collection
//===----------------------------------------------------------------------===//

void CoroFrameBuilder::collectFrameFields() {
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  FunctionType fnType = funcOp.getFunctionType();

  // ---- (a) State field: always at offset 0. ----
  {
    FrameFieldInfo stateField;
    stateField.kind = FieldKind::State;
    stateField.storageType = ui16Ty;
    stateField.alignment = 2;
    stateField.isPerPE = false;
    stateField.debugName = "state";
    stateField.conventionOffset = kCoroStateOffset;
    analysis.staticFields.push_back(std::move(stateField));
  }

  // ---- (b) Result field (async fn with non-void return). ----
  // Shares starting offset with params (offset 4+). Result is written
  // only after params are dead, so they safely overlap.
  uint32_t conventionCursor = kCoroConventionPrefixSize;
  if (auto asyncResultType = funcOp.getAsyncResultType();
      asyncResultType && !mlir::isa<csl::VoidType>(*asyncResultType)) {
    uint32_t resultOffset =
        coroConventionOffset(conventionCursor, *asyncResultType, slotAlign);
    FrameFieldInfo resultField;
    resultField.kind = FieldKind::Result;
    resultField.storageType = *asyncResultType;
    resultField.alignment = getAlignmentForType(*asyncResultType);
    resultField.isPerPE = csl::hasPerPELayout(*asyncResultType);
    resultField.debugName = "result";
    resultField.conventionOffset = resultOffset;
    if (resultField.isPerPE) {
      analysis.hasDynamicRegion = true;
      analysis.dynamicFields.push_back(std::move(resultField));
    } else {
      analysis.staticFields.push_back(std::move(resultField));
    }
  }

  // ---- (c) Parameter fields (from CoroParamsOp, async fn only). ----
  // The function signature has been rewritten to (*coroframe) -> void by
  // Pass 1. The original param types are recorded in CoroParamsOp results.
  if (!funcOp.getIsTask()) {
    CoroParamsOp paramsOp = nullptr;
    funcOp.walk([&](CoroParamsOp op) { paramsOp = op; });

    if (paramsOp) {
      uint32_t paramCursor = kCoroConventionPrefixSize;
      for (unsigned i = 0; i < paramsOp.getNumResults(); ++i) {
        auto lvalTy =
            mlir::cast<csl::LValueType>(paramsOp.getResult(i).getType());
        Type paramTy = lvalTy.getValueType();

        paramCursor = coroConventionOffset(paramCursor, paramTy, slotAlign);

        FrameFieldInfo paramField;
        paramField.kind = FieldKind::Param;
        paramField.storageType = paramTy;
        paramField.alignment = getAlignmentForType(paramTy);
        paramField.isPerPE = csl::hasPerPELayout(paramTy);
        paramField.debugName = ("param_" + Twine(i)).str();
        paramField.conventionOffset = paramCursor;
        if (paramField.isPerPE) {
          analysis.hasDynamicRegion = true;
          analysis.dynamicFields.push_back(std::move(paramField));
        } else {
          analysis.staticFields.push_back(std::move(paramField));
        }

        auto sizeOpt = csl::getSizeInBytes(paramTy);
        assert(sizeOpt && "param type must have known size");
        paramCursor += *sizeOpt;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "    Synthetic fields: state"
                          << (fnType.getNumResults() > 0 ? " + result" : "")
                          << ", " << fnType.getNumInputs() << " params\n");

  // ---- (d) VariableOps live across suspend points. ----
  unsigned varCounter = 0;
  funcOp.walk([&](VariableOp varOp) {
    Value result = varOp.getResult();
    // Check if any use of this VariableOp crosses a suspend point.
    bool needsFrame = false;
    for (OpOperand &use : result.getUses()) {
      if (isCrossBlockUse(result, use)) {
        needsFrame = true;
        break;
      }
    }
    if (!needsFrame)
      return;

    // Store the raw value type, not the lvalue type.
    Type valueTy = mlir::cast<csl::LValueType>(varOp.getType()).getValueType();

    FrameFieldInfo varField;
    varField.kind = FieldKind::Variable;
    varField.value = result;
    varField.storageType = valueTy;
    varField.alignment = getAlignmentForType(valueTy);
    varField.isPerPE = csl::hasPerPELayout(valueTy);
    varField.debugName =
        ("var_" + varOp.getName() + "_" + Twine(varCounter++)).str();

    LLVM_DEBUG(llvm::dbgs() << "    VariableOp needs frame: " << varOp.getName()
                            << " : " << valueTy << "\n");

    if (varField.isPerPE) {
      analysis.hasDynamicRegion = true;
      analysis.dynamicFields.push_back(std::move(varField));
    } else {
      analysis.staticFields.push_back(std::move(varField));
    }
  });

  // ---- (e) SSA values with uses across suspend points. ----
  //
  // For each value defined in the function, check if any of its uses
  // crosses a suspend point. If so, it needs a frame slot.
  //
  // We skip VariableOps (handled above) and values whose defining op
  // is rematerializable (handled by rematerialize()).
  unsigned ssaCounter = 0;
  llvm::SetVector<Value> spilledValues;

  funcOp.walk([&](Operation *op) {
    // Skip VariableOps — handled above.
    if (mlir::isa<VariableOp>(op))
      return;

    // CoroParamsOp results are handled as FieldKind::Param (section c).
    // Skip them here even when unused — the op must survive DCE to
    // reserve frame space for the caller's param stores.
    if (mlir::isa<CoroParamsOp>(op))
      return;

    for (Value result : op->getResults()) {
      // Check each use individually.
      bool needsSpill = false;
      SmallVector<OpOperand *, 4> crossingUses;

      for (OpOperand &use : result.getUses()) {
        if (isCrossBlockUse(result, use)) {
          needsSpill = true;
          crossingUses.push_back(&use);
        }
      }

      if (!needsSpill)
        continue;

      // Skip already-collected values (shouldn't happen, but guard).
      if (!spilledValues.insert(result))
        continue;

      Type ty = result.getType();
      assert(!mlir::isa<csl::LValueType>(ty) &&
             "SSA spill values must not be lvalue-typed; "
             "lvalues should be handled via VariableOps or block arg skipping");

      FrameFieldInfo ssaField;
      ssaField.kind = FieldKind::Spill;
      ssaField.value = result;
      ssaField.storageType = ty;
      ssaField.alignment = getAlignmentForType(ty);
      ssaField.isPerPE = csl::hasPerPELayout(ty);
      ssaField.debugName = ("live_" + Twine(ssaCounter++)).str();
      ssaField.spillUses = std::move(crossingUses);

      LLVM_DEBUG(llvm::dbgs()
                 << "    SSA spill: " << ssaField.debugName << " : " << ty
                 << " (" << ssaField.spillUses.size() << " crossing uses)\n");

      if (ssaField.isPerPE) {
        analysis.hasDynamicRegion = true;
        analysis.dynamicFields.push_back(std::move(ssaField));
      } else {
        analysis.staticFields.push_back(std::move(ssaField));
      }
    }
  });

  // Block arguments do not need spilling in coroutines:
  // - Entry block args are function parameters, re-supplied on every
  //   coroutine invocation (the coroframe ptr for async fns, or
  //   hardware-supplied params for tasks).
  // - Non-entry blocks have no block arguments in our coroutine IR.

  // ---- (f) Embedded callee frames (async calls within this function). ----
  //
  // Each call producing !csl.coroframe<...> means we need to embed the
  // callee's frame in our own frame. The callee's frame size is looked up
  // from calleeInfos (which requires callee-first analysis ordering).
  // These fields are always-live (no Value — liveness handled like state).
  if (calleeInfos) {
    auto ui8Ty = IntegerType::get(ctx, 8, IntegerType::Unsigned);
    csl::CompilationUnitOp unit =
        funcOp->getParentOfType<csl::CompilationUnitOp>();

    unsigned calleeCounter = 0;
    funcOp.walk([&](CallOp callOp) {
      if (!callOp.getResult() ||
          !mlir::isa<csl::CoroFrameType>(callOp.getResult().getType()))
        return;

      auto coroFrameTy =
          mlir::cast<csl::CoroFrameType>(callOp.getResult().getType());
      StringRef calleeName = coroFrameTy.getSymbolName().getValue();
      auto calleeFuncOp = unit.lookupSymbol<FuncOp>(calleeName);
      if (!calleeFuncOp)
        return;

      auto it = calleeInfos->find(calleeFuncOp);
      if (it == calleeInfos->end())
        return;

      uint32_t calleeFrameSize = it->second->layout.maxTotalSize;
      auto arrayTy =
          csl::ArrayType::get({static_cast<int64_t>(calleeFrameSize)}, ui8Ty);

      FrameFieldInfo calleeField;
      calleeField.kind = FieldKind::CalleeFrame;
      calleeField.value = callOp.getResult();
      calleeField.storageType = arrayTy;
      calleeField.alignment = 1; // byte array
      calleeField.isPerPE = false;
      calleeField.debugName =
          ("callee_frame_" + calleeName + "_" + Twine(calleeCounter++)).str();
      analysis.staticFields.push_back(std::move(calleeField));

      LLVM_DEBUG(llvm::dbgs() << "    Callee frame: " << calleeName << " ["
                              << calleeFrameSize << " bytes]\n");
    });
  }
}

//===----------------------------------------------------------------------===//
// Liveness computation
//===----------------------------------------------------------------------===//

void CoroFrameBuilder::computeLiveness() {
  // Build a block index for the bitsets.
  DenseMap<Block *, unsigned> blockIndex;
  unsigned numBlocks = 0;
  for (Block &block : funcOp.getBody())
    blockIndex[&block] = numBlocks++;

  auto computeFieldLiveness = [&](FrameFieldInfo &field) {
    field.liveness.resize(numBlocks);

    // State field is live everywhere.
    if (field.kind == FieldKind::State) {
      field.liveness.set();
      return;
    }

    // For fields with a Value (VariableOps, SSA spills): compute liveness
    // from def through all uses via BFS.
    if (field.value) {
      // Determine definition block.
      Block *defBlock = nullptr;
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(field.value))
        defBlock = blockArg.getOwner();
      else if (auto *defOp = field.value.getDefiningOp())
        defBlock = defOp->getBlock();

      if (!defBlock)
        return;

      // Collect all blocks where the value is used.
      llvm::SmallPtrSet<Block *, 8> useBlocks;
      for (OpOperand &use : field.value.getUses())
        useBlocks.insert(use.getOwner()->getBlock());

      // The value is live from its definition through all blocks between
      // the def and any use. We use a simple approach: BFS backwards from
      // use blocks to the def block, marking all blocks on the path.
      //
      // A block is live if it's on any path from the def to a use.
      // Simple approximation: mark def block, all use blocks, and all
      // blocks reachable between them.
      llvm::SmallPtrSet<Block *, 16> liveBlocks;
      SmallVector<Block *, 16> worklist;

      // Start from use blocks and walk predecessors back to def.
      for (Block *ub : useBlocks) {
        if (liveBlocks.insert(ub).second && ub != defBlock)
          worklist.push_back(ub);
      }

      while (!worklist.empty()) {
        Block *b = worklist.pop_back_val();
        for (Block *pred : b->getPredecessors()) {
          if (liveBlocks.insert(pred).second && pred != defBlock)
            worklist.push_back(pred);
        }
      }
      // Always include the def block.
      liveBlocks.insert(defBlock);

      // Set the bitset.
      for (Block *b : liveBlocks) {
        auto it = blockIndex.find(b);
        if (it != blockIndex.end())
          field.liveness.set(it->second);
      }
    } else {
      // Synthetic fields (result, params) without a Value:
      // Conservative approximation — mark as live everywhere.
      // (Result is live from store to function exit; params from entry to
      // last use. Full precision isn't critical for these few fields.)
      field.liveness.set();
    }
  };

  for (auto &field : analysis.staticFields)
    computeFieldLiveness(field);
  for (auto &field : analysis.dynamicFields)
    computeFieldLiveness(field);
}

//===----------------------------------------------------------------------===//
// Bin-packing layout
//===----------------------------------------------------------------------===//

FailureOr<CoroFrameLayout> CoroFrameBuilder::packLayout(unsigned slotAlign) {
  CoroFrameLayout layout;

  // --- Static region packing ---
  //
  // Sort fields by alignment (descending), then by size (descending).
  // This ensures larger, more-aligned fields are placed first, reducing
  // padding waste.
  auto &fields = analysis.staticFields;

  // Track placed fields for overlap checking.
  struct PlacedField {
    uint32_t offset;
    uint32_t size;
    unsigned fieldIdx;
  };
  SmallVector<PlacedField> placed;
  layout.staticSlots.resize(fields.size());
  uint32_t totalSize = 0;

  // Phase 1: Pre-place convention fields (state, result, params) at their
  // fixed offsets. These must match what the caller writes.
  for (unsigned i = 0; i < fields.size(); ++i) {
    auto &field = fields[i];
    if (!field.conventionOffset)
      continue;

    auto sizeOpt = csl::getSizeInBytes(field.storageType);
    if (!sizeOpt) {
      return funcOp->emitError("coroutine frame field '")
             << field.debugName << "' has type " << field.storageType
             << " with unknown static size — cannot pack into frame";
    }
    uint32_t fieldSize = *sizeOpt;
    uint32_t offset = *field.conventionOffset;

    layout.staticSlots[i] = {offset, field.storageType};
    placed.push_back({offset, fieldSize, i});
    totalSize = std::max(totalSize, offset + fieldSize);

    LLVM_DEBUG(llvm::dbgs() << "    Placed " << field.debugName << " @ "
                            << offset << " (size=" << fieldSize << ", align="
                            << field.alignment << ", convention)\n");
  }

  // Enforce minimum alignment for all fields when slotAlign > 1.
  // This ensures all offsets are multiples of slotAlign, enabling
  // word-pointer access patterns.
  if (slotAlign > 1) {
    for (auto &field : fields)
      field.alignment = std::max(field.alignment, slotAlign);
  }

  // Phase 2: Pack remaining fields using liveness-aware first-fit-decreasing.
  // Sort non-convention fields by alignment desc, then size desc.
  SmallVector<unsigned> sortedIndices;
  for (unsigned i = 0; i < fields.size(); ++i) {
    if (!fields[i].conventionOffset)
      sortedIndices.push_back(i);
  }

  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](unsigned a, unsigned b) {
              auto sizeA = csl::getSizeInBytes(fields[a].storageType);
              auto sizeB = csl::getSizeInBytes(fields[b].storageType);
              uint32_t sa = sizeA.value_or(0);
              uint32_t sb = sizeB.value_or(0);
              if (fields[a].alignment != fields[b].alignment)
                return fields[a].alignment > fields[b].alignment;
              return sa > sb;
            });

  for (unsigned sortedI : sortedIndices) {
    auto &field = fields[sortedI];
    auto sizeOpt = csl::getSizeInBytes(field.storageType);
    if (!sizeOpt) {
      return funcOp->emitError("coroutine frame field '")
             << field.debugName << "' has type " << field.storageType
             << " with unknown static size — cannot pack into frame";
    }
    uint32_t fieldSize = *sizeOpt;
    uint32_t fieldAlign = field.alignment;

    // Try candidate offsets starting from 0, aligned up.
    uint32_t candidateOffset = 0;
    bool placed_ok = false;

    while (!placed_ok) {
      candidateOffset = (candidateOffset + fieldAlign - 1) & ~(fieldAlign - 1);

      bool conflict = false;
      for (auto &pf : placed) {
        uint32_t pEnd = pf.offset + pf.size;
        uint32_t cEnd = candidateOffset + fieldSize;
        if (candidateOffset < pEnd && pf.offset < cEnd) {
          llvm::BitVector livenessAnd = field.liveness;
          livenessAnd &= fields[pf.fieldIdx].liveness;
          if (livenessAnd.any()) {
            conflict = true;
            candidateOffset = pEnd;
            break;
          }
        }
      }

      if (!conflict) {
        placed_ok = true;
        layout.staticSlots[sortedI] = {candidateOffset, field.storageType};
        placed.push_back({candidateOffset, fieldSize, sortedI});
        totalSize = std::max(totalSize, candidateOffset + fieldSize);

        LLVM_DEBUG(llvm::dbgs()
                   << "    Placed " << fields[sortedI].debugName << " @ "
                   << candidateOffset << " (size=" << fieldSize
                   << ", align=" << fields[sortedI].alignment << ")\n");
      }
    }
  }

  // Round up to slot alignment so the frame element count is exact.
  totalSize = (totalSize + slotAlign - 1) & ~(slotAlign - 1);
  layout.staticSize = totalSize;
  layout.maxTotalSize = totalSize; // No dynamic region yet.

  LLVM_DEBUG({
    llvm::dbgs() << "Frame layout for " << funcOp.getSymName() << ":\n";
    llvm::dbgs() << "  Static size: " << layout.staticSize << " bytes\n";
    for (auto [i, field] : llvm::enumerate(fields)) {
      auto size = csl::getSizeInBytes(field.storageType);
      llvm::dbgs() << "  [" << i << "] " << field.debugName << " @ offset "
                   << layout.staticSlots[i].offset
                   << " (size=" << (size ? std::to_string(*size) : "?")
                   << ", align=" << field.alignment << ")\n";
    }
  });

  return layout;
}

//===----------------------------------------------------------------------===//
// CoroutineFrameAccess
//===----------------------------------------------------------------------===//

Value CoroutineFrameAccess::ensureFramePtr(OpBuilder &b, Location loc,
                                           TypedValue<csl::PointerType> ptr,
                                           unsigned slotAlign) {
  auto ptrTy = ptr.getType();
  auto targetPtrTy = getCoroFramePtrType(ptr.getContext(), slotAlign);

  // Already the right type — nothing to do.
  if (ptrTy == targetPtrTy)
    return ptr;

  return CastOp::create(b, loc, targetPtrTy, ptr).getResult();
}

CoroutineFrameAccess::CoroutineFrameAccess(OpBuilder &b, Location loc,
                                           const CoroFrameLayout &layout,
                                           TypedValue<csl::PointerType> ptr,
                                           unsigned slotAlign)
    : layout(layout), slotAlign(slotAlign), ctx(ptr.getContext()) {
  Value framePtr = ensureFramePtr(b, loc, ptr, slotAlign);
  framePtrFactory = [framePtr](OpBuilder &, Location) { return framePtr; };
  framePtrPerBlock[b.getInsertionBlock()] = framePtr;
}

CoroutineFrameAccess::CoroutineFrameAccess(OpBuilder &b, Location loc,
                                           const CoroFrameLayout &layout,
                                           FuncOp funcOp, unsigned slotAlign)
    : layout(layout), slotAlign(slotAlign), ctx(funcOp.getContext()) {
  if (funcOp.getIsTask()) {
    // Task: global reference + address_of + ptrcast per block.
    auto globalName = (funcOp.getSymName() + "__coro_frame").str();
    auto unit = funcOp->getParentOfType<csl::CompilationUnitOp>();
    auto globalOp = unit.lookupSymbol<GlobalOp>(globalName);
    assert(globalOp && "task frame global not found — must be created before "
                       "constructing CoroutineFrameAccess");
    Type globalTy = globalOp.getType();
    StringRef symName = globalOp.getSymName();
    unsigned capturedAlign = slotAlign;
    framePtrFactory = [globalTy, symName,
                       capturedAlign](OpBuilder &b, Location loc) -> Value {
      auto frameLvalTy = csl::LValueType::get(globalTy, /*isConst=*/false);
      Value lval = createGlobalRef(b, loc, frameLvalTy, symName);
      auto ptrTy = csl::PointerType::get(globalTy, /*isConst=*/false,
                                         /*isArray=*/false);
      auto addr = mlir::cast<TypedValue<csl::PointerType>>(
          AddressOfOp::create(b, loc, ptrTy, lval).getResult());
      return ensureFramePtr(b, loc, addr, capturedAlign);
    };
  } else {
    // Fn: load from entry block arg + ptrcast per block.
    Value argLval = funcOp.getBody().front().getArgument(0);
    unsigned capturedAlign = slotAlign;
    framePtrFactory = [argLval, capturedAlign](OpBuilder &b,
                                               Location loc) -> Value {
      Value ptr = LoadOp::create(b, loc, argLval);
      return ensureFramePtr(
          b, loc, mlir::cast<TypedValue<csl::PointerType>>(ptr), capturedAlign);
    };
  }
  framePtrPerBlock[b.getInsertionBlock()] = framePtrFactory(b, loc);
}

CoroutineFrameAccess::CoroutineFrameAccess(OpBuilder &b, Location loc,
                                           const CoroFrameLayout &layout,
                                           VariableOp varOp, unsigned slotAlign)
    : layout(layout), slotAlign(slotAlign), ctx(varOp.getContext()) {
  Type valueTy = mlir::cast<csl::LValueType>(varOp.getType()).getValueType();
  auto ptrTy =
      csl::PointerType::get(valueTy, /*isConst=*/false, /*isArray=*/false);
  auto addr = mlir::cast<TypedValue<csl::PointerType>>(
      AddressOfOp::create(b, loc, ptrTy, varOp.getResult()).getResult());
  Value framePtr = ensureFramePtr(b, loc, addr, slotAlign);
  framePtrFactory = [framePtr](OpBuilder &, Location) { return framePtr; };
  framePtrPerBlock[b.getInsertionBlock()] = framePtr;
}

Value CoroutineFrameAccess::getOrCreateFramePtr(OpBuilder &b, Location loc) {
  Block *block = b.getInsertionBlock();
  auto it = framePtrPerBlock.find(block);
  if (it != framePtrPerBlock.end())
    return it->second;

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(block);
  Value ptr = framePtrFactory(b, loc);
  framePtrPerBlock[block] = ptr;
  return ptr;
}

void CoroutineFrameAccess::positionAfterFramePtr(OpBuilder &b, Location loc,
                                                 Block *block) {
  b.setInsertionPointToStart(block);
  Value ptr = getOrCreateFramePtr(b, loc);
  if (Operation *def = ptr.getDefiningOp())
    b.setInsertionPointAfter(def);
}

Value CoroutineFrameAccess::getElementLValueAtOffset(OpBuilder &b, Location loc,
                                                     Value offsetVal) {
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  auto mutableElemLvalTy = csl::LValueType::get(elemTy, /*isConst=*/false);
  Value ptr = getOrCreateFramePtr(b, loc);
  return PtrIndexAccessOp::create(b, loc, mutableElemLvalTy, ptr, offsetVal);
}

Value CoroutineFrameAccess::castAndDeref(OpBuilder &b, Location loc,
                                         Value elemPtr, Type targetType) {
  auto targetPtrTy =
      csl::PointerType::get(targetType, /*isConst=*/false, /*isArray=*/false);
  Value typedPtr = CastOp::create(b, loc, targetPtrTy, elemPtr).getResult();
  auto mutableLvalTy = csl::LValueType::get(targetType, /*isConst=*/false);
  return DereferenceOp::create(b, loc, mutableLvalTy, typedPtr);
}

Value CoroutineFrameAccess::getSlotLValue(OpBuilder &b, Location loc,
                                          unsigned fieldIdx, bool isConst) {
  auto &slot = layout.staticSlots[fieldIdx];
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  uint32_t elemOffset = byteToElementOffset(slot.offset, slotAlign);
  Value offsetVal =
      ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, elemOffset));
  Value elemLval = getElementLValueAtOffset(b, loc, offsetVal);
  Type elemTy = getCoroFrameElementType(ctx, slotAlign);
  auto elemPtrTy =
      csl::PointerType::get(elemTy, /*isConst=*/false, /*isArray=*/false);
  Value elemPtr = AddressOfOp::create(b, loc, elemPtrTy, elemLval);
  Value lval = castAndDeref(b, loc, elemPtr, slot.storageType);
  if (isConst) {
    auto constLvalTy = csl::LValueType::get(slot.storageType, /*isConst=*/true);
    lval = ConstCastOp::create(b, loc, constLvalTy, lval);
  }
  return lval;
}

Value CoroutineFrameAccess::loadSlot(OpBuilder &b, Location loc,
                                     unsigned fieldIdx) {
  Value lval = getSlotLValue(b, loc, fieldIdx);
  return LoadOp::create(b, loc, lval);
}

void CoroutineFrameAccess::storeSlot(OpBuilder &b, Location loc,
                                     unsigned fieldIdx, Value data) {
  Value lval = getSlotLValue(b, loc, fieldIdx);
  AssignOp::create(b, loc, lval, data);
}
