//===- ResourceBitmaps.cpp - Bitmap-based HW resource tracking ------------===//
//
// See ResourceBitmaps.h for the resource model documentation.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/Utils/ResourceBitmaps.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h" // countr_zero

using namespace mlir::csl;

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------

void ResourceBitmaps::bitSet(uint16_t &bm, uint16_t idx) {
  bm |= uint16_t(1) << idx;
}

void ResourceBitmaps::bitClear(uint16_t &bm, uint16_t idx) {
  bm &= ~(uint16_t(1) << idx);
}

std::optional<uint16_t> ResourceBitmaps::claim16(uint16_t &bm) {
  if (bm == 0)
    return std::nullopt;
  uint16_t idx = llvm::countr_zero(static_cast<unsigned>(bm));
  bitClear(bm, idx);
  return idx;
}

std::optional<uint16_t> ResourceBitmaps::claim16(uint16_t &bm, uint16_t idx) {
  if (!(bm & (uint16_t(1) << idx)))
    return std::nullopt;
  bitClear(bm, idx);
  return idx;
}

std::optional<uint16_t> ResourceBitmaps::claim32(uint16_t &lo, uint16_t &hi) {
  if (auto idx = claim16(lo))
    return idx;
  if (auto idx = claim16(hi))
    return *idx + 16;
  return std::nullopt;
}

std::optional<uint16_t> ResourceBitmaps::claim16High(uint16_t &bm) {
  if (bm == 0)
    return std::nullopt;
  // Highest set bit in 16-bit value: 15 - leading zeros.
  uint16_t idx = 15 - llvm::countl_zero(bm);
  bitClear(bm, idx);
  return idx;
}

std::optional<uint16_t> ResourceBitmaps::claim32High(uint16_t &lo,
                                                     uint16_t &hi) {
  if (auto idx = claim16High(hi))
    return *idx + 16;
  if (auto idx = claim16High(lo))
    return idx;
  return std::nullopt;
}

std::optional<uint16_t> ResourceBitmaps::claim32(uint16_t &lo, uint16_t &hi,
                                                 uint16_t idx) {
  if (idx < 16)
    return claim16(lo, idx) ? std::optional<uint16_t>(idx) : std::nullopt;
  return claim16(hi, idx - 16) ? std::optional<uint16_t>(idx) : std::nullopt;
}

std::optional<uint16_t>
ResourceBitmaps::claim32Ranged(uint16_t &lo, uint16_t &hi, uint16_t limit) {
  // Search lo half, masking to bits < min(limit, 16).
  if (limit > 0) {
    uint16_t loMask = (limit >= 16) ? lo : (lo & ((uint16_t(1) << limit) - 1));
    if (loMask != 0) {
      uint16_t idx = llvm::countr_zero(static_cast<unsigned>(loMask));
      bitClear(lo, idx);
      return idx;
    }
  }
  // Search hi half if limit extends past 16.
  if (limit > 16) {
    uint16_t hiLimit = limit - 16;
    uint16_t hiMask = hi & ((uint16_t(1) << hiLimit) - 1);
    if (hiMask != 0) {
      uint16_t idx = llvm::countr_zero(static_cast<unsigned>(hiMask));
      bitClear(hi, idx);
      return idx + 16;
    }
  }
  return std::nullopt;
}

void ResourceBitmaps::release32(uint16_t &lo, uint16_t &hi, uint16_t idx) {
  if (idx < 16)
    bitSet(lo, idx);
  else
    bitSet(hi, idx - 16);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

void ResourceBitmaps::init(TargetArch arch) {
  taskEntryFreeHi = (arch == TargetArch::WSE2) ? 0x1FFF : 0x001F;
  outputQueueFree = (arch == TargetArch::WSE2) ? 0x3F : 0xFF;

  // WSE-2: colors are tracked via taskEntryFree, not colorFree.
  if (arch == TargetArch::WSE2) {
    colorFreeLo = 0;
    colorFreeHi = 0;
  }

  blockMemcpyResources(arch);
}

void ResourceBitmaps::blockMemcpyResources(TargetArch arch) {
  // Colors 20-23 (on WSE-2 this claims from taskEntryFree)
  for (uint16_t c : {20, 21, 22, 23})
    claimColor(arch, c);
  // Local task IDs bound by memcpy (verified in the task tables of SDK 2.10
  // builds for WSE-2 and WSE-3 with one and two channels).  The binds are
  // reserved when they are evaluated, too, but blocking them up front keeps
  // a comptime claim evaluated earlier from taking them.
  //   - 21 = sys_mod__sys_mod__memcpyd2h_mod__f_send_data
  //   - 27 = sys_mod__sys_mod__f_ublk_c22
  //   - 28 = sys_mod__sys_mod__sys_mod__f_sys_exit
  //   - 30 = sys_mod__sys_mod__memcpyd2h_mod__f_send_ctrl
  for (uint16_t t : {21, 27, 28, 30})
    claimLocalTask(arch, t);
  // Input queues 0, 1
  for (uint16_t q : {0, 1})
    claimInputQueue(q);
  // Output queues 0, 1
  for (uint16_t q : {0, 1})
    claimOutputQueue(q);
  // Control task IDs 33-37 bound by memcpy
  for (uint16_t ct : {33, 34, 35, 36, 37})
    claimControlTask(ct);
}

// ---------------------------------------------------------------------------
// Claim methods
// ---------------------------------------------------------------------------

std::optional<uint16_t>
ResourceBitmaps::claimColor(TargetArch arch, std::optional<uint16_t> id) {
  if (arch == TargetArch::WSE2) {
    // WSE-2: colors ARE task entries 0-23.  Claim from taskEntryFree.
    if (id) {
      assert(*id < 24 && "WSE-2 color must be in range 0-23");
      return claim32(taskEntryFreeLo, taskEntryFreeHi, *id);
    }
    return claim32Ranged(taskEntryFreeLo, taskEntryFreeHi, /*limit=*/24);
  }
  // WSE-3: colors are independent of task entries.
  if (id)
    return claim32(colorFreeLo, colorFreeHi, *id);
  return claim32(colorFreeLo, colorFreeHi);
}

std::optional<uint16_t>
ResourceBitmaps::claimLocalTask(TargetArch arch, std::optional<uint16_t> id) {
  if (id) {
    uint16_t bitmapIdx = (arch == TargetArch::WSE3) ? (*id - 8) : *id;
    if (!claim32(taskEntryFreeLo, taskEntryFreeHi, bitmapIdx))
      return std::nullopt;
    return *id;
  }
  // High-first: see claimLocalTask docstring in ResourceBitmaps.h.
  auto bitmapIdx = claim32High(taskEntryFreeLo, taskEntryFreeHi);
  if (!bitmapIdx)
    return std::nullopt;
  if (arch == TargetArch::WSE3)
    return *bitmapIdx + 8;
  return bitmapIdx;
}

std::optional<uint16_t>
ResourceBitmaps::claimDataTask(TargetArch arch, std::optional<uint16_t> id) {
  if (arch == TargetArch::WSE2) {
    // WSE-2: data task = same slot as color = task entry 0-23.
    if (id) {
      assert(*id < 24 && "WSE-2 data task must be in range 0-23");
      return claim32(taskEntryFreeLo, taskEntryFreeHi, *id);
    }
    return claim32Ranged(taskEntryFreeLo, taskEntryFreeHi, /*limit=*/24);
  }
  // WSE-3: data task = input queue.
  if (id)
    return claim16(inputQueueFree, *id) ? std::optional<uint16_t>(*id)
                                        : std::nullopt;
  return claim16(inputQueueFree);
}

std::optional<uint16_t>
ResourceBitmaps::claimInputQueue(std::optional<uint16_t> id) {
  if (id)
    return claim16(inputQueueFree, *id) ? std::optional<uint16_t>(*id)
                                        : std::nullopt;
  return claim16(inputQueueFree);
}

std::optional<uint16_t>
ResourceBitmaps::claimOutputQueue(std::optional<uint16_t> id) {
  if (id)
    return claim16(outputQueueFree, *id) ? std::optional<uint16_t>(*id)
                                         : std::nullopt;
  return claim16(outputQueueFree);
}

std::optional<uint16_t>
ResourceBitmaps::claimControlTask(std::optional<uint16_t> id) {
  if (id) {
    uint16_t bitmapIdx = *id - 31;
    if (!claim32(ctrlTaskFreeLo, ctrlTaskFreeHi, bitmapIdx))
      return std::nullopt;
    return *id;
  }
  // High-first: see claimLocalTask docstring in ResourceBitmaps.h.
  auto bitmapIdx = claim32High(ctrlTaskFreeLo, ctrlTaskFreeHi);
  if (!bitmapIdx)
    return std::nullopt;
  return *bitmapIdx + 31;
}

void ResourceBitmaps::reserveBoundTask(TargetArch arch, Attribute taskId) {
  if (auto local = mlir::dyn_cast<LocalTaskIdAttr>(taskId)) {
    uint16_t first = arch == TargetArch::WSE3 ? 8 : 0;
    if (local.getValue() >= first && local.getValue() <= 28)
      (void)claimLocalTask(arch, local.getValue());
  } else if (auto data = mlir::dyn_cast<DataTaskIdAttr>(taskId)) {
    if (auto color = mlir::dyn_cast<ColorAttr>(data.getSource())) {
      if (arch == TargetArch::WSE2 && color.getValue() < 24)
        (void)claimDataTask(arch, color.getValue());
    } else if (auto queue = mlir::dyn_cast<InputQueueAttr>(data.getSource())) {
      if (arch == TargetArch::WSE3 && queue.getValue() < 8)
        (void)claimDataTask(arch, queue.getValue());
    }
  } else if (auto control = mlir::dyn_cast<ControlTaskIdAttr>(taskId)) {
    if (control.getValue() >= 31 && control.getValue() <= 62)
      (void)claimControlTask(control.getValue());
  }
}

bool ResourceBitmaps::isLocalTaskFree(TargetArch arch, uint16_t id) const {
  uint16_t first = arch == TargetArch::WSE3 ? 8 : 0;
  if (id < first || id > 28)
    return false;
  uint16_t bitmapIdx = id - first;
  if (bitmapIdx < 16)
    return taskEntryFreeLo & (1u << bitmapIdx);
  return taskEntryFreeHi & (1u << (bitmapIdx - 16));
}

std::optional<uint16_t> ResourceBitmaps::countFree(TargetArch arch,
                                                   Type idType) const {
  auto count = [](uint16_t bits) {
    return static_cast<uint16_t>(llvm::popcount(bits));
  };
  // On WSE-2, colors and data tasks are the task entries 0-23.
  auto countTaskEntriesBelow24 = [&]() {
    return static_cast<uint16_t>(count(taskEntryFreeLo) +
                                 count(taskEntryFreeHi & 0x00FF));
  };

  if (mlir::isa<ColorType>(idType))
    return arch == TargetArch::WSE2
               ? countTaskEntriesBelow24()
               : static_cast<uint16_t>(count(colorFreeLo) +
                                       count(colorFreeHi & 0x00FF));
  if (mlir::isa<LocalTaskIdType>(idType))
    return static_cast<uint16_t>(count(taskEntryFreeLo) +
                                 count(taskEntryFreeHi));
  if (mlir::isa<DataTaskIdType>(idType))
    return arch == TargetArch::WSE2 ? countTaskEntriesBelow24()
                                    : count(inputQueueFree);
  if (mlir::isa<InputQueueType>(idType))
    return count(inputQueueFree);
  if (mlir::isa<OutputQueueType>(idType))
    return count(outputQueueFree);
  if (mlir::isa<ControlTaskIdType>(idType))
    return static_cast<uint16_t>(count(ctrlTaskFreeLo) +
                                 count(ctrlTaskFreeHi));
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Release methods
// ---------------------------------------------------------------------------

void ResourceBitmaps::releaseColor(uint16_t id, TargetArch arch) {
  if (arch == TargetArch::WSE2) {
    // WSE-2: colors are task entries.
    assert(id < 24 && "WSE-2 color must be in range 0-23");
    release32(taskEntryFreeLo, taskEntryFreeHi, id);
  } else {
    release32(colorFreeLo, colorFreeHi, id);
  }
}

void ResourceBitmaps::releaseLocalTask(uint16_t id, TargetArch arch) {
  if (arch == TargetArch::WSE3)
    id -= 8;
  release32(taskEntryFreeLo, taskEntryFreeHi, id);
}

void ResourceBitmaps::releaseDataTask(uint16_t id, TargetArch arch) {
  if (arch == TargetArch::WSE2) {
    // WSE-2: data task = task entry.
    assert(id < 24 && "WSE-2 data task must be in range 0-23");
    release32(taskEntryFreeLo, taskEntryFreeHi, id);
  } else {
    bitSet(inputQueueFree, id);
  }
}

void ResourceBitmaps::releaseInputQueue(uint16_t id) {
  bitSet(inputQueueFree, id);
}

void ResourceBitmaps::releaseOutputQueue(uint16_t id) {
  bitSet(outputQueueFree, id);
}

void ResourceBitmaps::releaseControlTask(uint16_t id) {
  release32(ctrlTaskFreeLo, ctrlTaskFreeHi, id - 31);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void ResourceBitmaps::storeToUnit(CompilationUnitOp unit) const {
  MLIRContext *ctx = unit->getContext();
  auto u16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  SmallVector<NamedAttribute> fields;
  auto named = [&](StringRef name, uint16_t val) {
    fields.push_back(NamedAttribute(StringAttr::get(ctx, name),
                                    IntegerAttr::get(u16Ty, val)));
  };
  named("color_free_lo", colorFreeLo);
  named("color_free_hi", colorFreeHi);
  named("task_entry_free_lo", taskEntryFreeLo);
  named("task_entry_free_hi", taskEntryFreeHi);
  named("input_queue_free", inputQueueFree);
  named("output_queue_free", outputQueueFree);
  named("ctrl_task_free_lo", ctrlTaskFreeLo);
  named("ctrl_task_free_hi", ctrlTaskFreeHi);
  unit->setAttr(CSLDialect::getComptimeResourceBitmapsAttrName(),
                DictionaryAttr::get(ctx, fields));
}

ResourceBitmaps ResourceBitmaps::loadFromUnit(CompilationUnitOp unit,
                                              TargetArch arch) {
  ResourceBitmaps bitmaps;
  bitmaps.init(arch);

  auto dictAttr = unit->getAttrOfType<DictionaryAttr>(
      CSLDialect::getComptimeResourceBitmapsAttrName());
  if (!dictAttr)
    return bitmaps;

  auto load = [&](StringRef name, uint16_t &field) {
    if (auto val = dictAttr.getAs<IntegerAttr>(name))
      field = static_cast<uint16_t>(val.getValue().getZExtValue());
  };
  load("color_free_lo", bitmaps.colorFreeLo);
  load("color_free_hi", bitmaps.colorFreeHi);
  load("task_entry_free_lo", bitmaps.taskEntryFreeLo);
  load("task_entry_free_hi", bitmaps.taskEntryFreeHi);
  load("input_queue_free", bitmaps.inputQueueFree);
  load("output_queue_free", bitmaps.outputQueueFree);
  load("ctrl_task_free_lo", bitmaps.ctrlTaskFreeLo);
  load("ctrl_task_free_hi", bitmaps.ctrlTaskFreeHi);
  return bitmaps;
}

void ResourceBitmaps::removeFromUnit(CompilationUnitOp unit) {
  unit->removeAttr(CSLDialect::getComptimeResourceBitmapsAttrName());
}
