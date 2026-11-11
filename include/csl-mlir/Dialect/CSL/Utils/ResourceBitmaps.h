//===- ResourceBitmaps.h - Bitmap-based HW resource tracking ----*- C++ -*-===//
//
// Tracks availability of hardware resources (colors, task entries, queues)
// using u16 bitmaps.  Mirrors the runtime allocator in resources.csl so that
// comptime and runtime allocation share identical logic.
//
// ===== Resource model =====
//
// WSE-2 — unified task table
//   Task table entries 0-28 (29, 30 are system-reserved).
//   Entries 0-23 are **dual-use**: the same slot serves as a color (for data
//   task routing) AND as a local/data task entry.  Claiming a color, a local
//   task, or a data task in this range all consume the same slot.
//   Entries 24-28 are local-task-only (no color routing).
//
//   Because colors *are* task entries on WSE-2, we track them with a single
//   bitmap pair (`taskEntryFree`).  `colorFree` is unused (set to 0).
//
// WSE-3 — independent pools
//   Colors 0-23:        tracked by `colorFree` (independent of tasks).
//   Task entries 8-28:  tracked by `taskEntryFree` (bits 0-20, +8 offset).
//   Data tasks:         mapped to input queues 0-7 (`inputQueueFree`).
//   Colors and task entries are fully independent.
//
// Both architectures
//   Input queues 0-7:   `inputQueueFree`.
//   Output queues:      `outputQueueFree` (6 on WSE-2, 8 on WSE-3).
//   Control tasks 31-62: `ctrlTaskFree` (lo/hi, +31 offset).
//
//===----------------------------------------------------------------------===//

#ifndef CSL_UTILS_RESOURCE_BITMAPS_H
#define CSL_UTILS_RESOURCE_BITMAPS_H

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include <cstdint>
#include <optional>

namespace mlir::csl {

struct ResourceBitmaps {
  // Colors 0-23.  WSE-3 only; set to 0 on WSE-2 (colors tracked via
  // taskEntryFree).
  uint16_t colorFreeLo = 0xFFFF; // colors 0-15
  uint16_t colorFreeHi = 0x00FF; // colors 16-23

  // Task table entries.
  //   WSE-2: entries 0-28 (also covers colors 0-23).
  //   WSE-3: entries 0-20 (mapped to IDs 8-28 via +8 offset).
  uint16_t taskEntryFreeLo = 0xFFFF;
  uint16_t taskEntryFreeHi = 0; // arch-dependent

  uint16_t inputQueueFree = 0xFF;   // 8 input queues
  uint16_t outputQueueFree = 0;     // arch-dependent (6 or 8)
  uint16_t ctrlTaskFreeLo = 0xFFFF; // control tasks 31-46
  uint16_t ctrlTaskFreeHi = 0xFFFF; // control tasks 47-62

  /// Initialize architecture-dependent fields.
  void init(TargetArch arch);

  /// Block resources reserved by the memcpy SDK module.
  void blockMemcpyResources(TargetArch arch);

  // --- Claim methods ---------------------------------------------------------
  // Without an ID: claim the lowest free entry (O(1) via CTZ).
  // With an ID: claim that specific entry, returning nullopt if already taken.

  /// Claim a color (0-23).
  /// WSE-2: claims from taskEntryFree (entries 0-23 are colors).
  /// WSE-3: claims from colorFree (independent of task entries).
  std::optional<uint16_t> claimColor(TargetArch arch,
                                     std::optional<uint16_t> id = std::nullopt);

  /// Claim a local task entry.
  /// WSE-2: 0-28 from taskEntryFree (0-23 also consume the color).
  /// WSE-3: 8-28 from taskEntryFree (bits 0-20, +8 offset).
  ///
  /// Without an explicit id, allocates the *highest* free entry, intentionally
  /// segregating from the runtime allocator (resources.csl) which hands out
  /// the lowest free entry.  This avoids a class of bugs where cslc inlines a
  /// comptime-bound task body at the slot's memory address (slot N+0..7),
  /// spilling into adjacent slot memory (slots N+1, N+2, ...).  If the runtime
  /// allocator hands out the same low slots that cslc inlined into, the
  /// runtime `bind_task()` patching would corrupt the comptime-bound task's
  /// body.  By placing comptime allocations at the back, overflow goes into
  /// memory beyond the bitmap-tracked range (or into already-blocked memcpy
  /// slots which cslc resolves on its own), keeping the low end clean for
  /// runtime binds.
  std::optional<uint16_t>
  claimLocalTask(TargetArch arch, std::optional<uint16_t> id = std::nullopt);

  /// Claim a data task.
  /// WSE-2: 0-23 from taskEntryFree (same slot as color + local task).
  /// WSE-3: 0-7 from inputQueueFree.
  ///
  /// Without an explicit id, allocates the highest free entry (same rationale
  /// as claimLocalTask).
  std::optional<uint16_t>
  claimDataTask(TargetArch arch, std::optional<uint16_t> id = std::nullopt);

  /// Claim an input queue (0-7).
  std::optional<uint16_t>
  claimInputQueue(std::optional<uint16_t> id = std::nullopt);

  /// Claim an output queue. WSE-2: 0-5, WSE-3: 0-7.
  std::optional<uint16_t>
  claimOutputQueue(std::optional<uint16_t> id = std::nullopt);

  /// Claim a control task. Returns absolute ID (31-62).
  ///
  /// Without an explicit id, allocates the highest free entry (same rationale
  /// as claimLocalTask).
  std::optional<uint16_t>
  claimControlTask(std::optional<uint16_t> id = std::nullopt);

  /// Mark the task that a @bind_*_task binds as taken, so the runtime
  /// allocator never hands it out.  `taskId` is a #csl.local_task_id,
  /// #csl.data_task_id or #csl.control_task_id.  Binding a task that is
  /// already taken, e.g. one claimed at comptime, changes nothing.
  void reserveBoundTask(TargetArch arch, Attribute taskId);

  /// Whether the runtime allocator may hand out the local task entry `id`.
  bool isLocalTaskFree(TargetArch arch, uint16_t id) const;

  /// The number of free resources of the kind whose ID type is `idType`
  /// (color, local_task_id, data_task_id, input_queue, output_queue or
  /// control_task_id), or nullopt if `idType` is none of these.
  std::optional<uint16_t> countFree(TargetArch arch, Type idType) const;

  // --- Release methods -------------------------------------------------------

  void releaseColor(uint16_t id, TargetArch arch);
  void releaseLocalTask(uint16_t id, TargetArch arch);
  void releaseDataTask(uint16_t id, TargetArch arch);
  void releaseInputQueue(uint16_t id);
  void releaseOutputQueue(uint16_t id);
  void releaseControlTask(uint16_t id); // takes absolute ID (31-62)

  // --- Serialization to/from CompilationUnitOp -------------------------------

  /// Store bitmap state as a DictionaryAttr on the CompilationUnitOp.
  void storeToUnit(CompilationUnitOp unit) const;

  /// Load bitmap state from a CompilationUnitOp's attribute.
  /// Returns default-initialized bitmaps if the attribute is not present.
  static ResourceBitmaps loadFromUnit(CompilationUnitOp unit, TargetArch arch);

  /// Remove the bitmap attribute from the CompilationUnitOp.
  static void removeFromUnit(CompilationUnitOp unit);

private:
  static std::optional<uint16_t> claim16(uint16_t &bm);
  static std::optional<uint16_t> claim16(uint16_t &bm, uint16_t idx);
  static std::optional<uint16_t> claim32(uint16_t &lo, uint16_t &hi);
  static std::optional<uint16_t> claim32(uint16_t &lo, uint16_t &hi,
                                         uint16_t idx);

  /// Claim the *highest* free bit in a u16 bitmap.  Used for task allocations
  /// (claimLocalTask, claimDataTask, claimControlTask) so comptime-allocated
  /// task IDs cluster at the high end of the range, while the runtime
  /// allocator (resources.csl) keeps the low end.  See claimLocalTask docs.
  static std::optional<uint16_t> claim16High(uint16_t &bm);
  static std::optional<uint16_t> claim32High(uint16_t &lo, uint16_t &hi);

  /// Claim from a u16-pair, restricted to indices < limit.
  static std::optional<uint16_t> claim32Ranged(uint16_t &lo, uint16_t &hi,
                                               uint16_t limit);

  static void release32(uint16_t &lo, uint16_t &hi, uint16_t idx);
  static void bitSet(uint16_t &bm, uint16_t idx);
  static void bitClear(uint16_t &bm, uint16_t idx);
};

} // namespace mlir::csl

#endif // CSL_UTILS_RESOURCE_BITMAPS_H
