//===- DwarfInlineReader.h - DWARF inlining facts from a Cerebras ELF
//*-C++-*-===//
//
// Reads `DW_TAG_subprogram` and `DW_TAG_inlined_subroutine` entries from
// a linked PE ELF and answers: "did this function survive linking out of
// line, or was it fully inlined?"  Required for the fiber stack-sizing
// resolver to distinguish "F was inlined away → its frame is rolled into
// the inlining caller's `.stack_sizes` entry" from "F is missing → bug".
//
// A finer per-site inlining map (caller → callee × call_file/line/col)
// would let the resolver tell, for a given source call F→g, whether THIS
// particular site got inlined or stayed as an out-of-line call.  That
// requires the EmitCSL to emit call-site location metadata, which is not
// wired up yet.  Until it is, the resolver works at function granularity:
// either F survives everywhere or F is inlined everywhere.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_DWARFINLINEREADER_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_DWARFINLINEREADER_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::typed_csl {

/// Coarse inlining summary extracted from a PE ELF's DWARF.
struct DwarfInlineInfo {
  /// Every linkage name that appears as a `DW_TAG_subprogram` somewhere
  /// in the unit (i.e. survived linking out-of-line).
  llvm::StringSet<> survivingSubprograms;

  /// Every linkage name that appears as a `DW_TAG_inlined_subroutine`
  /// somewhere in the unit (i.e. at least one call site was inlined).
  llvm::StringSet<> seenInlined;

  /// Return true iff this name was *only* seen inlined and never as a
  /// surviving out-of-line subprogram.
  bool isInlinedOnly(llvm::StringRef linkageName) const {
    return seenInlined.contains(linkageName) &&
           !survivingSubprograms.contains(linkageName);
  }
};

/// Walk the ELF's DWARF and collect (surviving subprograms, inlined
/// subroutines).  Emits diagnostics to stderr and returns `failure` on
/// I/O / parse errors.
FailureOr<DwarfInlineInfo> readDwarfInlineInfo(llvm::StringRef elfPath);

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_DWARFINLINEREADER_H
