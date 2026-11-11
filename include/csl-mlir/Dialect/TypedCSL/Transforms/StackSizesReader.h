//===- StackSizesReader.h - Read `.stack_sizes` from a Cerebras ELF
//-*-C++-*-===//
//
// Decodes the (Cerebras-dialect) `.stack_sizes` section emitted by the
// `-fstack-size-section` Clang flag.  Returns a map of `function linkage
// name → frame size in bytes`.  Required by the fiber stack-sizing resolver.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_STACKSIZESREADER_H
#define CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_STACKSIZESREADER_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::typed_csl {

/// Parse a per-PE ELF's `.stack_sizes` section and return a map of
/// `linkage name → frame size in bytes`.  The section format:
///
///   record := u16 little-endian byte_addr ; ULEB128 stack_size
///
/// SDK 2.10's linker stores the final function entry byte address directly.
/// Symbol names come from the ELF's `STT_FUNC` symbol table.
///
/// On failure (no section, malformed records, etc.) emits a diagnostic to
/// stderr and returns `failure`.
FailureOr<llvm::StringMap<uint32_t>>
readStackSizesFromElf(llvm::StringRef elfPath);

/// Walk every `*.elf` under `binDir` (and `binDir/../{west,east}/bin` if
/// present — peripheral PE blocks), parse their PT_LOAD segments, and
/// return a `fabric tile index → ELF path` map.  Each Cerebras ELF
/// encodes its target tile in the high bits of the segment's `p_paddr`:
///
///   tile_index = (p_paddr >> 40) & 0xFFFF
///              = fabric_y * fabric_width + fabric_x
///
/// A "shared" ELF (compiled once, mapped to multiple tiles) has one
/// PT_LOAD per target tile; each contributes one entry to the map.
/// Filenames are opaque — they are NOT `out_<x>_<y>.elf`.
///
/// On failure (binDir doesn't exist, no ELFs, malformed PT_LOAD)
/// emits a diagnostic and returns `failure`.
FailureOr<llvm::DenseMap<uint16_t, std::string>>
buildTileToElfMapping(llvm::StringRef binDir);

} // namespace mlir::typed_csl

#endif // CSL_MLIR_DIALECT_TYPEDCSL_TRANSFORMS_STACKSIZESREADER_H
