//===- DwarfInlineReader.cpp - DWARF inlining summary from a PE ELF -----===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/DwarfInlineReader.h"

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::typed_csl {

namespace {

/// Pick the best name for a function DIE — linkage name if present,
/// falling back to the plain DW_AT_name.  Returns an empty string if
/// neither is set.
static std::string dieLinkageOrName(const llvm::DWARFDie &die) {
  if (const char *linkage = die.getLinkageName())
    return std::string(linkage);
  if (const char *plain = die.getName(llvm::DINameKind::ShortName))
    return std::string(plain);
  return {};
}

/// For an inlined-subroutine DIE, recover the linkage name of the
/// abstract origin (the function that was inlined here).
static std::string inlinedAbstractOriginName(const llvm::DWARFDie &die) {
  llvm::DWARFDie origin =
      die.getAttributeValueAsReferencedDie(llvm::dwarf::DW_AT_abstract_origin);
  if (!origin)
    origin =
        die.getAttributeValueAsReferencedDie(llvm::dwarf::DW_AT_specification);
  if (!origin)
    return {};
  return dieLinkageOrName(origin);
}

static void walkDie(const llvm::DWARFDie &die, DwarfInlineInfo &info) {
  llvm::dwarf::Tag tag = die.getTag();
  if (tag == llvm::dwarf::DW_TAG_subprogram) {
    // Concrete out-of-line definitions have low_pc/high_pc.  Skip pure
    // declarations (no PC range) and inlined-instance abstract origins.
    bool isDeclaration = die.find(llvm::dwarf::DW_AT_declaration).has_value();
    bool isInlineInstance = die.find(llvm::dwarf::DW_AT_inline).has_value();
    bool hasAddress = die.find(llvm::dwarf::DW_AT_low_pc).has_value() ||
                      die.find(llvm::dwarf::DW_AT_ranges).has_value();
    if (!isDeclaration && !isInlineInstance && hasAddress) {
      std::string name = dieLinkageOrName(die);
      if (!name.empty())
        info.survivingSubprograms.insert(name);
    }
  } else if (tag == llvm::dwarf::DW_TAG_inlined_subroutine) {
    std::string origin = inlinedAbstractOriginName(die);
    if (!origin.empty())
      info.seenInlined.insert(origin);
  }

  for (llvm::DWARFDie child : die.children())
    walkDie(child, info);
}

} // namespace

FailureOr<DwarfInlineInfo> readDwarfInlineInfo(llvm::StringRef elfPath) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOrErr =
      llvm::MemoryBuffer::getFile(elfPath);
  if (std::error_code ec = bufOrErr.getError()) {
    llvm::errs() << "error: cannot open ELF '" << elfPath
                 << "' for DWARF: " << ec.message() << "\n";
    return failure();
  }
  std::unique_ptr<llvm::MemoryBuffer> buf = std::move(*bufOrErr);

  llvm::Expected<std::unique_ptr<llvm::object::Binary>> binOrErr =
      llvm::object::createBinary(buf->getMemBufferRef());
  if (!binOrErr) {
    llvm::errs() << "error: cannot parse '" << elfPath
                 << "' as a binary: " << llvm::toString(binOrErr.takeError())
                 << "\n";
    return failure();
  }
  auto *obj = llvm::dyn_cast<llvm::object::ObjectFile>(binOrErr->get());
  if (!obj) {
    llvm::errs() << "error: '" << elfPath << "' is not an object file\n";
    return failure();
  }

  std::unique_ptr<llvm::DWARFContext> ctx = llvm::DWARFContext::create(*obj);
  DwarfInlineInfo info;
  for (const std::unique_ptr<llvm::DWARFUnit> &cu : ctx->normal_units()) {
    llvm::DWARFDie unitDie = cu->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
    if (unitDie)
      walkDie(unitDie, info);
  }
  return info;
}

} // namespace mlir::typed_csl
