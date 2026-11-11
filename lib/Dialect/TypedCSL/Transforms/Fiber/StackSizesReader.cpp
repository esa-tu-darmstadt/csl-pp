//===- StackSizesReader.cpp - Read `.stack_sizes` from a Cerebras ELF ---===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/StackSizesReader.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::typed_csl {

namespace {

/// Map function entry byte addresses → linkage name from the ELF's
/// STT_FUNC symbols.  In SDK 2.10 the Cerebras `.stack_sizes` section stores
/// function entry byte addresses directly as little-endian u16 values.
static llvm::DenseMap<uint64_t, std::string>
buildAddrToSymbolMap(const llvm::object::ObjectFile &obj) {
  llvm::DenseMap<uint64_t, std::string> result;
  for (const llvm::object::SymbolRef &sym : obj.symbols()) {
    llvm::Expected<llvm::object::SymbolRef::Type> type = sym.getType();
    if (!type) {
      llvm::consumeError(type.takeError());
      continue;
    }
    if (*type != llvm::object::SymbolRef::ST_Function)
      continue;
    llvm::Expected<llvm::StringRef> name = sym.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      continue;
    }
    llvm::Expected<uint64_t> addr = sym.getAddress();
    if (!addr) {
      llvm::consumeError(addr.takeError());
      continue;
    }
    // First name wins on addr collision (multiple symbols at the same
    // address — e.g. weak aliases).  In practice the .stack_sizes entry
    // is one per address, so either name resolves the same frame.
    result.try_emplace(*addr, name->str());
  }
  return result;
}

} // namespace

FailureOr<llvm::StringMap<uint32_t>>
readStackSizesFromElf(llvm::StringRef elfPath) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOrErr =
      llvm::MemoryBuffer::getFile(elfPath);
  if (std::error_code ec = bufOrErr.getError()) {
    llvm::errs() << "error: cannot open ELF '" << elfPath
                 << "': " << ec.message() << "\n";
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

  // Find the `.stack_sizes` section.
  llvm::object::SectionRef stackSizesSection;
  bool foundSection = false;
  for (const llvm::object::SectionRef &s : obj->sections()) {
    llvm::Expected<llvm::StringRef> name = s.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      continue;
    }
    if (*name == ".stack_sizes") {
      stackSizesSection = s;
      foundSection = true;
      break;
    }
  }
  if (!foundSection) {
    llvm::errs() << "error: '" << elfPath
                 << "' has no `.stack_sizes` section, so the stack sizes "
                    "of async tasks cannot be measured: libinject_mllvm.so "
                    "did not add `-fstack-size-section` to the cslc compile. "
                    "This happens if the Clang in the Cerebras SDK is not "
                    "the one libinject_mllvm.so was written for; set "
                    "CSL_INJECT_MLLVM_VERBOSE=1 to see whether it is "
                    "active.\n";
    return failure();
  }

  llvm::Expected<llvm::StringRef> contentsOrErr =
      stackSizesSection.getContents();
  if (!contentsOrErr) {
    llvm::errs() << "error: cannot read `.stack_sizes` from '" << elfPath
                 << "': " << llvm::toString(contentsOrErr.takeError()) << "\n";
    return failure();
  }
  llvm::StringRef contents = *contentsOrErr;

  // Build symbol-address map up front.
  auto addrToName = buildAddrToSymbolMap(*obj);

  llvm::StringMap<uint32_t> sizes;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(contents.data());
  const uint8_t *end = p + contents.size();
  while (p < end) {
    if (p + 2 > end) {
      llvm::errs() << "error: truncated `.stack_sizes` record in '" << elfPath
                   << "'\n";
      return failure();
    }
    uint16_t byteAddr =
        llvm::support::endian::read<uint16_t, 1>(p, llvm::endianness::little);
    p += 2;
    unsigned n = 0;
    uint64_t stackBytes = llvm::decodeULEB128(p, &n, end);
    if (n == 0) {
      llvm::errs() << "error: malformed ULEB128 in `.stack_sizes` of '"
                   << elfPath << "'\n";
      return failure();
    }
    p += n;

    auto it = addrToName.find(byteAddr);
    if (it == addrToName.end()) {
      // .stack_sizes entry without a matching STT_FUNC.  Plausible for
      // anonymous helpers the linker stripped; skip with a note rather
      // than failing — the resolver will hard-error later if a captured
      // graph function ends up with no entry.
      continue;
    }
    sizes[it->second] = static_cast<uint32_t>(stackBytes);
  }

  return sizes;
}

namespace {

/// Walk one ELF's PT_LOAD segments and return the set of tile indices
/// it serves (one entry per PT_LOAD; "shared" ELFs have multiple).
/// `tile_index = (p_paddr >> 40) & 0xFFFF`.
template <class ELFT>
static LogicalResult
collectTileIndices(const llvm::object::ELFObjectFile<ELFT> &obj,
                   llvm::StringRef elfPath,
                   llvm::SmallVectorImpl<uint16_t> &out) {
  llvm::Expected<typename ELFT::PhdrRange> phdrsOrErr =
      obj.getELFFile().program_headers();
  if (!phdrsOrErr) {
    llvm::errs() << "error: cannot read program headers from '" << elfPath
                 << "': " << llvm::toString(phdrsOrErr.takeError()) << "\n";
    return failure();
  }
  for (const auto &phdr : *phdrsOrErr) {
    if (phdr.p_type != llvm::ELF::PT_LOAD)
      continue;
    out.push_back(static_cast<uint16_t>((phdr.p_paddr >> 40) & 0xFFFFu));
  }
  return success();
}

} // namespace

FailureOr<llvm::DenseMap<uint16_t, std::string>>
buildTileToElfMapping(llvm::StringRef binDir) {
  if (!llvm::sys::fs::is_directory(binDir)) {
    llvm::errs() << "error: bin directory '" << binDir << "' does not exist\n";
    return failure();
  }

  // Collect ELF dirs.  cslc produces `bin/`; for builds with east/west
  // peripheral PE blocks, sibling `west/bin/` and `east/bin/` show up too.
  llvm::SmallVector<llvm::SmallString<256>, 3> elfDirs;
  elfDirs.emplace_back(binDir);
  llvm::SmallString<256> parent(binDir);
  llvm::sys::path::remove_filename(parent);
  for (llvm::StringRef sub : {"west", "east"}) {
    llvm::SmallString<256> p(parent);
    llvm::sys::path::append(p, sub, "bin");
    if (llvm::sys::fs::is_directory(p))
      elfDirs.push_back(std::move(p));
  }

  llvm::DenseMap<uint16_t, std::string> mapping;
  bool sawAnyElf = false;
  for (const auto &dir : elfDirs) {
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(dir, ec), end; it != end && !ec;
         it.increment(ec)) {
      llvm::StringRef path = it->path();
      if (!path.ends_with(".elf"))
        continue;
      sawAnyElf = true;

      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOrErr =
          llvm::MemoryBuffer::getFile(path);
      if (std::error_code e = bufOrErr.getError()) {
        llvm::errs() << "error: cannot open ELF '" << path
                     << "': " << e.message() << "\n";
        return failure();
      }
      llvm::Expected<std::unique_ptr<llvm::object::Binary>> binOrErr =
          llvm::object::createBinary(bufOrErr->get()->getMemBufferRef());
      if (!binOrErr) {
        llvm::errs() << "error: cannot parse '" << path << "' as a binary: "
                     << llvm::toString(binOrErr.takeError()) << "\n";
        return failure();
      }

      llvm::SmallVector<uint16_t, 8> tileIndices;
      // Cerebras's backend emits 64-bit ELFs; the tile index sits in
      // bits 40–55 of `p_paddr`, which only fits in Elf64_Addr.  An
      // ELF32 here would mean a different toolchain than we expect.
      if (auto *e = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(
              binOrErr->get())) {
        if (failed(collectTileIndices(*e, path, tileIndices)))
          return failure();
      } else if (auto *e = llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(
                     binOrErr->get())) {
        if (failed(collectTileIndices(*e, path, tileIndices)))
          return failure();
      } else {
        llvm::errs() << "error: '" << path
                     << "' is not a 64-bit ELF; Cerebras-emitted ELFs "
                        "should be Elf64\n";
        return failure();
      }

      for (uint16_t idx : tileIndices) {
        auto [it2, inserted] = mapping.try_emplace(idx, path.str());
        if (!inserted && it2->second != path) {
          llvm::errs() << "error: tile index " << idx << " is claimed by both '"
                       << it2->second << "' and '" << path << "'\n";
          return failure();
        }
      }
    }
    if (ec) {
      llvm::errs() << "error: iterating '" << dir << "': " << ec.message()
                   << "\n";
      return failure();
    }
  }

  if (!sawAnyElf) {
    llvm::errs() << "error: no `*.elf` files found under '" << binDir << "'\n";
    return failure();
  }
  return mapping;
}

} // namespace mlir::typed_csl
