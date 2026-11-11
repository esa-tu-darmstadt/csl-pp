//===- FiberSizeSource.cpp - Concrete size oracle implementations -------===//

#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizeSource.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

#define DEBUG_TYPE "fiber-sizing"

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

// Placeholders for functions that don't have an explicit `stack_size(N)`.
// Round-1 ELFs are never executed, only inspected for `.stack_sizes`, so
// any positive values that compile cleanly will do.
constexpr uint16_t kPlaceholderCalleeU16Words = 128; // = 256 B
constexpr uint32_t kPlaceholderRootU32Words = 250;   // = 1000 B

/// Round bytes → even u16-word count (= multiple of 4 bytes).  The chained
/// worker SPs derived by subtraction stay 32-bit aligned only if every
/// callee's word count is even.
static uint16_t bytesToEvenU16Words(uint64_t bytes) {
  uint64_t words = (bytes + 1) / 2;
  if (words & 1)
    ++words;
  if (words > std::numeric_limits<uint16_t>::max())
    words = std::numeric_limits<uint16_t>::max();
  return static_cast<uint16_t>(words);
}

static uint32_t bytesToU32Words(uint64_t bytes) {
  return static_cast<uint32_t>((bytes + 3) / 4);
}

} // namespace

uint16_t PlaceholderSizeSource::getCalleeStackU16Words(FuncOp callee) {
  if (std::optional<uint32_t> n = callee.getStackSize()) {
    uint16_t words = bytesToEvenU16Words(*n);
    LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: placeholder source: callee '"
                            << callee.getSymName() << "' → " << words
                            << " u16-words (= stack_size(" << *n << ") B)\n");
    return words;
  }
  LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: placeholder source: callee '"
                          << callee.getSymName() << "' → "
                          << kPlaceholderCalleeU16Words
                          << " u16-words (= placeholder "
                          << (kPlaceholderCalleeU16Words * 2) << " B)\n");
  return kPlaceholderCalleeU16Words;
}

uint32_t PlaceholderSizeSource::getRootStackU32Words(FuncOp root) {
  // The root task's initial arg-area takes the TOP `kArgAreaU16Slots`
  // u16 words of the stack; the function frame chain must fit BELOW.
  // For `stack_size(N)`, N is interpreted as "max bytes the function
  // chain reaches below sp_initial" — add the arg-area on top.
  constexpr uint32_t argAreaBytes = kArgAreaU16Slots * 2;
  if (std::optional<uint32_t> n = root.getStackSize()) {
    uint32_t words = bytesToU32Words(*n + argAreaBytes);
    LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: placeholder source: root '"
                            << root.getSymName() << "' → " << words
                            << " u32-words (= stack_size(" << *n << ") + "
                            << argAreaBytes << " B arg-area)\n");
    return words;
  }
  LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: placeholder source: root '"
                          << root.getSymName() << "' → "
                          << kPlaceholderRootU32Words
                          << " u32-words (= placeholder "
                          << (kPlaceholderRootU32Words * 4) << " B)\n");
  return kPlaceholderRootU32Words;
}

uint16_t ResolvedSizeSource::getCalleeStackU16Words(FuncOp callee) {
  auto it = sizes.calleeSizeWords.find(callee.getSymName());
  if (it == sizes.calleeSizeWords.end())
    llvm::report_fatal_error(
        llvm::Twine("stack size resolution is missing callee '") +
        callee.getSymName() + "'");
  LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: resolved source: callee '"
                          << callee.getSymName() << "' → " << it->second
                          << " u16-words (= " << (it->second * 2) << " B)\n");
  return it->second;
}

uint32_t ResolvedSizeSource::getRootStackU32Words(FuncOp root) {
  auto it = sizes.rootStackU32Words.find(root.getSymName());
  if (it == sizes.rootStackU32Words.end())
    llvm::report_fatal_error(
        llvm::Twine("stack size resolution is missing root task '") +
        root.getSymName() + "'");
  LLVM_DEBUG(llvm::dbgs() << "fiber-sizing: resolved source: root '"
                          << root.getSymName() << "' → " << it->second
                          << " u32-words (= " << (it->second * 4) << " B)\n");
  return it->second;
}
