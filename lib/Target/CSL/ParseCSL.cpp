//===- ParseCSL.cpp - CSL Parser Entry Point -----------------------------===//
//
// Lexes CSL source and parses it with the recursive-descent parser.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Target/CSL/ParseCSL.h"

#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"

#include "csl-mlir/Target/CSL/Parser3/Lexer.h"
#include "csl-mlir/Target/CSL/Parser3/Parser.h"

#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::csl::parser3;

LogicalResult csl::parseCSLIntoRegion(
    const std::shared_ptr<llvm::SourceMgr> &sourceMgr, unsigned bufferID,
    Region &region,
    const llvm::DenseMap<llvm::StringRef, ParamBinding> &paramBindings,
    unsigned lineOffset, llvm::StringRef sourceFilename,
    parsed_csl::ModuleOp currentModule, bool lspMode, const CslFlags &cslFlags,
    bool isSystemModule) {
  const llvm::MemoryBuffer *buffer = sourceMgr->getMemoryBuffer(bufferID);
  llvm::StringRef source = buffer->getBuffer();

  Lexer lexer(source);
  TokenStream tokens = lexer.lexAll();

  MLIRContext *context = region.getContext();
  Parser parser(*context, tokens, sourceMgr, bufferID, paramBindings,
                lineOffset, sourceFilename, currentModule, lspMode, cslFlags,
                isSystemModule, nullptr);

  parseContainerMembers(parser, region);
  if (parser.hadError())
    return failure();

  // The parser reads every expression that it uses as a value. Check that it
  // did, since an implicit read would happen at its user instead.
  for (Block &block : region)
    for (Operation &op : block)
      if (failed(parsed_csl::verifyExplicitReads(&op)))
        return failure();
  return success();
}
