//===- csl-lex-roundtrip.cpp - Lexer roundtrip verification tool ----------===//
//
// Lexes a CSL source file using Parser3's hand-written lexer, then
// reconstructs the source from tokens + trivia and writes it to stdout.
// Use `diff` to verify byte-identical roundtrip.
//
// With --dump-tokens, prints an annotated token stream instead.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Target/CSL/Parser3/Lexer.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir::csl::parser3;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<bool> dumpTokens("dump-tokens",
                                cl::desc("Print annotated token stream"),
                                cl::init(false));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "CSL lexer roundtrip tool\n");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    errs() << "error: could not open input file '" << inputFilename
           << "': " << ec.message() << "\n";
    return 1;
  }

  StringRef source = fileOrErr.get()->getBuffer();

  Lexer lexer(source);
  TokenStream stream = lexer.lexAll();

  if (dumpTokens) {
    stream.dump(outs());
    return 0;
  }

  // Default: reconstruct source from tokens + trivia and write to stdout.
  stream.reconstruct(outs());
  return 0;
}
