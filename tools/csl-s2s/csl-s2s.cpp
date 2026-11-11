//===- csl-s2s.cpp - CSL Source-to-Source Transformation Tool ------------===//
//
// This tool parses CSL source files, performs MLIR-based transformations,
// and emits CSL code.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Conversion/Passes.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Pipelines/Passes.h"
#include "csl-mlir/Target/CSL/EmitCSL/EmitCSL.h"
#include "csl-mlir/Target/CSL/ParseCSL.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Tool-specific command line options
//===----------------------------------------------------------------------===//

static cl::OptionCategory cslS2SCategory("csl-s2s Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::cat(cslS2SCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(cslS2SCategory));

static cl::opt<bool> emitAfterParsing(
    "emit-after-parsing",
    cl::desc("Emit parsed MLIR without running transformation passes"),
    cl::init(false), cl::cat(cslS2SCategory));

static cl::opt<bool>
    emitMLIR("emit-mlir",
             cl::desc("Emit MLIR output instead of CSL (default: emit CSL)"),
             cl::init(false), cl::cat(cslS2SCategory));

static cl::opt<bool>
    lspMode("lsp-mode", cl::desc("Enable LSP-style error-tolerant parsing"),
            cl::init(false), cl::cat(cslS2SCategory));

/// Configuration options for csl-s2s that require special initialization.
/// This is a singleton that manages command line options with callbacks.
struct CslS2SConfig {
  std::string splitInputFileFlag;
  bool verifyDiagnosticsFlag = false;

  CslS2SConfig() {
    static cl::opt<std::string, /*ExternalStorage=*/true> splitInputFile{
        "split-input-file",
        cl::ValueOptional,
        cl::callback([&](const std::string &str) {
          // Use default marker if flag was used without value
          if (str.empty())
            splitInputFile.setValue(kDefaultSplitMarker);
        }),
        cl::desc("Split the input file into chunks using the given or "
                 "default marker and process each chunk independently"),
        cl::location(splitInputFileFlag),
        cl::init(""),
        cl::cat(cslS2SCategory)};

    static cl::opt<bool, /*ExternalStorage=*/true> verifyDiagnostics{
        "verify-diagnostics",
        cl::desc("Check that emitted diagnostics match "
                 "expected-* patterns in source"),
        cl::location(verifyDiagnosticsFlag), cl::init(false),
        cl::cat(cslS2SCategory)};
  }
};

static CslS2SConfig &getConfig() {
  static CslS2SConfig config;
  return config;
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// Perform the core processing: parse CSL, run passes, emit output.
static LogicalResult
performActions(raw_ostream &os,
               const std::shared_ptr<csl::CslModuleResolver> &resolver,
               unsigned bufferID, unsigned lineOffset, MLIRContext *context,
               StringRef filename, const csl::CslFlags &flags) {
  // Parse the CSL source file
  mlir::OwningOpRef<mlir::ModuleOp> module(
      ModuleOp::create(UnknownLoc::get(context)));

  // Create CompilationUnitOp
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module->getBody());
  auto compilationUnit = csl::CompilationUnitOp::create(
      builder, FileLineColLoc::get(context, filename, 0, 0));

  // Parse into the CompilationUnitOp's region
  auto sourceMgr = resolver->getSourceMgr();
  auto paramBindings = flags.toBindings(context);
  if (failed(csl::parseCSLIntoRegion(
          sourceMgr, bufferID, compilationUnit.getRegion(), paramBindings,
          lineOffset, filename,
          /*currentModule=*/nullptr, lspMode, flags)) &&
      !lspMode) {
    return failure();
  }

  if (emitAfterParsing) {
    // Just emit the parsed MLIR without running transformation passes
    module->print(os);
    os << "\n";
    return success();
  }

  // Run transformation passes
  PassManager pm(context);
  pm.enableVerifier(flags.verifyBetweenPasses);
  if (flags.profilePasses)
    pm.enableTiming();

  // Add a nested pass manager for CompilationUnitOp
  OpPassManager &compilationUnitPM = pm.nest<csl::CompilationUnitOp>();

  // Build the pass pipeline using the centralized infrastructure
  // processLayoutFile is false for csl-s2s (PE programs)
  csl::buildCslPassPipeline(compilationUnitPM, flags, resolver,
                            /*processLayoutFile=*/false);

  (void)mlir::applyPassManagerCLOptions(pm);

  // Enable IR printing after each pass if requested.
  if (flags.printIRAfterAll == csl::PrintIRMode::All)
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
        /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false);

  // Run the pass manager
  if (failed(pm.run(module.get()))) {
    return failure();
  }

  // Emit output
  if (emitMLIR) {
    // Emit MLIR output
    module->print(os);
    os << "\n";
  } else {
    // Emit CSL output (default)
    if (failed(csl::emitCSL(compilationUnit, os))) {
      return failure();
    }
  }

  return success();
}

/// Process a single buffer (chunk or whole file).
/// Creates a new MLIRContext for each buffer, following mlir-opt's pattern.
static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              const llvm::MemoryBufferRef &sourceBuffer,
              const DialectRegistry &registry, StringRef filename,
              unsigned lineOffset, bool shouldVerifyDiagnostics,
              SourceMgrDiagnosticVerifierHandler *verifyHandler,
              const csl::CslFlags &flags) {
  // Set up module resolver for parsing
  auto resolver = std::make_shared<csl::CslModuleResolver>(filename, flags);

  // Add the chunk buffer - this is what we'll parse from
  unsigned chunkBufferID = resolver->addBuffer(std::move(ownedBuffer));

  // Create a fresh context for this buffer (following mlir-opt pattern)
  MLIRContext context(registry);
  auto threadPool = flags.configureContext(&context);
  context.loadAllAvailableDialects();

  // Register the verifier handler with this context if verifying
  if (verifyHandler)
    verifyHandler->registerInContext(&context);

  if (!shouldVerifyDiagnostics) {
    // Normal mode: use standard diagnostic handler
    SourceMgrDiagnosticHandler sourceMgrHandler(*resolver->getSourceMgr(),
                                                &context, llvm::errs());
    return performActions(os, resolver, chunkBufferID, lineOffset, &context,
                          filename, flags);
  }

  // Verify diagnostics mode: we don't care about pass success/failure,
  // only whether diagnostics match expectations
  (void)performActions(os, resolver, chunkBufferID, lineOffset, &context,
                       filename, flags);
  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register dialects
  DialectRegistry registry;
  registry.insert<parsed_csl::ParsedCSLDialect, typed_csl::TypedCSLDialect,
                  cf::ControlFlowDialect>();

  // Register comptime evaluation external models
  mlir::parsed_csl::registerComptimeModels(registry);

  // Register MLIR command-line options like --mlir-print-debuginfo
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerPassManagerCLOptions();

  // Register CSL command-line options
  csl::registerCslCLOptions();

  // Initialize config to register command line options before parsing
  CslS2SConfig &config = getConfig();

  // Hide MLIR/LLVM internal options from --help (still visible via
  // --help-hidden).
  const cl::OptionCategory *categories[] = {&cslS2SCategory,
                                            &csl::getCslOptionCategory()};
  cl::HideUnrelatedOptions(categories);

  // Splice any tokens from $CSL_FLAGS in just after argv[0].
  llvm::BumpPtrAllocator envArgAlloc;
  llvm::StringSaver envArgSaver(envArgAlloc);
  auto args = csl::addCommandLineOptionsFromEnv(argc, argv, envArgSaver);

  // Parse command line options
  cl::ParseCommandLineOptions(args.size(), args.data(),
                              "CSL source-to-source transformer\n");

  // Always enable emitting after parsing in LSP mode
  if (lspMode && !emitAfterParsing) {
    llvm::errs()
        << "Note: --lsp-mode forces --emit-after-parsing since transformation "
           "passes may not be robust to incomplete/erroneous input.\n";
    emitAfterParsing = true;
  }

  // Create CslFlags from parsed command-line options
  Expected<csl::CslFlags> flags = csl::CslFlags::fromCommandLine();
  if (Error err = flags.takeError()) {
    llvm::errs() << "Error parsing command-line options: "
                 << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  // Set up the input file
  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Set up the output file
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Set up a SourceMgr with the original buffer for:
  // 1. Computing line offsets for split-input-file support
  // 2. Verifying diagnostics if requested
  llvm::SourceMgr originalSourceMgr;
  originalSourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(inputFile->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/false),
      SMLoc());

  // Set up verifier handler if verifying diagnostics.
  // Following mlir-opt: create one verifier for the whole file, register it
  // with each chunk's context.
  MLIRContext verifierCtx(registry);
  std::unique_ptr<SourceMgrDiagnosticVerifierHandler> verifierHandler;
  if (config.verifyDiagnosticsFlag) {
    verifierHandler = std::make_unique<SourceMgrDiagnosticVerifierHandler>(
        originalSourceMgr, &verifierCtx, llvm::errs(),
        SourceMgrDiagnosticVerifierHandler::Level::All);
  }

  // Define the chunk processing function
  auto processChunk = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                          const llvm::MemoryBufferRef &sourceBuffer,
                          raw_ostream &os) -> LogicalResult {
    // Compute line offset for split-input-file support.
    // The chunk buffer is a non-owning view into the original buffer,
    // so we can use the original SourceMgr to find where the chunk starts.
    unsigned lineOffset = 0;
    const char *chunkStart = chunkBuffer->getBufferStart();
    const char *originalStart = sourceBuffer.getBufferStart();
    const char *originalEnd = sourceBuffer.getBufferEnd();
    if (chunkStart > originalStart && chunkStart <= originalEnd) {
      SMLoc chunkLoc = SMLoc::getFromPointer(chunkStart);
      // getLineAndColumn returns 1-based line, convert to 0-based offset
      lineOffset =
          originalSourceMgr.getLineAndColumn(chunkLoc, /*BufferID=*/1).first -
          1;
    }

    return processBuffer(os, std::move(chunkBuffer), sourceBuffer, registry,
                         inputFilename, lineOffset,
                         config.verifyDiagnosticsFlag, verifierHandler.get(),
                         *flags);
  };

  // Process the input, potentially splitting into chunks
  LogicalResult result = splitAndProcessBuffer(
      llvm::MemoryBuffer::getMemBuffer(inputFile->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/false),
      processChunk, output->os(), config.splitInputFileFlag);

  // Verify diagnostics if requested
  if (config.verifyDiagnosticsFlag && failed(verifierHandler->verify()))
    result = failure();

  if (failed(result))
    return 1;

  output->keep();
  return 0;
}
