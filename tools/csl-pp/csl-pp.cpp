//===- csl-pp.cpp - CSL Preprocessor ------------===//
//
// This tool preprocesses CSL programs, supporting an extended syntax
// (coroutines, variable-sized arrays, etc.) and emits standard CSL code.
// For layout files, it processes @set_tile_code calls, compiling PE programs
// with baked-in parameters.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Conversion/Passes.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/CSL/Utils/Version.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizeSource.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizingContext.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberStackResolver.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/StackSizesReader.h"
#include "csl-mlir/Pipelines/Passes.h"
#include "csl-mlir/Target/CSL/EmitCSL/EmitCSL.h"
#include "csl-mlir/Target/CSL/ParseCSL.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"

#include <unistd.h>

using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Tool-specific command line options
//===----------------------------------------------------------------------===//

static cl::OptionCategory cslPPCategory("csl-pp Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::cat(cslPPCategory));

static cl::opt<std::string>
    outputDir("o", cl::desc("Output directory (passed to cslc)"),
              cl::value_desc("directory"), cl::init("out"),
              cl::cat(cslPPCategory));

static cl::opt<bool>
    noCslc("no-cslc",
           cl::desc("Skip cslc invocation, only emit preprocessed files"),
           cl::init(false), cl::cat(cslPPCategory));

static cl::opt<bool> execCslc(
    "exec-cslc",
    cl::desc("Use exec() to replace this process with cslc after "
             "preprocessing. Frees all compiler memory before cslc runs and "
             "shows as 'cslc' in process listings. Useful for reducing peak "
             "memory when running many csl-pp instances in parallel."),
    cl::init(true), cl::cat(cslPPCategory));

static cl::opt<std::string>
    cslcPath("cslc", cl::desc("Path to cslc executable (default: search PATH)"),
             cl::value_desc("path"), cl::init(""), cl::cat(cslPPCategory));

static cl::opt<std::string> workDir(
    "work-dir",
    cl::desc("Working directory for preprocessed files (default: auto temp)"),
    cl::value_desc("directory"), cl::init(""), cl::cat(cslPPCategory));

// cslc-compatible options: captured here so LLVM's cl parser handles their
// values correctly, then forwarded verbatim to the downstream cslc.
static cl::opt<bool>
    enableMemcpy("memcpy", cl::desc("Add memcpy support (forwarded to cslc)"),
                 cl::init(false), cl::cat(cslPPCategory));

static cl::opt<unsigned>
    channels("channels",
             cl::desc("Number of memcpy I/O channels (forwarded to cslc)"),
             cl::init(0), cl::cat(cslPPCategory));

static cl::opt<unsigned>
    widthWestBuf("width-west-buf",
                 cl::desc("Width of west buffer (forwarded to cslc)"),
                 cl::init(0), cl::cat(cslPPCategory));

static cl::opt<unsigned>
    widthEastBuf("width-east-buf",
                 cl::desc("Width of east buffer (forwarded to cslc)"),
                 cl::init(0), cl::cat(cslPPCategory));

static cl::opt<bool> verbose("verbose",
                             cl::desc("Verbose output (forwarded to cslc)"),
                             cl::init(false), cl::cat(cslPPCategory));

static cl::list<std::string>
    cslcPassthroughArgs(cl::Sink, cl::desc("Arguments passed through to cslc"),
                        cl::cat(cslPPCategory));

//===----------------------------------------------------------------------===//
// cslc invocation helpers
//===----------------------------------------------------------------------===//

/// Find cslc executable from --cslc or PATH
static FailureOr<std::string> findCslc() {
  if (!cslcPath.empty()) {
    if (!llvm::sys::fs::exists(cslcPath)) {
      llvm::errs() << "error: cslc not found at '" << cslcPath << "'\n";
      return failure();
    }
    return cslcPath.getValue();
  }

  auto result = llvm::sys::findProgramByName("cslc");
  if (!result) {
    llvm::errs() << "error: cslc not found on PATH. "
                 << "Specify --cslc=<path> or use --no-cslc to skip.\n";
    return failure();
  }
  return *result;
}

/// The Cerebras SDK image next to the cslc script `cslc`, which runs it, if
/// there is exactly one.
static std::optional<std::string> findSdkImageNextTo(StringRef cslc) {
  llvm::SmallString<256> realCslc;
  if (llvm::sys::fs::real_path(cslc, realCslc))
    return std::nullopt;
  StringRef sdkDir = llvm::sys::path::parent_path(realCslc);

  std::optional<std::string> image;
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(sdkDir, ec), end;
       it != end && !ec; it.increment(ec)) {
    if (llvm::sys::path::extension(it->path()) != ".sif")
      continue;
    if (image)
      return std::nullopt;
    image = it->path();
  }
  return image;
}

/// Build the cslc argument list from flags and options.
static std::vector<std::string> buildCslcArgs(StringRef cslcExecutable,
                                              StringRef layoutFile,
                                              const csl::CslFlags &flags) {
  std::vector<std::string> argStrings;

  argStrings.push_back(cslcExecutable.str()); // argv[0]
  argStrings.push_back(layoutFile.str());     // positional: input file

  // -o output directory
  argStrings.push_back("-o");
  argStrings.push_back(outputDir.getValue());

  // Pass through import paths
  for (const auto &path : flags.importPaths) {
    argStrings.push_back("--import-path");
    argStrings.push_back(path);
  }

  // Pass through arch if specified
  argStrings.push_back("--arch=" +
                       csl::stringifyTargetArch(flags.targetArch).str());

  // Pass through fabric dimensions if specified
  if (flags.fabricDims) {
    argStrings.push_back(
        "--fabric-dims=" + std::to_string(flags.fabricDims.width) + "," +
        std::to_string(flags.fabricDims.height));
  }

  // Pass through fabric offsets if specified
  if (flags.fabricOffsets) {
    argStrings.push_back(
        "--fabric-offsets=" + std::to_string(flags.fabricOffsets.width) + "," +
        std::to_string(flags.fabricOffsets.height));
  }

  // Pass through params so cslc writes them into out.json
  for (const auto &[name, value] : flags.params.values) {
    switch (value.type) {
    case csl::ParamValue::Type::Int:
      argStrings.emplace_back("--params=" + name + ":" +
                              std::to_string(value.intVal));
      break;
    case csl::ParamValue::Type::Float:
      argStrings.emplace_back("--params=" + name + ":" +
                              std::to_string(value.floatVal));
      break;
    case csl::ParamValue::Type::Color:
      argStrings.emplace_back("--colors=" + name + ":" +
                              std::to_string(value.colorVal));
      break;
    }
  }

  // Pass through cslc-compatible options
  if (enableMemcpy)
    argStrings.push_back("--memcpy");
  if (channels > 0)
    argStrings.push_back("--channels=" + std::to_string(channels));
  if (widthWestBuf > 0)
    argStrings.push_back("--width-west-buf=" + std::to_string(widthWestBuf));
  if (widthEastBuf > 0)
    argStrings.push_back("--width-east-buf=" + std::to_string(widthEastBuf));
  if (verbose)
    argStrings.push_back("--verbose");

  // Always forward --fp16-format (defaults to f16)
  argStrings.push_back("--fp16-format=" +
                       csl::stringifyFp16Format(flags.fp16Format).str());

  // Pass through all sink arguments
  for (const auto &arg : cslcPassthroughArgs) {
    argStrings.push_back(arg);
  }

  return argStrings;
}

/// Print the cslc command line to stderr.
static void printCslcCommand(const std::vector<std::string> &args,
                             bool useExec) {
  if (!verbose)
    return;
  llvm::errs() << (useExec ? "Exec into cslc:" : "Invoking cslc:");
  for (const auto &arg : args)
    llvm::errs() << " " << arg;
  llvm::errs() << "\n";
}

/// Invoke cslc with preprocessed layout file (fork+exec, waits for
/// completion). argStrings[0] must be the cslc executable path. If
/// `outputLog` is given, cslc's output goes there and is printed only if cslc
/// fails.
static LogicalResult
invokeCslcCompiler(const std::vector<std::string> &argStrings,
                   std::optional<StringRef> outputLog = std::nullopt) {
  SmallVector<StringRef> args;
  for (const auto &s : argStrings)
    args.push_back(s);

  printCslcCommand(argStrings, /*useExec=*/false);

  SmallVector<std::optional<StringRef>, 3> redirects;
  if (outputLog)
    redirects = {std::nullopt, *outputLog, *outputLog};

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(
      argStrings[0], args, /*Env=*/std::nullopt, redirects,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (result != 0) {
    if (result < 0) {
      llvm::errs() << "error: failed to execute cslc: " << errMsg << "\n";
    }
    // cslc prints its own errors, just propagate failure
    if (outputLog) {
      if (llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> log =
              llvm::MemoryBuffer::getFile(*outputLog))
        llvm::errs() << (*log)->getBuffer();
    }
    return failure();
  }

  return success();
}

/// Replace this process with cslc using exec(). Does not return on success.
[[noreturn]] static void
execIntoCslc(const std::vector<std::string> &argStrings) {
  printCslcCommand(argStrings, /*useExec=*/true);

  // Flush C++ and C streams before exec replaces the process image.
  llvm::outs().flush();
  llvm::errs().flush();
  std::fflush(nullptr);

  // Build null-terminated C argv array.
  std::vector<const char *> argv;
  argv.reserve(argStrings.size() + 1);
  for (const auto &s : argStrings)
    argv.push_back(s.c_str());
  argv.push_back(nullptr);

  execvp(argv[0], const_cast<char *const *>(argv.data()));

  // Only reached if exec() failed.
  perror("error: failed to exec cslc");
  _exit(1);
}

//===----------------------------------------------------------------------===//
// Two-round fiber stack sizing
//===----------------------------------------------------------------------===//

/// Compute the absolute path to libinject_mllvm.so relative to csl-pp's
/// install layout (<install>/bin/csl-pp → <install>/lib/libinject_mllvm.so).
static std::string findInjectMllvmSo(const char *argv0) {
  static int probe = 0;
  llvm::SmallString<256> exePath(
      llvm::sys::fs::getMainExecutable(argv0, &probe));
  llvm::sys::path::remove_filename(exePath); // .../bin/
  llvm::sys::path::remove_filename(exePath); // .../
  llvm::sys::path::append(exePath, "lib", "libinject_mllvm.so");
  return std::string(exePath);
}

/// Append `val` (comma-separated) to environment variable `var`.  Used
/// for APPTAINER_BIND / SINGULARITY_BIND which take a comma-separated
/// bind list — preserving any existing entries the caller already set.
static void appendCommaEnv(const char *var, llvm::StringRef val) {
  const char *existing = getenv(var);
  std::string newVal;
  if (existing && *existing) {
    newVal = existing;
    newVal += ',';
  }
  newVal += val.str();
  setenv(var, newVal.c_str(), /*overwrite=*/1);
}

/// Set the env vars the SDK's singularity container needs so the
/// container-hosted `cslc-backend` can `LD_PRELOAD` our shim:
///
///   APPTAINERENV_LD_PRELOAD / SINGULARITYENV_LD_PRELOAD
///     Apptainer / Singularity translate these to LD_PRELOAD INSIDE
///     the container — that makes the container's ld.so load the shim.
///
///   APPTAINER_BIND / SINGULARITY_BIND
///     The shim file lives on the host filesystem and is not visible
///     inside the container by default.  We bind-mount the shim's
///     parent directory at the same path so the LD_PRELOAD value
///     above resolves.  Appended (comma-separated) to any pre-existing
///     bind list the caller may already have configured.
///
/// Both Apptainer- and Singularity-prefixed variants are set because
/// the SDK's installs can use either tool.  Idempotent.
static void enableInjectShim(llvm::StringRef shimPath) {
  setenv("APPTAINERENV_LD_PRELOAD", shimPath.str().c_str(), /*overwrite=*/1);
  setenv("SINGULARITYENV_LD_PRELOAD", shimPath.str().c_str(), /*overwrite=*/1);

  // Let the shim report whether it is active, if the user asked for that.
  if (const char *verbose = getenv("CSL_INJECT_MLLVM_VERBOSE")) {
    setenv("APPTAINERENV_CSL_INJECT_MLLVM_VERBOSE", verbose, /*overwrite=*/1);
    setenv("SINGULARITYENV_CSL_INJECT_MLLVM_VERBOSE", verbose,
           /*overwrite=*/1);
  }

  llvm::SmallString<256> shimDir(shimPath);
  llvm::sys::path::remove_filename(shimDir);
  appendCommaEnv("APPTAINER_BIND", shimDir);
  appendCommaEnv("SINGULARITY_BIND", shimDir);
}

/// Resolve fiber sizes for each entry, re-run stages 2 + 3 on the entry's
/// pre-fiber clone with a ResolvedSizeSource backed by the round-1 ELF,
/// and overwrite `entry.emittedCSLPath` with the re-emitted CSL.  The
/// next cslc invocation picks up the resolved sizes.
static LogicalResult
applyResolvedFiberSizes(MLIRContext &context, const csl::CslFlags &flags,
                        llvm::StringRef cslcOutputDir,
                        std::vector<typed_csl::FiberSizingEntry> &entries) {
  csl::CslPipelineOptions opts;
  opts.cslFlags = flags;
  opts.processLayoutFile = false;

  // Build the fabric tile_index → ELF path map once.  cslc filenames are
  // opaque (hash-derived), so we resolve the entry's recorded (x, y) via
  // the PT_LOAD `p_paddr` encoding (see ctf.py's `build_elf_mapping`):
  //   tile_index = (p_paddr >> 40) & 0xFFFF
  //              = fabric_y * fabric_width + fabric_x
  // A single shared ELF can serve multiple tiles via multiple PT_LOADs.
  llvm::SmallString<256> binDir(cslcOutputDir);
  llvm::sys::path::append(binDir, "bin");
  FailureOr<llvm::DenseMap<uint16_t, std::string>> mappingOrErr =
      typed_csl::buildTileToElfMapping(binDir);
  if (failed(mappingOrErr))
    return failure();
  const llvm::DenseMap<uint16_t, std::string> &tileToElf = *mappingOrErr;

  if (flags.fabricDims.width == 0) {
    llvm::errs() << "error: --fabric-dims width is zero; cannot decode "
                    "fabric tile indices for stack sizing\n";
    return failure();
  }

  for (auto &entry : entries) {
    if (entry.tileCoords.empty()) {
      llvm::errs() << "error: stack sizing entry for '" << entry.peSourceFile
                   << "' has no recorded tile coordinates\n";
      return failure();
    }
    auto [rectX, rectY] = entry.tileCoords.front();
    uint16_t fabricX = static_cast<uint16_t>(rectX) + flags.fabricOffsets.width;
    uint16_t fabricY =
        static_cast<uint16_t>(rectY) + flags.fabricOffsets.height;
    uint16_t tileIndex = fabricY * flags.fabricDims.width + fabricX;

    auto it = tileToElf.find(tileIndex);
    if (it == tileToElf.end()) {
      llvm::errs() << "error: no round-1 ELF claims fabric tile index "
                   << tileIndex << " (rect (" << rectX << "," << rectY
                   << ") + fabric_offsets (" << flags.fabricOffsets.width << ","
                   << flags.fabricOffsets.height << ") in "
                   << flags.fabricDims.width << "-wide fabric) for PE source '"
                   << entry.peSourceFile << "'\n";
      return failure();
    }
    llvm::StringRef elfPath = it->second;

    FailureOr<typed_csl::ResolvedSizes> resolvedOrErr =
        typed_csl::resolveFiberSizes(elfPath, entry.callGraph);
    if (failed(resolvedOrErr))
      return failure();

    typed_csl::ResolvedSizeSource source(*resolvedOrErr);

    PassManager pm(&context);
    pm.enableVerifier(flags.verifyBetweenPasses);
    if (flags.profilePasses)
      pm.enableTiming();
    OpPassManager &stagePM = pm.nest<csl::CompilationUnitOp>();
    csl::buildCslPassPipelineStage2(stagePM, opts, &source);
    csl::buildCslPassPipelineStage3(stagePM, opts);
    if (failed(pm.run(*entry.clone))) {
      llvm::errs() << "error: round-2 pipeline failed for '"
                   << entry.peSourceFile << "'\n";
      return failure();
    }

    // Find the inner CompilationUnitOp for emission.
    csl::CompilationUnitOp peUnit;
    entry.clone->walk([&](csl::CompilationUnitOp u) {
      peUnit = u;
      return WalkResult::interrupt();
    });
    if (!peUnit) {
      llvm::errs() << "error: round-2 clone of '" << entry.peSourceFile
                   << "' has no csl.compilation_unit\n";
      return failure();
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(entry.emittedCSLPath, ec);
    if (ec) {
      llvm::errs() << "error: cannot overwrite '" << entry.emittedCSLPath
                   << "': " << ec.message() << "\n";
      return failure();
    }
    if (failed(csl::emitCSL(peUnit, os))) {
      llvm::errs() << "error: round-2 emit failed for '" << entry.peSourceFile
                   << "'\n";
      return failure();
    }

    if (verbose)
      llvm::errs() << "csl-pp: round-2 re-emit for " << entry.peSourceFile
                   << " (rect tile (" << rectX << "," << rectY
                   << ") → fabric tile " << tileIndex << " → " << elfPath
                   << ") → " << entry.emittedCSLPath << "\n";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static LogicalResult
processLayoutFile(StringRef outputPath,
                  std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                  MLIRContext *context, const csl::CslFlags &flags,
                  StringRef peOutputDirectory,
                  typed_csl::FiberSizingContext *fiberSizingContext = nullptr) {
  // Create system module cache once (shared across all PE program
  // compilations).
  auto systemModuleCache = csl::SystemModuleCache::create(flags);

  // Build module resolver
  auto resolver = std::make_shared<csl::CslModuleResolver>(inputFilename, flags,
                                                           systemModuleCache);
  unsigned layoutBufferID = resolver->addBuffer(std::move(inputBuffer));
  auto sourceMgr = resolver->getSourceMgr();

  // Attach a diagnostic handler to the source manager
  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, context);

  // Create CompilationUnitOp
  OpBuilder builder(context);
  Location inputFileLoc =
      FileLineColRange::get(StringAttr::get(context, inputFilename));
  mlir::OwningOpRef<csl::CompilationUnitOp> unit(
      csl::CompilationUnitOp::create(builder, inputFileLoc));

  // Parse the layout file the CompilationUnitOp's region
  auto paramBindings = flags.toBindings(context);
  if (failed(csl::parseCSLIntoRegion(
          sourceMgr, layoutBufferID, unit->getRegion(), paramBindings,
          /*lineOffset=*/0, /*sourceFilename=*/"", /*currentModule=*/nullptr,
          /*lspMode=*/false, flags))) {
    return failure();
  }

  // Run transformation passes
  PassManager pm(context);
  pm.enableVerifier(flags.verifyBetweenPasses);
  if (flags.profilePasses)
    pm.enableTiming();

  // Build the pass pipeline using the centralized infrastructure
  // Enable layout file processing to handle @set_tile_code calls
  csl::buildCslPassPipeline(pm, flags, resolver,
                            /*processLayoutFile=*/true, peOutputDirectory,
                            systemModuleCache, fiberSizingContext);

  // Apply pass manager command-line options
  if (failed(mlir::applyPassManagerCLOptions(pm))) {
    return failure();
  }

  // Enable IR printing for layout if requested.
  if (flags.printIRAfterAll == csl::PrintIRMode::Layout ||
      flags.printIRAfterAll == csl::PrintIRMode::All)
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
        /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/true);

  // Run the pass manager
  if (failed(pm.run(*unit))) {
    llvm::errs() << "Failed to process layout file.\n";
    return failure();
  }

  // Write output to file
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    llvm::errs() << "error: cannot write to '" << outputPath
                 << "': " << ec.message() << "\n";
    return failure();
  }

  // csl-pp has substituted the values of --params into the tile programs,
  // but still passes them to cslc, which records them in out.json for the
  // host program. Declare them, so that cslc does not warn that they are
  // unused.
  bool declaredParams = false;
  for (const auto &[name, value] : flags.params.values) {
    if (value.type == csl::ParamValue::Type::Color)
      continue;
    if (!declaredParams)
      os << "// Set with --params; csl-pp has substituted their values.\n";
    os << "param " << name << ";\n";
    declaredParams = true;
  }
  if (declaredParams)
    os << "\n";

  if (failed(csl::emitCSL(*unit, os))) {
    return failure();
  }

  return success();
}

int main(int argc, char **argv) {
  // These survive past the scoped block for --exec-cslc mode.
  std::vector<std::string> cslcArgs;
  bool useExec = false;

  {
    InitLLVM y(argc, argv);

    // Register dialects
    DialectRegistry registry;
    registry.insert<parsed_csl::ParsedCSLDialect, typed_csl::TypedCSLDialect>();

    // Register comptime evaluation external models
    mlir::parsed_csl::registerComptimeModels(registry);

    // Register MLIR command-line options like --mlir-print-debuginfo
    registerMLIRContextCLOptions();
    registerAsmPrinterCLOptions();
    registerPassManagerCLOptions();

    // Register CSL command-line options
    csl::registerCslCLOptions();

    // Hide MLIR/LLVM internal options from --help (still visible via
    // --help-hidden).
    const cl::OptionCategory *categories[] = {&cslPPCategory,
                                              &csl::getCslOptionCategory()};
    cl::HideUnrelatedOptions(categories);

    // Splice any tokens from $CSL_FLAGS in just after argv[0] so the
    // env-var path goes through the same parse pipeline as the CLI.
    llvm::BumpPtrAllocator envArgAlloc;
    llvm::StringSaver envArgSaver(envArgAlloc);
    auto args = csl::addCommandLineOptionsFromEnv(argc, argv, envArgSaver);

    cl::SetVersionPrinter(
        [](raw_ostream &os) { csl::printVersion(os, "csl-pp"); });

    // Parse command line options
    cl::ParseCommandLineOptions(
        args.size(), args.data(),
        "CSL preprocessor - drop-in replacement for cslc\n\n"
        "Preprocesses CSL layout files with PE program inlining.\n"
        "By default invokes cslc after preprocessing.\n"
        "Use --no-cslc to only emit preprocessed files.\n");

    // Create CslFlags from parsed command-line options
    llvm::Expected<csl::CslFlags> flags = csl::CslFlags::fromCommandLine();
    if (llvm::Error err = flags.takeError()) {
      llvm::errs() << "Error parsing command-line options: "
                 << llvm::toString(std::move(err)) << "\n";
      return 1;
    }

    // An installed csl-pp finds the csl++ libraries without --import-path.
    flags->addInstalledLibraries(argv[0]);

    if (flags->rectangle) {
      llvm::errs()
          << "Warning: --rectangle option is ignored by csl-pp. Set "
             "rectangle dimensions in the layout block with @set_rectangle "
             "instead.\n";
    }

    // Set up the input file
    std::string errorMessage;
    std::unique_ptr<MemoryBuffer> inputFile =
        openInputFile(inputFilename, &errorMessage);
    if (!inputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }

    // Find cslc if we're going to invoke it
    std::string cslcExecutable;
    if (!noCslc) {
      auto cslcOrErr = findCslc();
      if (failed(cslcOrErr))
        return 1;
      cslcExecutable = *cslcOrErr;
    }

    // Like cslc, read the system modules from the SDK image next to the
    // cslc script, unless told where they are.
    if (flags->systemModuleSif.empty() && flags->systemImportPaths.empty()) {
      std::string cslcForImage = cslcExecutable;
      if (cslcForImage.empty()) {
        if (!cslcPath.empty())
          cslcForImage = cslcPath.getValue();
        else if (llvm::ErrorOr<std::string> found =
                     llvm::sys::findProgramByName("cslc"))
          cslcForImage = *found;
      }
      if (!cslcForImage.empty()) {
        if (std::optional<std::string> image =
                findSdkImageNextTo(cslcForImage)) {
          if (llvm::Error err = flags->useSystemModuleSif(*image)) {
            llvm::errs() << "error: " << llvm::toString(std::move(err))
                         << "\n";
            return 1;
          }
        }
      }
    }

    // Determine working directory for preprocessed files
    llvm::SmallString<256> effectiveWorkDir;
    if (workDir.empty()) {
      if (auto ec = llvm::sys::fs::createUniqueDirectory("csl-pp-work",
                                                         effectiveWorkDir)) {
        llvm::errs() << "error: failed to create temp directory: "
                     << ec.message() << "\n";
        return 1;
      }
    } else {
      effectiveWorkDir = workDir.getValue();
      if (auto ec = llvm::sys::fs::create_directories(effectiveWorkDir)) {
        llvm::errs() << "error: failed to create directory '"
                     << effectiveWorkDir << "': " << ec.message() << "\n";
        return 1;
      }
    }

    // Determine output layout file path
    llvm::SmallString<256> layoutOutputPath(effectiveWorkDir);
    llvm::StringRef inputBasename = llvm::sys::path::filename(inputFilename);
    llvm::sys::path::append(layoutOutputPath, inputBasename);

    {
      MLIRContext context(registry);
      auto threadPool = flags->configureContext(&context);
      context.loadAllAvailableDialects();

      // The FiberSizingContext clones live in `context`; declare it
      // INSIDE this scope so the clones are destroyed BEFORE the
      // MLIRContext.  In the two-round path below we also keep the
      // round-1 cslc invocation + round-2 re-emit inside this scope so
      // the resolver can mutate the clones with the context still alive.
      typed_csl::FiberSizingContext fiberSizingContext;

      // Process the layout file (round-1 emit).
      if (failed(processLayoutFile(layoutOutputPath, std::move(inputFile),
                                   &context, *flags, effectiveWorkDir.str(),
                                   &fiberSizingContext))) {
        return 1;
      }

      size_t fiberSizingEntries = fiberSizingContext.size();
      if (verbose && fiberSizingEntries > 0)
        llvm::errs() << "csl-pp: stack sizing captured "
                     << fiberSizingEntries
                     << " PE program(s) with resolver-pending sizes\n";

      // Build the cslc argument list while flags are still alive.  We
      // need it both for --no-cslc's print path and for the actual
      // cslc invocation (one or two rounds).
      if (!noCslc)
        cslcArgs = buildCslcArgs(cslcExecutable, layoutOutputPath, *flags);

      // Fiber stack sizing gate.
      bool needTwoRound = false;
      if (fiberSizingEntries > 0) {
        if (noCslc) {
          // --no-cslc only writes the preprocessed CSL.  It's a valid
          // intermediate, but the per-callee partition arithmetic embeds
          // placeholder sizes — compiling it through cslc directly would
          // produce an ELF with under-sized fiber partitions.  Warn.
          llvm::errs()
              << "warning: the stack sizes of the async tasks in "
              << fiberSizingEntries
              << " PE program(s) are measured by compiling them with cslc, "
                 "which --no-cslc skips; the CSL written to the work "
                 "directory contains placeholder stack sizes that may be "
                 "too small.  Run csl-pp without --no-cslc, or declare "
                 "`stack_size(N)` on every async task and every callee of "
                 "@await_parallel.\n";
        } else {
          // Real compile path — need the shim on disk.
          std::string shimPath = findInjectMllvmSo(argv[0]);
          if (!llvm::sys::fs::exists(shimPath)) {
            llvm::errs()
                << "error: csl-pp needs '" << shimPath
                << "' to measure the stack sizes of the async tasks in "
                << fiberSizingEntries
                << " PE program(s), but the file does not exist.  It is "
                   "built and installed together with csl-pp, so reinstall "
                   "csl-pp.  Programs that declare `stack_size(N)` on every "
                   "async task and every callee of @await_parallel do not "
                   "need it.\n";
            return 1;
          }
          enableInjectShim(shimPath);
          needTwoRound = true;
        }
      }

      // --no-cslc early exit (after the warning above for fiber PEs).
      if (noCslc) {
        auto args = buildCslcArgs("cslc", layoutOutputPath, *flags);
        llvm::outs() << "Preprocessed files written to: " << effectiveWorkDir
                     << "\n";
        llvm::outs() << "Args to invoke cslc with this layout file:\n";
        llvm::interleave(args, llvm::outs(), " \\\n");
        llvm::outs() << "\n";
        return 0;
      }

      if (needTwoRound) {
        // Round 1: cslc must finish so we can read .stack_sizes from the
        // per-PE ELFs.  Always fork+wait (never exec — we have round-2
        // work to do after this returns).
        if (verbose)
          llvm::errs() << "csl-pp: round 1 cslc (measure stack sizes)\n";
        // The user sees the output of the final compile; this one only
        // matters if it fails.
        llvm::SmallString<256> roundOneLog(effectiveWorkDir);
        llvm::sys::path::append(roundOneLog, "cslc-measure-stacks.log");
        if (failed(invokeCslcCompiler(
                cslcArgs,
                verbose ? std::nullopt
                        : std::optional<StringRef>(roundOneLog.str()))))
          return 1;

        // Resolve sizes + re-run stages 2 + 3 on each clone + re-emit.
        std::vector<typed_csl::FiberSizingEntry> entries =
            fiberSizingContext.takeEntries();
        if (failed(applyResolvedFiberSizes(context, *flags,
                                           outputDir.getValue(), entries))) {
          return 1;
        }
        // Drop clones before MLIRContext goes out of scope.
        entries.clear();

        if (verbose)
          llvm::errs() << "csl-pp: round 2 cslc (final ELF)\n";
      }

      // Round 2 (or the single round in the no-fiber case).  --exec-cslc
      // can apply here since the round-1 work (if any) is already done.
      if (execCslc) {
        useExec = true;
      } else if (failed(invokeCslcCompiler(cslcArgs))) {
        return 1;
      }
    } // MLIRContext destroyed — compiler memory freed.
  } // InitLLVM, flags, registry — everything properly destructed.

  if (useExec) {
    // Process is now lightweight. Replace it with cslc.
    execIntoCslc(cslcArgs);
    // Never returns.
  }

  return 0;
}
