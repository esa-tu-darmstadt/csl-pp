//===- ProcessTileCodePass.cpp - Process @set_tile_code calls -------------===//
//
// This pass processes @set_tile_code calls in layout blocks, compiles the
// referenced PE programs with baked-in parameters, and rewrites the calls
// to point to the compiled output files.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/CSL/Utils/ResourceBitmaps.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizingContext.h"
#include "csl-mlir/Pipelines/Passes.h"
#include "csl-mlir/Target/CSL/EmitCSL/EmitCSL.h"
#include "csl-mlir/Target/CSL/ParseCSL.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <mutex>
#include <optional>

#define DEBUG_TYPE "process-tile-code"

namespace mlir::csl {
#define GEN_PASS_DEF_PROCESSTILECODEPASS
#include "csl-mlir/Pipelines/Passes.h.inc"
} // namespace mlir::csl

using namespace mlir;
using namespace mlir::csl;

namespace {

/// Generate a cache key from filename and parameters.
static std::string generateCacheKey(llvm::StringRef filename,
                                    csl::StructValueAttr params) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << filename << ":";
  os << params;
  return key;
}

/// Generate an output filename using MD5 hash of the cache key.
static std::string generateOutputFilename(llvm::StringRef originalFilename,
                                          llvm::StringRef cacheKey) {
  llvm::MD5 md5;
  md5.update(cacheKey);
  llvm::MD5::MD5Result result;
  md5.final(result);

  llvm::SmallString<128> stem(llvm::sys::path::stem(originalFilename));
  std::string hashStr = llvm::toHex(result, /*LowerCase=*/true).substr(0, 12);
  return (stem + "__" + hashStr + ".csl").str();
}

/// A compilation task for a unique (filename, params) combination.
struct PECompilationTask {
  std::string cacheKey;
  std::string filename;
  csl::StructValueAttr params;
  std::optional<Location> loc;
  SmallVector<typed_csl::BuiltinCallOp> callsToRewrite;
  unsigned tileX;
  unsigned tileY;
  /// The diagnostics of compiling this PE program, emitted after all PE
  /// programs are compiled, and how many of them are errors.
  std::vector<Diagnostic> diagnostics;
  unsigned errorCount = 0;
};

/// The PE program the current thread is compiling, if any. Diagnostics are
/// emitted on the thread that compiles the program they belong to.
thread_local PECompilationTask *currentPETask = nullptr;

/// Marks the current thread as compiling `task` while it is alive.
class CurrentPETaskScope {
public:
  explicit CurrentPETaskScope(PECompilationTask &task) {
    currentPETask = &task;
  }
  ~CurrentPETaskScope() { currentPETask = nullptr; }
  CurrentPETaskScope(const CurrentPETaskScope &) = delete;
  CurrentPETaskScope &operator=(const CurrentPETaskScope &) = delete;
};

/// A key that is equal for two diagnostics that print the same text.
static std::string getDiagnosticKey(Diagnostic &diag) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << static_cast<int>(diag.getSeverity()) << '\0' << diag.getLocation()
     << '\0' << diag.str();
  for (Diagnostic &note : diag.getNotes())
    os << '\0' << note.getLocation() << '\0' << note.str();
  return key;
}

/// Pass that processes @set_tile_code calls in layout blocks.
struct ProcessTileCodePass
    : public csl::impl::ProcessTileCodePassBase<ProcessTileCodePass> {
  using csl::impl::ProcessTileCodePassBase<
      ProcessTileCodePass>::ProcessTileCodePassBase;

  ProcessTileCodePass(const ProcessTileCodePass &other)
      : csl::impl::ProcessTileCodePassBase<ProcessTileCodePass>(other),
        moduleResolver(other.moduleResolver),
        systemModuleCache(other.systemModuleCache),
        fiberSizingContext(other.fiberSizingContext) {}

  ProcessTileCodePass(const CslFlags &flags,
                      std::shared_ptr<CslModuleResolver> resolver,
                      llvm::StringRef outputDir,
                      std::shared_ptr<SystemModuleCache> cache,
                      typed_csl::FiberSizingContext *fiberSizing)
      : moduleResolver(std::move(resolver)),
        systemModuleCache(std::move(cache)), fiberSizingContext(fiberSizing) {
    cslFlags = flags;
    peOutputDir = outputDir.str();
  }

  void runOnOperation() override;

private:
  std::shared_ptr<CslModuleResolver> moduleResolver;
  std::shared_ptr<SystemModuleCache> systemModuleCache;
  typed_csl::FiberSizingContext *fiberSizingContext = nullptr;

  // Layout-level resource bitmap (propagated to PE programs).
  csl::ResourceBitmaps layoutResourceBitmaps;

  // Runtime state
  llvm::StringMap<std::string> compiledPECache;
  std::mutex cacheMutex;
  std::mutex resolverMutex; // guards moduleResolver access from parallel tasks
  std::atomic<bool> firstPEPrinted{false};
  uint16_t rectWidth = 0;
  uint16_t rectHeight = 0;
  std::string outputDirectory;
  std::string layoutFileDir;

  LogicalResult extractRectangleDims(typed_csl::LayoutOp layout);
  FailureOr<llvm::StringMap<PECompilationTask>>
  collectCompilationTasks(typed_csl::LayoutOp layout);
  LogicalResult
  compileTasksInParallel(llvm::StringMap<PECompilationTask> &tasks);
  LogicalResult
  rewriteSetTileCodeCalls(llvm::StringMap<PECompilationTask> &tasks);
  FailureOr<std::string> compilePEProgram(const PECompilationTask &task);
  LogicalResult ensureOutputDirectory();
  llvm::DenseMap<llvm::StringRef, csl::ParamBinding>
  createParamBindings(MLIRContext *ctx, csl::StructValueAttr params,
                      Location definitionLoc);
};

} // namespace

void ProcessTileCodePass::runOnOperation() {
  auto unit = getOperation();

  // Determine the layout file directory from the compilation unit's location
  if (auto fileLoc = mlir::dyn_cast<FileLineColLoc>(unit.getLoc())) {
    layoutFileDir = llvm::sys::path::parent_path(fileLoc.getFilename()).str();
  }

  if (failed(ensureOutputDirectory())) {
    signalPassFailure();
    return;
  }

  // Find layout block
  typed_csl::LayoutOp layout;
  unit.walk([&](typed_csl::LayoutOp op) {
    layout = op;
    return WalkResult::interrupt();
  });

  if (!layout)
    return; // No layout block - nothing to do

  // Capture the layout's resource bitmap so PE programs start with
  // layout-level claims (colors, tasks, etc.) already reserved.
  layoutResourceBitmaps =
      csl::ResourceBitmaps::loadFromUnit(unit, cslFlags.targetArch);

  if (failed(extractRectangleDims(layout))) {
    signalPassFailure();
    return;
  }

  // Phase 1: Collect all unique (filename, params) combinations
  auto tasksOrErr = collectCompilationTasks(layout);
  if (failed(tasksOrErr)) {
    signalPassFailure();
    return;
  }
  auto &tasks = *tasksOrErr;

  if (tasks.empty())
    return;

  // Phase 2: Compile PE programs in parallel
  if (failed(compileTasksInParallel(tasks))) {
    signalPassFailure();
    return;
  }

  // Phase 3: Rewrite @set_tile_code calls with compiled filenames
  if (failed(rewriteSetTileCodeCalls(tasks))) {
    signalPassFailure();
    return;
  }
}

LogicalResult
ProcessTileCodePass::extractRectangleDims(typed_csl::LayoutOp layout) {
  bool found = false;

  layout.walk([&](typed_csl::BuiltinCallOp call) {
    if (call.getBuiltin() != csl::BuiltinKind::SetRectangle)
      return WalkResult::advance();

    auto args = call.getArgs();
    if (args.size() != 2) {
      call.emitError("@set_rectangle expects 2 arguments");
      return WalkResult::interrupt();
    }

    auto widthOp = args[0].getDefiningOp<typed_csl::ConstantOp>();
    auto heightOp = args[1].getDefiningOp<typed_csl::ConstantOp>();

    if (!widthOp || !heightOp) {
      call.emitError("@set_rectangle arguments must be constants");
      return WalkResult::interrupt();
    }

    auto widthAttr = mlir::dyn_cast<csl::ComptimeIntAttr>(widthOp.getValue());
    auto heightAttr = mlir::dyn_cast<csl::ComptimeIntAttr>(heightOp.getValue());

    if (!widthAttr || !heightAttr) {
      call.emitError("@set_rectangle arguments must be comptime_int");
      return WalkResult::interrupt();
    }

    rectWidth = static_cast<uint16_t>(widthAttr.getInt());
    rectHeight = static_cast<uint16_t>(heightAttr.getInt());
    found = true;
    return WalkResult::interrupt();
  });

  if (!found) {
    layout.emitError("layout block must contain @set_rectangle call");
    return failure();
  }

  return success();
}

FailureOr<llvm::StringMap<PECompilationTask>>
ProcessTileCodePass::collectCompilationTasks(typed_csl::LayoutOp layout) {
  llvm::StringMap<PECompilationTask> tasks;

  SmallVector<typed_csl::BuiltinCallOp> setTileCodeCalls;
  layout.walk([&](typed_csl::BuiltinCallOp call) {
    if (call.getBuiltin() == csl::BuiltinKind::SetTileCode)
      setTileCodeCalls.push_back(call);
  });

  for (auto call : setTileCodeCalls) {
    auto args = call.getArgs();
    if (args.size() != 4) {
      return call.emitError("@set_tile_code expects 4 arguments");
    }

    auto filenameOp = args[2].getDefiningOp<typed_csl::ConstantOp>();
    if (!filenameOp) {
      return call.emitError("@set_tile_code filename must be a constant");
    }

    auto filenameAttr =
        mlir::dyn_cast<csl::ComptimeStringAttr>(filenameOp.getValue());
    if (!filenameAttr) {
      return call.emitError(
          "@set_tile_code filename must be a comptime_string");
    }
    std::string filename = filenameAttr.getValue().str();

    auto paramsOp = args[3].getDefiningOp<typed_csl::ConstantOp>();
    if (!paramsOp) {
      return call.emitError("@set_tile_code params must be a constant");
    }

    auto paramsAttr = mlir::dyn_cast<csl::StructValueAttr>(paramsOp.getValue());
    if (!paramsAttr) {
      return call.emitError("@set_tile_code params must be a struct");
    }

    auto extractIntArg = [](Value value) -> FailureOr<unsigned> {
      auto constOp = value.getDefiningOp<typed_csl::ConstantOp>();
      if (!constOp)
        return failure();
      if (auto comptimeIntAttr =
              mlir::dyn_cast<csl::ComptimeIntAttr>(constOp.getValue()))
        return static_cast<unsigned>(comptimeIntAttr.getInt());
      else if (auto intAttr =
                   mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
        return static_cast<unsigned>(intAttr.getInt());
      return failure();
    };

    FailureOr<unsigned> tileXOrErr = extractIntArg(args[0]);
    FailureOr<unsigned> tileYOrErr = extractIntArg(args[1]);
    if (failed(tileXOrErr) || failed(tileYOrErr)) {
      return call.emitError(
          "@set_tile_code tile coordinates must be constant integers");
    }

    std::string cacheKey = generateCacheKey(filename, paramsAttr);

    auto &task = tasks[cacheKey];
    if (task.cacheKey.empty()) {
      task.cacheKey = cacheKey;
      task.filename = filename;
      task.params = paramsAttr;
      task.loc = call.getLoc();
      task.tileX = *tileXOrErr;
      task.tileY = *tileYOrErr;
    }
    task.callsToRewrite.push_back(call);
  }

  return tasks;
}

LogicalResult ProcessTileCodePass::compileTasksInParallel(
    llvm::StringMap<PECompilationTask> &tasks) {
  MLIRContext *ctx = &getContext();

  // Compile in tile order, so that diagnostics come out in the same order on
  // every run.
  SmallVector<PECompilationTask *> taskPtrs;
  for (auto &entry : tasks)
    taskPtrs.push_back(&entry.second);
  llvm::sort(taskPtrs, [](PECompilationTask *a, PECompilationTask *b) {
    return std::make_pair(a->tileY, a->tileX) <
           std::make_pair(b->tileY, b->tileX);
  });

  // Hold back the diagnostics of each PE program until all are compiled.
  ScopedDiagnosticHandler collect(ctx, [](Diagnostic &diag) {
    PECompilationTask *task = currentPETask;
    if (!task)
      return failure();
    if (diag.getSeverity() == DiagnosticSeverity::Error)
      ++task->errorCount;
    task->diagnostics.push_back(std::move(diag));
    return success();
  });

  std::atomic<bool> anyFailed = false;
  auto compile = [&](PECompilationTask *task) {
    [[maybe_unused]] std::chrono::high_resolution_clock::time_point
        startTime = std::chrono::high_resolution_clock::now();
    FailureOr<std::string> result;
    {
      CurrentPETaskScope scope(*task);
      result = compilePEProgram(*task);
    }
    if (failed(result)) {
      anyFailed = true;
      return;
    }

    LLVM_DEBUG({
      std::chrono::high_resolution_clock::duration elapsedTime =
          std::chrono::high_resolution_clock::now() - startTime;
      llvm::dbgs() << "Transpiled tile code for tile (" << task->tileX << ", "
                   << task->tileY << ") in "
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          elapsedTime)
                          .count()
                   << " ms to file " << *result << "\n";
    });

    std::lock_guard<std::mutex> lock(cacheMutex);
    compiledPECache[task->cacheKey] = *result;
  };

  if (ctx->isMultithreadingEnabled() && taskPtrs.size() > 1) {
    ctx->enterMultiThreadedExecution();
    llvm::ThreadPoolTaskGroup group(ctx->getThreadPool());
    for (PECompilationTask *task : taskPtrs)
      group.async([&compile, task] { compile(task); });
    group.wait();
    ctx->exitMultiThreadedExecution();
  } else {
    for (PECompilationTask *task : taskPtrs)
      compile(task);
  }

  // Every distinct set of @set_tile_code parameters is compiled separately,
  // so an error in the PE source would be reported once per set. Report each
  // diagnostic once, and name the tile it was first found for.
  llvm::StringSet<> reported;
  for (PECompilationTask *task : taskPtrs) {
    for (Diagnostic &diag : task->diagnostics) {
      if (!reported.insert(getDiagnosticKey(diag)).second)
        continue;
      if (diag.getSeverity() == DiagnosticSeverity::Error ||
          diag.getSeverity() == DiagnosticSeverity::Warning)
        diag.attachNote(*task->loc)
            << "while compiling '" << task->filename << "' for tile ("
            << task->tileX << ", " << task->tileY << ")";
      ctx->getDiagEngine().emit(std::move(diag));
    }
    task->diagnostics.clear();
  }

  return failure(anyFailed);
}

LogicalResult ProcessTileCodePass::rewriteSetTileCodeCalls(
    llvm::StringMap<PECompilationTask> &tasks) {

  for (auto &entry : tasks) {
    PECompilationTask &task = entry.second;
    std::string outputFilename = compiledPECache[task.cacheKey];

    for (auto call : task.callsToRewrite) {
      OpBuilder builder(call);

      auto newFilenameAttr =
          csl::ComptimeStringAttr::get(call.getContext(), outputFilename);
      auto newFilenameOp = typed_csl::ConstantOp::create(
          builder, call.getLoc(), newFilenameAttr.getType(), newFilenameAttr);

      auto emptyStructAttr = csl::StructValueAttr::get(
          call.getContext(), /*field_values=*/{},
          /*def_location=*/LocationAttr(), /*field_names=*/{});
      auto emptyStructOp = typed_csl::ConstantOp::create(
          builder, call.getLoc(), emptyStructAttr.getType(), emptyStructAttr);

      call.setOperand(2, newFilenameOp.getResult());
      call.setOperand(3, emptyStructOp.getResult());
    }
  }

  return success();
}

FailureOr<std::string>
ProcessTileCodePass::compilePEProgram(const PECompilationTask &task) {
  llvm::StringRef filename = task.filename;
  csl::StructValueAttr params = task.params;
  Location loc = *task.loc;
  MLIRContext *ctx = &getContext();

  // A parse or compile step that fails has reported why. Only a failure
  // without an error needs a message of its own.
  unsigned errorCountAtStart = task.errorCount;
  auto failedSilently = [&](StringRef step) -> FailureOr<std::string> {
    if (task.errorCount != errorCountAtStart)
      return failure();
    return emitError(loc) << step << " '" << filename
                          << "' failed without reporting an error (internal "
                             "error)";
  };

  // Resolve the PE file path. System module paths use angle brackets
  // (e.g. "<kernels/fft/fft3d_rpc>") and are resolved through system import
  // paths, the same way @import_module handles them.
  bool isSystemModule = filename.starts_with("<") && filename.ends_with(">");
  llvm::SmallString<256> peFilePath;
  std::unique_ptr<llvm::MemoryBuffer> peSource;

  if (isSystemModule) {
    if (!moduleResolver || !moduleResolver->hasSystemImportPaths()) {
      return emitError(loc)
             << "PE program references system module " << filename
             << " but no system import paths are configured. "
                "Use --system-module-sif or --system-import-path.";
    }

    llvm::StringRef modulePath = filename.drop_front(1).drop_back(1);

    // CslModuleResolver is not thread-safe (its SourceMgr uses a plain
    // std::vector for buffers). Serialize resolve + buffer-copy so that
    // concurrent compilePEProgram calls don't race on the shared resolver.
    {
      std::lock_guard<std::mutex> lock(resolverMutex);
      auto resolved = moduleResolver->resolveSystemModule(modulePath);
      if (failed(resolved)) {
        return emitError(loc)
               << "PE program file not found in system module paths: "
               << filename << " (searched in --system-import-path directories)";
      }

      // Get the buffer from the layout's SourceMgr and make a copy for the PE
      // compilation (which uses its own SourceMgr).
      auto layoutSrcMgr = moduleResolver->getSourceMgr();
      const llvm::MemoryBuffer *origBuf =
          layoutSrcMgr->getMemoryBuffer(resolved->bufferID);
      peSource = llvm::MemoryBuffer::getMemBufferCopy(
          origBuf->getBuffer(), origBuf->getBufferIdentifier());
      peFilePath = resolved->resolvedPath;
    }
  } else {
    if (llvm::sys::path::is_absolute(filename)) {
      peFilePath = filename;
    } else {
      peFilePath = layoutFileDir;
      llvm::sys::path::append(peFilePath, filename);
    }

    if (!llvm::sys::fs::exists(peFilePath)) {
      return emitError(loc) << "PE program file not found: " << peFilePath;
    }

    auto peSourceOrErr = llvm::MemoryBuffer::getFile(peFilePath);
    if (!peSourceOrErr) {
      return emitError(loc) << "Failed to read PE program file: " << peFilePath
                            << ": " << peSourceOrErr.getError().message();
    }
    peSource = std::move(*peSourceOrErr);
  }

  // Build CslFlags for PE compilation from this pass's cslFlags,
  // but override rectangle with the extracted @set_rectangle dims.
  CslFlags peFlags = this->cslFlags;
  peFlags.rectangle = {rectWidth, rectHeight};

  auto peResolver = std::make_shared<CslModuleResolver>(peFilePath, peFlags,
                                                        systemModuleCache);
  unsigned peBufferID = peResolver->addBuffer(std::move(peSource));
  auto peSourceMgr = peResolver->getSourceMgr();

  // NOTE: We intentionally do NOT register a per-PE SourceMgrDiagnosticHandler
  // here. MLIR's DiagnosticEngine calls ALL registered handlers for every
  // diagnostic (serialized by a mutex), but the handler callback accesses its
  // SourceMgr — which races with the owning thread's concurrent parsing.
  // Diagnostics fall through to the main csl-pp handler instead, which safely
  // uses the layout's SourceMgr (not being parsed at this point).
  // FIXME: Do we get a problem when we get a diagnostic from a module relative
  // to the PE program and layout file is not in the same directory?

  mlir::OwningOpRef<mlir::ModuleOp> peModule(
      ModuleOp::create(UnknownLoc::get(ctx)));
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(peModule->getBody());

  auto peCompilationUnit = csl::CompilationUnitOp::create(
      builder, FileLineColLoc::get(ctx, peFilePath, 0, 0));

  auto paramBindings = createParamBindings(ctx, params, loc);

  if (failed(csl::parseCSLIntoRegion(
          peSourceMgr, peBufferID, peCompilationUnit.getRegion(), paramBindings,
          /*lineOffset=*/0, /*sourceFilename=*/"", /*currentModule=*/nullptr,
          /*lspMode=*/false, peFlags))) {
    return failedSilently("parsing");
  }

  // Propagate layout-level resource claims to the PE so its comptime
  // allocator starts with those resources already reserved.
  layoutResourceBitmaps.storeToUnit(peCompilationUnit);

  // The PE pipeline is split into three stages so the two-round fiber
  // stack sizer can clone the module BEFORE fiber lowering and re-run
  // stages 2 + 3 on the clone in round 2 with resolved per-callee sizes.
  // FiberLoweringPass takes a FiberSizeSource pointer; the round-1 path
  // here passes a PlaceholderSizeSource (also honoring user `stack_size(N)`
  // overrides).
  CslPipelineOptions pipelineOpts;
  pipelineOpts.cslFlags = peFlags;
  pipelineOpts.processLayoutFile = false;

  auto makePassManager = [&]() {
    auto pm = std::make_unique<PassManager>(ctx);
    if (peFlags.profilePasses)
      pm->enableTiming();
    pm->enableVerifier(peFlags.verifyBetweenPasses);
    return pm;
  };

  auto enableIRPrintingIfRequested = [&](PassManager &pm,
                                         StringRef stageLabel) {
    bool printThisPE = false;
    if (peFlags.printIRAfterAll == csl::PrintIRMode::AllPE ||
        peFlags.printIRAfterAll == csl::PrintIRMode::All)
      printThisPE = true;
    else if (peFlags.printIRAfterAll == csl::PrintIRMode::FirstPE)
      printThisPE = !firstPEPrinted.exchange(true);
    if (printThisPE) {
      llvm::errs() << "// ===== PE IR " << stageLabel << ": " << filename
                   << " =====\n";
      pm.enableIRPrinting(
          /*shouldPrintBeforePass=*/[](Pass *, Operation *) { return false; },
          /*shouldPrintAfterPass=*/[](Pass *, Operation *) { return true; },
          /*printModuleScope=*/true,
          /*printAfterOnlyOnChange=*/true);
    }
  };

  // Stage 1: ParsedCSL → TypedCSL, up to but not including fiber/coro
  // lowering.  After this stage the IR still has @await_parallel and
  // async-marked FuncOps; this is the snapshot the two-round sizer
  // re-uses in round 2.
  {
    auto pm = makePassManager();
    OpPassManager &stagePM = pm->nest<csl::CompilationUnitOp>();
    buildCslPassPipelineStage1(stagePM, pipelineOpts, peResolver,
                               systemModuleCache);
    enableIRPrintingIfRequested(*pm, "Stage 1");
    if (failed(pm->run(peModule.get())))
      return failedSilently("compiling");
  }

  // Snapshot the pre-fiber IR into the FiberSizingContext if this PE
  // has at least one async fn that FiberLoweringPass will query the
  // FiberSizeSource for without a `stack_size(N)` override.  See
  // typed_csl::peNeedsFiberSizeResolution for the exact rule.  Push to
  // the context after emit (below) so the entry carries the final
  // emittedCSLPath.
  std::optional<typed_csl::FiberSizingEntry> pendingEntry;
  if (fiberSizingContext && peFlags.coroMode == csl::CoroMode::Fiber &&
      typed_csl::peNeedsFiberSizeResolution(*peModule)) {
    typed_csl::FiberSizingEntry entry;
    entry.clone = mlir::cast<ModuleOp>(peModule->clone());
    entry.peSourceFile = std::string(peFilePath);
    entry.tileCoords.emplace_back(static_cast<int>(task.tileX),
                                  static_cast<int>(task.tileY));
    pendingEntry = std::move(entry);
  }

  // Stage 2: fiber / coroutine lowering.  Round-1 path uses the
  // PlaceholderSizeSource baked into FiberLoweringPass (the
  // `fiberSizeSource = nullptr` default).
  {
    auto pm = makePassManager();
    OpPassManager &stagePM = pm->nest<csl::CompilationUnitOp>();
    buildCslPassPipelineStage2(stagePM, pipelineOpts,
                               /*fiberSizeSource=*/nullptr);
    enableIRPrintingIfRequested(*pm, "Stage 2");
    if (failed(pm->run(peModule.get())))
      return failedSilently("compiling");
  }

  // Capture the call graph (attached to FuncOps by FiberLoweringPass) into
  // the entry.  Done post-stage-2 because the capture step lives inside
  // FiberLoweringPass.
  if (pendingEntry) {
    pendingEntry->callGraph = typed_csl::extractCapturedCallGraph(*peModule);
  }

  // Stage 3: late cleanup (canonicalize, DCE, CSE).  Same shape on both
  // rounds — the round-2 path also re-runs stages 2 + 3 on the clone.
  {
    auto pm = makePassManager();
    OpPassManager &stagePM = pm->nest<csl::CompilationUnitOp>();
    buildCslPassPipelineStage3(stagePM, pipelineOpts);
    enableIRPrintingIfRequested(*pm, "Stage 3");
    if (failed(pm->run(peModule.get())))
      return failedSilently("compiling");
  }

  std::string cacheKey = generateCacheKey(filename, params);
  std::string outputFilename = generateOutputFilename(filename, cacheKey);

  llvm::SmallString<256> outputPath(outputDirectory);
  llvm::sys::path::append(outputPath, outputFilename);

  std::error_code ec;
  llvm::raw_fd_ostream outputStream(outputPath, ec);
  if (ec) {
    return emitError(loc) << "Failed to open output file: " << outputPath
                          << ": " << ec.message();
  }

  if (failed(csl::emitCSL(peCompilationUnit, outputStream))) {
    return emitError(loc) << "Failed to emit PE program: " << outputPath;
  }

  // Now that we know the round-1 emit path, hand the snapshot off to the
  // FiberSizingContext.  Round 2 re-runs stages 2 + 3 on `entry.clone`
  // with a ResolvedSizeSource and re-emits, overwriting this file.
  if (pendingEntry) {
    pendingEntry->emittedCSLPath = std::string(outputPath);
    fiberSizingContext->add(std::move(*pendingEntry));
  }

  return outputFilename;
}

LogicalResult ProcessTileCodePass::ensureOutputDirectory() {
  if (peOutputDir.empty()) {
    llvm::SmallString<128> tempDir;
    if (auto ec =
            llvm::sys::fs::createUniqueDirectory("csl-pe-output", tempDir)) {
      emitError(UnknownLoc::get(&getContext()))
          << "Failed to create temporary output directory: " << ec.message();
      return failure();
    }
    outputDirectory = tempDir.str().str();
  } else {
    outputDirectory = peOutputDir;
    if (auto ec = llvm::sys::fs::create_directories(outputDirectory)) {
      emitError(UnknownLoc::get(&getContext()))
          << "Failed to create output directory '" << outputDirectory
          << "': " << ec.message();
      return failure();
    }
  }
  return success();
}

llvm::DenseMap<llvm::StringRef, csl::ParamBinding>
ProcessTileCodePass::createParamBindings(MLIRContext *ctx,
                                         csl::StructValueAttr params,
                                         Location definitionLoc) {
  llvm::DenseMap<llvm::StringRef, csl::ParamBinding> bindings;

  if (params.isTuple())
    return bindings;

  for (size_t i = 0; i < params.getNumFields(); ++i) {
    auto nameResult = params.getFieldName(i);
    if (failed(nameResult))
      continue;

    llvm::StringRef name = *nameResult;
    TypedAttr value = params.getField(i);
    bindings.try_emplace(name, csl::ParamBinding{value, definitionLoc});
  }

  return bindings;
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::csl::createProcessTileCodePass(
    const CslFlags &flags, std::shared_ptr<CslModuleResolver> resolver,
    llvm::StringRef peOutputDir,
    std::shared_ptr<SystemModuleCache> systemModuleCache,
    typed_csl::FiberSizingContext *fiberSizingContext) {
  return std::make_unique<ProcessTileCodePass>(
      flags, std::move(resolver), peOutputDir, std::move(systemModuleCache),
      fiberSizingContext);
}
