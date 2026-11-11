//===- CslPipeline.cpp - CSL Pass Pipeline Construction -------------------===//
//
// Centralized pass pipeline construction for CSL tools.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Conversion/Passes.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "csl-mlir/Pipelines/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::csl;

void mlir::csl::buildCslPassPipelineStage1(
    OpPassManager &pm, const CslPipelineOptions &options,
    std::shared_ptr<CslModuleResolver> resolver,
    std::shared_ptr<SystemModuleCache> systemModuleCache) {

  const CslFlags &flags = options.cslFlags;

  // Build ComptimeEvalPass options from CslFlags
  parsed_csl::ComptimeEvalPassOptions comptimeOpts;
  comptimeOpts.cslFlags = flags;

  // Must precede comptime evaluation: the runtime is imported like any other
  // module, and it is comptime evaluation that resolves imports and
  // specializes their contents.
  pm.addPass(parsed_csl::createInjectCoroRuntimePass());

  // Core pipeline (ParsedCSL -> TypedCSL)
  pm.addPass(parsed_csl::createComptimeEvalPass(resolver, comptimeOpts));
  pm.addPass(parsed_csl::createFinalizeInstantiationPass());
  pm.addPass(createCanonicalizerPass());
  parsed_csl::InlineModulesPassOptions inlineOpts;
  inlineOpts.cslFlags = flags;
  pm.addPass(parsed_csl::createInlineModulesPass(inlineOpts));
  pm.addPass(parsed_csl::createLowerTensorAccessPass());
  pm.addPass(parsed_csl::createResolveReadsPass());
  pm.addPass(parsed_csl::createNormalizeCastsPass());
  pm.addPass(parsed_csl::createValidateParsedCSLPass());
  pm.addPass(createConvertParsedCSLToTypedCSLPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(typed_csl::createValidateTypedCSLPass());
  pm.addPass(typed_csl::createCheckLinknameCollisionsPass());
  // Before any coroutine lowering, so that stack sizing sees the loops.
  pm.addPass(typed_csl::createTailRecursionToLoopPass());

  // Static accounting of WSE task slots used by this PE program
  if (flags.countTasks && !options.processLayoutFile)
    pm.addPass(typed_csl::createCountTasksPass());
}

void mlir::csl::buildCslPassPipelineStage2(
    OpPassManager &pm, const CslPipelineOptions &options,
    typed_csl::FiberSizeSource *fiberSizeSource) {

  const CslFlags &flags = options.cslFlags;

  // Coroutine lowering — selected by --coro-mode flag.
  if (flags.coroMode == CoroMode::Fiber) {
    typed_csl::FiberLoweringPassOptions fiberOpts;
    fiberOpts.targetArch = flags.targetArch;
    fiberOpts.outlineAwaitParallel =
        csl::instrumentsAwaitParallel(flags.coroInstrument);
    pm.addPass(typed_csl::createFiberLoweringPass(fiberOpts, fiberSizeSource));
  } else {
    // Trampoline pipeline:
    //   1. Prepare CF: lower calling convention (futures, await loops)
    //   2. Lower suspending SCF to CF: flatten suspending regions to blocks
    //   3. Frame rewrite: analyze frame layout, spill/reload across suspends
    //   4. Trampoline emit: build structured while+switch state machine
    typed_csl::CoroPrepareCFPassOptions coroOpts;
    coroOpts.slotAlignment = flags.coroFrameSlotAlign;
    pm.addPass(typed_csl::createCoroPrepareCFPass(coroOpts));
    pm.addPass(typed_csl::createLowerSuspendingSCFToCFPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(typed_csl::createCoroFrameRewritePass());
    pm.addPass(typed_csl::createInlineSmallNonResumeBlocksPass());
    pm.addPass(typed_csl::createTrampolineEmitPass());
  }
}

void mlir::csl::buildCslPassPipelineStage3(OpPassManager &pm,
                                           const CslPipelineOptions &options) {
  const CslFlags &flags = options.cslFlags;

  typed_csl::LowerCustomBuiltinsPassOptions lowerBuiltinOpts;
  lowerBuiltinOpts.targetArch = flags.targetArch;
  pm.addPass(typed_csl::createLowerCustomBuiltinsPass(lowerBuiltinOpts));
  typed_csl::StripRuntimeExportsPassOptions stripOpts;
  stripOpts.runtimeNoinline = csl::instrumentsRuntime(flags.coroInstrument);
  pm.addPass(typed_csl::createStripRuntimeExportsPass(stripOpts));
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void mlir::csl::buildCslPassPipelineAllStages(
    OpPassManager &pm, const CslPipelineOptions &options,
    std::shared_ptr<CslModuleResolver> resolver,
    std::shared_ptr<SystemModuleCache> systemModuleCache,
    typed_csl::FiberSizeSource *fiberSizeSource) {
  buildCslPassPipelineStage1(pm, options, resolver, systemModuleCache);
  buildCslPassPipelineStage2(pm, options, fiberSizeSource);
  buildCslPassPipelineStage3(pm, options);
}

void mlir::csl::buildCslPassPipeline(
    OpPassManager &pm, const CslPipelineOptions &options,
    std::shared_ptr<CslModuleResolver> resolver,
    std::shared_ptr<SystemModuleCache> systemModuleCache,
    typed_csl::FiberSizingContext *fiberSizingContext) {

  const CslFlags &flags = options.cslFlags;

  // No FiberSizeSource is passed in this top-level entry — the round-1
  // path uses the default PlaceholderSizeSource baked into
  // FiberLoweringPass.  ProcessTileCodePass takes the two-round path
  // for fiber PEs and threads its own source through stages 2 + 3.
  buildCslPassPipelineAllStages(pm, options, resolver, systemModuleCache);

  // Optional: Process layout file (@set_tile_code calls)
  if (options.processLayoutFile) {
    pm.addPass(createProcessTileCodePass(flags, resolver, options.peOutputDir,
                                         systemModuleCache,
                                         fiberSizingContext));
    // Compact repeated builtin calls back into loops
    pm.addPass(typed_csl::createRollLayoutLoopsPass());
  }
}

void mlir::csl::buildCslPassPipeline(
    OpPassManager &pm, const CslFlags &flags,
    std::shared_ptr<CslModuleResolver> resolver, bool processLayoutFile,
    llvm::StringRef peOutputDir,
    std::shared_ptr<SystemModuleCache> systemModuleCache,
    typed_csl::FiberSizingContext *fiberSizingContext) {
  CslPipelineOptions opts;
  opts.cslFlags = flags;
  opts.processLayoutFile = processLayoutFile;
  opts.peOutputDir = peOutputDir.str();

  buildCslPassPipeline(pm, opts, std::move(resolver),
                       std::move(systemModuleCache), fiberSizingContext);
}

void mlir::csl::registerCslPipeline() {
  PassPipelineRegistration<CslPipelineOptions>(
      "csl-pipeline", "CSL compilation pipeline (ParsedCSL -> TypedCSL)",
      [](OpPassManager &pm, const CslPipelineOptions &options) {
        // Note: sourceMgr must be provided separately when using
        // programmatically
        buildCslPassPipeline(pm, options, nullptr);
      });
}
