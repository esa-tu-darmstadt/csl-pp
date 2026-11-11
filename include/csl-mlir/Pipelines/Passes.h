//===- Passes.h - CSL Pipeline Infrastructure ------------------*- C++ -*-===//
//
// Centralized pass pipeline construction for CSL tools.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_PIPELINES_PASSES_H
#define CSL_MLIR_PIPELINES_PASSES_H

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <memory>
#include <string>
#include <vector>

namespace mlir::typed_csl {
struct FiberSizingContext;
struct FiberSizeSource;
} // namespace mlir::typed_csl

namespace mlir::csl {

class CslModuleResolver;
struct SystemModuleCache;

//===----------------------------------------------------------------------===//
// CslPipelineOptions
//===----------------------------------------------------------------------===//

/// Options for the CSL compilation pipeline.
/// Uses a single CslFlags for all compilation flags, plus pipeline-specific
/// options.
struct CslPipelineOptions : public PassPipelineOptions<CslPipelineOptions> {
  PassOptions::Option<CslFlags> cslFlags{
      *this, "csl-flags", llvm::cl::desc("CSL compilation flags")};

  // Whether to add ProcessTileCodePass (layout file processing)
  PassOptions::Option<bool> processLayoutFile{
      *this, "process-layout-file",
      llvm::cl::desc("Process @set_tile_code calls in layout blocks"),
      llvm::cl::init(false)};

  // Output directory for compiled PE programs (only if processLayoutFile=true)
  PassOptions::Option<std::string> peOutputDir{
      *this, "pe-output-dir",
      llvm::cl::desc("Output directory for compiled PE programs"),
      llvm::cl::init("")};
};

//===----------------------------------------------------------------------===//
// Pipeline Construction
//===----------------------------------------------------------------------===//

/// Build the CSL pass pipeline (ParsedCSL -> TypedCSL).
/// \param pm The pass manager to populate
/// \param options Pipeline options
/// \param resolver Module resolver for parsing imported modules
/// \param systemModuleCache Cached SIF VFS to reuse across resolver instances
void buildCslPassPipeline(
    OpPassManager &pm, const CslPipelineOptions &options,
    std::shared_ptr<CslModuleResolver> resolver,
    std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr,
    typed_csl::FiberSizingContext *fiberSizingContext = nullptr);

/// Build stage 1 of the CSL pipeline: everything UP TO BUT NOT INCLUDING
/// the fiber / trampoline coroutine lowering.  After this stage the IR
/// still has `@await_parallel` / `@await` ops and async-marked FuncOps.
/// The two-round fiber stack sizer clones the PE module here so it can
/// re-run stages 2 + 3 in round 2 with resolved sizes.
void buildCslPassPipelineStage1(
    OpPassManager &pm, const CslPipelineOptions &options,
    std::shared_ptr<CslModuleResolver> resolver,
    std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr);

/// Build stage 2 of the CSL pipeline: the coroutine lowering pass
/// (FiberLoweringPass or the trampoline pipeline).  In fiber mode the
/// `fiberSizeSource` controls partition-arithmetic constants — round 1
/// passes a `PlaceholderSizeSource`, round 2 a `ResolvedSizeSource`.
void buildCslPassPipelineStage2(
    OpPassManager &pm, const CslPipelineOptions &options,
    typed_csl::FiberSizeSource *fiberSizeSource = nullptr);

/// Build stage 3 of the CSL pipeline: late cleanup passes that DCE unused
/// symbols, canonicalize, and CSE.  Safe to run on the original (for
/// round-1 emit) and on the clone (for round-2 emit).
void buildCslPassPipelineStage3(OpPassManager &pm,
                                const CslPipelineOptions &options);

/// Convenience wrapper that runs all three stages back-to-back.  Used by
/// tools that don't need the round-1/round-2 split (csl-s2s, csl-opt's
/// `csl-pipeline`).
void buildCslPassPipelineAllStages(
    OpPassManager &pm, const CslPipelineOptions &options,
    std::shared_ptr<CslModuleResolver> resolver,
    std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr,
    typed_csl::FiberSizeSource *fiberSizeSource = nullptr);

//===----------------------------------------------------------------------===//
// ProcessTileCodePass
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "csl-mlir/Pipelines/Passes.h.inc"

/// Create ProcessTileCodePass for processing @set_tile_code calls in layout
/// files. This pass compiles referenced PE programs with baked-in parameters.
///
/// \param flags CSL compilation flags
/// \param resolver Module resolver for parsing PE programs
/// \param peOutputDir Output directory for compiled PE programs (empty = temp)
/// \param systemModuleCache Cached SIF VFS to reuse across resolver instances
/// \param fiberSizingContext Optional output sink for the post-stage-A
///        PE clones + captured call graphs.  Populated only for PEs that
///        actually need resolver-driven fiber sizing; csl-pp consumes it
///        after round-1 cslc completes.
std::unique_ptr<Pass> createProcessTileCodePass(
    const CslFlags &flags, std::shared_ptr<CslModuleResolver> resolver,
    llvm::StringRef peOutputDir = "",
    std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr,
    typed_csl::FiberSizingContext *fiberSizingContext = nullptr);

/// Build the CSL pass pipeline from CslFlags.
/// Convenience overload that converts CslFlags to CslPipelineOptions.
/// \param pm The pass manager to populate
/// \param flags CSL flags from command line
/// \param resolver Module resolver for parsing imported modules
/// \param processLayoutFile Whether to add ProcessTileCodePass
/// \param peOutputDir Output directory for PE programs (if processLayoutFile)
/// \param systemModuleCache Cached SIF VFS to reuse across resolver instances
void buildCslPassPipeline(
    OpPassManager &pm, const CslFlags &flags,
    std::shared_ptr<CslModuleResolver> resolver, bool processLayoutFile = false,
    llvm::StringRef peOutputDir = "",
    std::shared_ptr<SystemModuleCache> systemModuleCache = nullptr,
    typed_csl::FiberSizingContext *fiberSizingContext = nullptr);

/// Register the CSL pipeline for command-line use (e.g., with mlir-opt).
void registerCslPipeline();

} // namespace mlir::csl

#endif // CSL_MLIR_PIPELINES_PASSES_H
