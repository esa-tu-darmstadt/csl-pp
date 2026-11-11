# TypedCSL transformation passes

add_mlir_dialect_library(MLIRTypedCSLTransforms
  CheckLinknameCollisionsPass.cpp
  CountTasksPass.cpp
  Coroutine/CoroFrameBuilder.cpp
  Coroutine/CoroFrameRewritePass.cpp
  Coroutine/CoroPrepareCFPass.cpp
  Coroutine/WorkerPoolBuilder.cpp
  Coroutine/InlineSmallNonResumeBlocksPass.cpp
  Coroutine/LowerSuspendingSCFToCFPass.cpp
  Coroutine/TrampolineEmitPass.cpp
  Fiber/DwarfInlineReader.cpp
  Fiber/FiberLoweringPass.cpp
  Fiber/FiberSizeSource.cpp
  Fiber/FiberSizingContext.cpp
  Fiber/FiberStackResolver.cpp
  Fiber/FiberValidation.cpp
  Fiber/StackSizesReader.cpp
  LowerCustomBuiltinsPass.cpp
  StripRuntimeExportsPass.cpp
  TailRecursionToLoopPass.cpp
  Passes.cpp
  RollLayoutLoopsPass.cpp
  ValidateTypedCSLPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/TypedCSL

  DEPENDS
  MLIRTypedCSLTransformsPassesIncGen

  # LLVM components used by StackSizesReader / DwarfInlineReader /
  # FiberStackResolver.  MLIR's add_mlir_dialect_library requires LLVM
  # libraries to be passed via LINK_COMPONENTS, not LINK_LIBS.
  LINK_COMPONENTS
  Object
  DebugInfoDWARF
  Support

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRControlFlowDialect
  MLIRCSLDialect
  MLIRCSLUtils
  MLIRTypedCSLAnalysis
  MLIRTypedCSLDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )
