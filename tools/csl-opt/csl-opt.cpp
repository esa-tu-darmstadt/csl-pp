#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

using namespace mlir;
int main(int argc, char **argv) {
  DialectRegistry registry;

  // Register CSL dialects
  registry.insert<mlir::parsed_csl::ParsedCSLDialect>();
  registry.insert<mlir::typed_csl::TypedCSLDialect>();
  registry.insert<mlir::csl::CSLDialect>();

  // Register comptime evaluation external models
  mlir::parsed_csl::registerComptimeModels(registry);

  // Register CF dialect (needed by coroutine passes)
  registry.insert<mlir::cf::ControlFlowDialect>();

  // Register CSL passes
  mlir::parsed_csl::registerPasses();
  mlir::typed_csl::registerTypedCSLPasses();
  registerCanonicalizerPass();

  // Splice any tokens from $CSL_FLAGS in just after argv[0]. MlirOptMain
  // calls ParseCommandLineOptions internally, so we do this here.
  llvm::BumpPtrAllocator envArgAlloc;
  llvm::StringSaver envArgSaver(envArgAlloc);
  auto args = mlir::csl::addCommandLineOptionsFromEnv(argc, argv, envArgSaver);

  return failed(MlirOptMain(args.size(), const_cast<char **>(args.data()),
                            "csl-mlir modular optimizer driver\n", registry));
}
