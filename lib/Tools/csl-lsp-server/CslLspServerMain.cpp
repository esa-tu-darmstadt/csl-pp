//===- CslLspServerMain.cpp - CSL Language Server main
//---------------------===//
//
// CLI parsing, transport setup, and server wiring for the CSL LSP server.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Tools/csl-lsp-server/CslLspServerMain.h"
#include "CSLServer.h"
#include "LSPServer.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/CSL/Utils/Version.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"

using llvm::lsp::JSONStreamStyle;
using llvm::lsp::Logger;

llvm::LogicalResult mlir::csl::CslLspServerMain(int argc, char **argv) {
  llvm::setBugReportMsg("PLEASE report this bug to the csl-mlir developers\n");
  llvm::EnablePrettyStackTrace();
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  static llvm::cl::OptionCategory lspCategory("csl-lsp-server Options");

  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual LSP protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(JSONStreamStyle::Standard),
      llvm::cl::Hidden,
      llvm::cl::cat(lspCategory),
  };
  llvm::cl::opt<bool> litTest{
      "lit-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify lit tests"),
      llvm::cl::init(false),
      llvm::cl::cat(lspCategory),
  };
  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      llvm::cl::init(Logger::Level::Info),
      llvm::cl::cat(lspCategory),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
      llvm::cl::cat(lspCategory),
  };
  llvm::cl::opt<bool> filterOverlappingErrors{
      "filter-overlapping-errors",
      llvm::cl::desc(
          "When multiple error diagnostics overlap, show only the innermost"),
      llvm::cl::init(true),
      llvm::cl::cat(lspCategory),
  };

  // Register CSL-specific CLI options (--import-path, --target-arch, etc.)
  registerCslCLOptions();

  // Hide MLIR/LLVM internal options from --help (still visible via
  // --help-hidden).
  const llvm::cl::OptionCategory *categories[] = {&lspCategory,
                                                  &getCslOptionCategory()};
  llvm::cl::HideUnrelatedOptions(categories);

  // Splice any tokens from $CSL_FLAGS in just after argv[0].
  llvm::BumpPtrAllocator envArgAlloc;
  llvm::StringSaver envArgSaver(envArgAlloc);
  auto args = addCommandLineOptionsFromEnv(argc, argv, envArgSaver);

  llvm::cl::SetVersionPrinter(
      [](llvm::raw_ostream &os) { printVersion(os, "csl-lsp-server"); });
  llvm::cl::ParseCommandLineOptions(args.size(), args.data(),
                                    "CSL LSP Language Server");

  if (litTest) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
  }

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  llvm::lsp::JSONTransport transport(stdin, llvm::outs(), inputStyle,
                                     prettyPrint);

  // Build CSL flags from the registered CLI options.
  llvm::Expected<CslFlags> flags = CslFlags::fromCommandLine();
  if (llvm::Error err = flags.takeError()) {
    llvm::errs() << "Error parsing command-line options: "
                 << llvm::toString(std::move(err)) << "\n";
    return failure();
  }
  // An installed server finds the csl++ libraries without an import path.
  flags->addInstalledLibraries(argv[0]);

  // Configure the server and start the main language server.
  CSLServer::Options options;
  options.flags = *flags;
  options.filterOverlappingErrors = filterOverlappingErrors;
  CSLServer server(options);
  return runCslLSPServer(server, transport);
}
