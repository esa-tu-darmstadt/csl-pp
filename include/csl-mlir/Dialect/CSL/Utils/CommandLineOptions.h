//===- CommandLineOptions.h - Shared CSL Tool Options -----------*- C++ -*-===//
//
// Shared command-line options for CSL tools (csl-s2s, csl-pp, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef CSL_MLIR_DIALECT_CSL_UTILS_COMMANDLINEOPTIONS_H
#define CSL_MLIR_DIALECT_CSL_UTILS_COMMANDLINEOPTIONS_H

#include "csl-mlir/Dialect/CSL/Utils/ParameterParsing.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Threading.h"
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace csl {

//===----------------------------------------------------------------------===//
// Fp16Format
//===----------------------------------------------------------------------===//

/// The FP16 format selected by the --fp16-format command-line option.
enum class Fp16Format { F16, BF16, CB16 };

//===----------------------------------------------------------------------===//
// DiagLevel
//===----------------------------------------------------------------------===//

/// Severity level for a diagnostic flag (-W<name>=off|warn|error).
enum class DiagLevel { Off, Warn, Error };

/// Parse a DiagLevel from a string. Returns std::nullopt on invalid input.
std::optional<DiagLevel> parseDiagLevel(llvm::StringRef str);

/// Convert a DiagLevel to its string representation.
llvm::StringRef diagLevelToString(DiagLevel level);

//===----------------------------------------------------------------------===//
// PrintIRMode
//===----------------------------------------------------------------------===//

/// Modes for --print-ir-after-all flag.
enum class PrintIRMode {
  Off,     ///< No IR printing
  Layout,  ///< Print IR for layout file only (csl-pp)
  FirstPE, ///< Print IR for the first PE program only (csl-pp)
  AllPE,   ///< Print IR for all PE programs (csl-pp)
  All      ///< Print IR for everything (layout + all PEs, or csl-s2s)
};

//===----------------------------------------------------------------------===//
// CoroMode
//===----------------------------------------------------------------------===//

/// Coroutine lowering strategy selected by --coro-mode.
enum class CoroMode {
  Fiber,     ///< Fiber-based stack switching
  Trampoline ///< Stackless state-machine trampolines
};

//===----------------------------------------------------------------------===//
// CoroInstrument
//===----------------------------------------------------------------------===//

/// Profiling instrumentation for coroutine lowering, selected by
/// --coro-instrument. Runtime/All deliberately change generated calls and
/// frames. AwaitParallel introduces an ordinary named function before backend
/// optimization so its DWARF inline scope is attributable; any timing effect
/// is measured rather than assumed absent. Fiber mode only.
enum class CoroInstrument {
  None,          ///< No instrumentation (default)
  Runtime,       ///< Keep the coroutine runtime functions out of line
  AwaitParallel, ///< Outline each @await_parallel site into its own fn
  All            ///< Both of the above
};

/// True if `mode` keeps the coroutine runtime functions out of line.
inline bool instrumentsRuntime(CoroInstrument mode) {
  return mode == CoroInstrument::Runtime || mode == CoroInstrument::All;
}

/// True if `mode` outlines @await_parallel sites.
inline bool instrumentsAwaitParallel(CoroInstrument mode) {
  return mode == CoroInstrument::AwaitParallel || mode == CoroInstrument::All;
}

//===----------------------------------------------------------------------===//
// TargetArch
//===----------------------------------------------------------------------===//

/// Target architecture selected by --arch.
enum class TargetArch {
  WSE2, ///< WSE2 architecture
  WSE3  ///< WSE3 architecture
};

//===----------------------------------------------------------------------===//
// CslDiagnosticFlags
//===----------------------------------------------------------------------===//

/// Diagnostic severity configuration.
struct CslDiagnosticFlags {
  // Generated from CslFlagsFields.def:
  // def is a lowercase string like "off"; *parseDiagLevel("off") == Off.
#define CSL_FLAG_DIAG(name, key, def, desc)                                    \
  DiagLevel name = *parseDiagLevel(def);
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  bool operator==(const CslDiagnosticFlags &rhs) const {
    return true
#define CSL_FLAG_DIAG(name, key, def, desc) &&name == rhs.name
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"
        ;
  }
};

//===----------------------------------------------------------------------===//
// CslFlags
//===----------------------------------------------------------------------===//

/// Flags that control CSL tool behavior. Can be constructed from command-line
/// options (after registerCslCLOptions() is called) or programmatically.
///
/// CslFlags is also usable as an MLIR pass option type via
/// `PassOptions::Option<CslFlags>`. The serialization format uses semicolons
/// to separate key=value pairs:
///   arch=wse3;rectangle=4,3;fp16-format=bf16;import-path=/foo,/bar
///
/// Usage:
///   // In main(), before cl::ParseCommandLineOptions():
///   csl::registerCslCLOptions();
///
///   // After parsing command line:
///   CslFlags flags;  // Automatically populated from CLI options
///
///   // Or programmatically:
///   CslFlags flags;
///   flags.targetArch = "wse3";
///   flags.importPaths.push_back("/path/to/imports");
///
class CslFlags {
public:
  //===--------------------------------------------------------------------===//
  // Public data members (generated from CslFlagsFields.def)
  //===--------------------------------------------------------------------===//

  // Scalar, enum, and list flags
#define CSL_FLAG(Type, name, key, def, desc) Type name = Type(def);
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...) Type name = def;
#define CSL_FLAG_LIST(ElemType, name, key, desc) std::vector<ElemType> name;
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // Diagnostic flags
  CslDiagnosticFlags diagnostics;

  // Manual fields (not in .def — complex parsing or special handling)
  ParsedParams params;

  //===--------------------------------------------------------------------===//
  // Construction
  //===--------------------------------------------------------------------===//

  /// Construct an empty CslFlags with defaults.
  CslFlags() = default;

  /// Parse command-line options into a CslFlags struct. Returns an error if
  /// parsing fails (e.g. invalid format for --params).
  static llvm::Expected<CslFlags> fromCommandLine();

  /// Read the system modules from the Cerebras SDK image `sifPath`, from the
  /// SDK's default module directories unless system import paths are set.
  /// Returns an error if the image cannot be read.
  llvm::Error useSystemModuleSif(llvm::StringRef sifPath);

  /// Append the csl++ libraries of the installation that contains the running
  /// executable, <prefix>/share/csl-mlir/libs for <prefix>/bin/<tool>, to the
  /// import paths, so that they are searched after all other import paths.
  /// Does nothing if the directory does not exist, as in a build tree.
  void addInstalledLibraries(const char *argv0);

  //===--------------------------------------------------------------------===//
  // Serialization (for MLIR pass option support)
  //===--------------------------------------------------------------------===//

  /// Parse CslFlags from a semicolon-separated key=value string.
  /// Format: "arch=wse3;rectangle=4,3;fp16-format=bf16;import-path=/a,/b"
  /// An empty string produces default flags.
  static llvm::Expected<CslFlags> parseFromString(llvm::StringRef str);

  /// Print CslFlags in the semicolon-separated format (only non-default
  /// values).
  void print(llvm::raw_ostream &os) const;

  //===--------------------------------------------------------------------===//
  // Derived accessors (non-trivial logic; members are public for direct use)
  //===--------------------------------------------------------------------===//

  /// Create parameter and color bindings for the given MLIR context.
  ParamBindingsMap toBindings(MLIRContext *ctx) const {
    return params.toBindings(ctx);
  }

  /// Append parameter/color bindings from another ParsedParams.
  void addParams(const ParsedParams &p) {
    for (const auto &pv : p.values)
      params.values.push_back(pv);
  }

  /// Parse and add parameter bindings from a string like "DIM:2,Nz:4".
  LogicalResult addParamsFromString(llvm::StringRef paramsStr);

  /// Parse and add color bindings from a string like "x_in:1,ax_out:3".
  LogicalResult addColorsFromString(llvm::StringRef colorsStr);

  //===--------------------------------------------------------------------===//
  // Comparison
  //===--------------------------------------------------------------------===//

  bool operator==(const CslFlags &other) const;

  //===--------------------------------------------------------------------===//
  // Context configuration
  //===--------------------------------------------------------------------===//

  /// Configure an MLIRContext based on these flags (e.g. set multithreading).
  /// Returns a thread pool if one was created for this context, which
  /// the caller is responsible for keeping alive as long as the context is
  /// alive.
  [[nodiscard]] std::unique_ptr<llvm::ThreadPoolInterface>
  configureContext(MLIRContext *ctx) const;
};

/// Stream operator for CslFlags (uses the semicolon-separated format).
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CslFlags &flags);

//===----------------------------------------------------------------------===//
// CLI Registration
//===----------------------------------------------------------------------===//

/// Register command-line options for CSL tools. Must be called before
/// cl::ParseCommandLineOptions().
void registerCslCLOptions();

/// If `envName` is set in the environment, tokenize its value as a GNU-style
/// command line (via cl::TokenizeGNUCommandLine) and splice the resulting
/// tokens in just after argv[0]. The returned vector is suitable to pass to
/// cl::ParseCommandLineOptions(). CLI args from `argv` follow the env tokens
/// so that, for "last-value-wins" scalar options, the CLI takes precedence
/// while list options accumulate from both sources.
///
/// `saver` must outlive the returned vector — tokens are allocated through it.
llvm::SmallVector<const char *>
addCommandLineOptionsFromEnv(int argc, const char *const *argv,
                             llvm::StringSaver &saver,
                             const char *envName = "CSL_FLAGS");

/// Convert enum flag values to/from their CLI string representation.
/// Generated from CslFlagsFields.def for all CSL_FLAG_ENUM types.
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...)                         \
  llvm::StringRef stringify##Type(Type val);                                   \
  std::optional<Type> parse##Type(llvm::StringRef str);                        \
  llvm::cl::ValuesClass enumValuesFor##Type();
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

/// Option category for shared CSL flags (--arch, --import-path, etc.).
/// Tools should pass this (along with their own tool-specific category)
/// to cl::HideUnrelatedOptions() to suppress MLIR/LLVM internal options
/// from --help output.
llvm::cl::OptionCategory &getCslOptionCategory();

/// Parse a "width,height" dimension string into a Dimensions struct.
/// Returns failure if parsing fails or values are out of range.
FailureOr<Dimensions> parseDimensions(llvm::StringRef str,
                                      llvm::StringRef optionName);

} // namespace csl
} // namespace mlir

//===----------------------------------------------------------------------===//
// LLVM CL Parsers (enables CLI and MLIR pass option support)
//===----------------------------------------------------------------------===//

namespace llvm {
namespace cl {

template <>
class parser<mlir::csl::CslFlags> : public basic_parser<mlir::csl::CslFlags> {
public:
  using OptVal = OptionValue<mlir::csl::CslFlags>;

  parser(Option &opt) : basic_parser(opt) {}

  bool parse(Option &O, StringRef ArgName, StringRef Arg,
             mlir::csl::CslFlags &Val);

  StringRef getValueName() const override { return "csl-flags"; }

  static void print(raw_ostream &os, const mlir::csl::CslFlags &val);

  void printOptionDiff(const Option &O, const mlir::csl::CslFlags &V,
                       const OptVal &Default, size_t GlobalWidth) const;
};

template <>
class parser<mlir::csl::Dimensions>
    : public basic_parser<mlir::csl::Dimensions> {
public:
  using OptVal = OptionValue<mlir::csl::Dimensions>;

  parser(Option &opt) : basic_parser(opt) {}

  bool parse(Option &O, StringRef ArgName, StringRef Arg,
             mlir::csl::Dimensions &Val);

  StringRef getValueName() const override { return "width,height"; }

  static void print(raw_ostream &os, const mlir::csl::Dimensions &val);

  void printOptionDiff(const Option &O, const mlir::csl::Dimensions &V,
                       const OptVal &Default, size_t GlobalWidth) const;
};

} // namespace cl
} // namespace llvm

#endif // CSL_MLIR_DIALECT_CSL_UTILS_COMMANDLINEOPTIONS_H
