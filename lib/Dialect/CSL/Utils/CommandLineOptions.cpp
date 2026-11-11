//===- CommandLineOptions.cpp - Shared CSL Tool Options -------------------===//
//
// Implementation of shared command-line options for CSL tools.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Support/Sif/SifFileSystem.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

using namespace mlir;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// Enum stringify helpers (generated from CslFlagsFields.def)
//===----------------------------------------------------------------------===//

#define CSL_ENUM_VAL(enumVal, cliName, cliDesc) {enumVal, cliName}
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...)                         \
  llvm::StringRef mlir::csl::stringify##Type(Type val) {                       \
    struct E {                                                                 \
      Type v;                                                                  \
      const char *s;                                                           \
    };                                                                         \
    E entries[] = {__VA_ARGS__};                                               \
    for (const auto &e : entries)                                              \
      if (e.v == val)                                                          \
        return e.s;                                                            \
    llvm_unreachable("unknown " #Type " value");                               \
  }                                                                            \
  std::optional<Type> mlir::csl::parse##Type(llvm::StringRef str) {            \
    struct E {                                                                 \
      Type v;                                                                  \
      const char *s;                                                           \
    };                                                                         \
    E entries[] = {__VA_ARGS__};                                               \
    for (const auto &e : entries)                                              \
      if (str == e.s)                                                          \
        return e.v;                                                            \
    return std::nullopt;                                                       \
  }
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

// cl::ValuesClass factories (separate .def expansion because CSL_ENUM_VAL
// needs to expand to clEnumValN here, not the struct-based form above).
#define CSL_ENUM_VAL(enumVal, cliName, cliDesc)                                \
  clEnumValN(enumVal, cliName, cliDesc)
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...)                         \
  llvm::cl::ValuesClass mlir::csl::enumValuesFor##Type() {                     \
    return llvm::cl::values(__VA_ARGS__);                                      \
  }
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

//===----------------------------------------------------------------------===//
// DiagLevel helpers
//===----------------------------------------------------------------------===//

std::optional<DiagLevel> mlir::csl::parseDiagLevel(llvm::StringRef str) {
  if (str == "off")
    return DiagLevel::Off;
  if (str == "warn")
    return DiagLevel::Warn;
  if (str == "error")
    return DiagLevel::Error;
  return std::nullopt;
}

llvm::StringRef mlir::csl::diagLevelToString(DiagLevel level) {
  switch (level) {
  case DiagLevel::Off:
    return "off";
  case DiagLevel::Warn:
    return "warn";
  case DiagLevel::Error:
    return "error";
  }
  llvm_unreachable("unknown DiagLevel");
}

//===----------------------------------------------------------------------===//
// CslToolOptions - Internal struct holding cl::opt instances
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory cslCategory("CSL Options");

namespace {

struct CslToolOptions {
  // Scalar flags (from .def)
  // Note: 'flagDesc' avoids collision with llvm::cl::desc.
#define CSL_FLAG(Type, name, key, def, flagDesc)                               \
  llvm::cl::opt<Type> name{key, llvm::cl::desc(flagDesc), llvm::cl::init(def), \
                           llvm::cl::cat(cslCategory)};
#define CSL_FLAG_ENUM(Type, name, key, def, flagDesc, ...)                     \
  llvm::cl::opt<Type> name{key, llvm::cl::desc(flagDesc), llvm::cl::init(def), \
                           enumValuesFor##Type(), llvm::cl::cat(cslCategory)};
#define CSL_FLAG_LIST(ElemType, name, key, flagDesc)                           \
  llvm::cl::list<ElemType> name{key, llvm::cl::desc(flagDesc),                 \
                                llvm::cl::cat(cslCategory)};
#define CSL_FLAG_DIAG(name, diagKey, def, flagDesc)                            \
  llvm::cl::opt<std::string> diag_##name{                                      \
      "W" diagKey,                                                             \
      llvm::cl::desc("Diagnostic level for " flagDesc                          \
                     " (off, warn, or error)"),                                \
      llvm::cl::value_desc("level"), llvm::cl::init(def),                      \
      llvm::cl::cat(cslCategory)};
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // Manual options (not in .def)
  llvm::cl::list<std::string> params{
      "params",
      llvm::cl::desc(
          "Comma-separated parameter bindings (NAME:VALUE,...). "
          "Values containing '.' or 'e/E' are comptime_float, "
          "otherwise comptime_int. Can be specified multiple times."),
      llvm::cl::value_desc("NAME:VALUE,..."), llvm::cl::cat(cslCategory)};

  llvm::cl::list<std::string> colors{
      "colors",
      llvm::cl::desc("Comma-separated color bindings (NAME:VALUE,...). "
                     "Values are color IDs (uint16). "
                     "Can be specified multiple times."),
      llvm::cl::value_desc("NAME:VALUE,..."), llvm::cl::cat(cslCategory)};
};

} // namespace

static llvm::ManagedStatic<CslToolOptions> clOptions;

//===----------------------------------------------------------------------===//
// registerCslCLOptions
//===----------------------------------------------------------------------===//

void mlir::csl::registerCslCLOptions() {
  // Force construction of the options struct, which registers the cl::opts.
  *clOptions;
}

llvm::cl::OptionCategory &mlir::csl::getCslOptionCategory() {
  return cslCategory;
}

//===----------------------------------------------------------------------===//
// addCommandLineOptionsFromEnv
//===----------------------------------------------------------------------===//

llvm::SmallVector<const char *>
mlir::csl::addCommandLineOptionsFromEnv(int argc, const char *const *argv,
                                        llvm::StringSaver &saver,
                                        const char *envName) {
  llvm::SmallVector<const char *> result;
  result.reserve(argc + 8);

  // argv[0] (program name) always comes first.
  if (argc > 0)
    result.push_back(argv[0]);

  // Tokenize env var (if set) and splice in just after argv[0]. Env tokens
  // precede CLI tokens so that "last-value-wins" scalar options let the CLI
  // override; list options accumulate from both.
  if (std::optional<std::string> env = llvm::sys::Process::GetEnv(envName);
      env && !env->empty()) {
    llvm::cl::TokenizeGNUCommandLine(*env, saver, result);
  }

  // Remaining CLI args.
  for (int i = 1; i < argc; ++i)
    result.push_back(argv[i]);

  return result;
}

//===----------------------------------------------------------------------===//
// parseDimensions
//===----------------------------------------------------------------------===//

FailureOr<Dimensions> mlir::csl::parseDimensions(llvm::StringRef str,
                                                 llvm::StringRef optionName) {
  if (str.empty())
    return Dimensions{0, 0};

  auto commaPos = str.find(',');
  if (commaPos == llvm::StringRef::npos) {
    llvm::errs() << "Invalid " << optionName << " format '" << str
                 << "', expected 'width,height'\n";
    return failure();
  }

  llvm::StringRef widthStr = str.substr(0, commaPos).trim();
  llvm::StringRef heightStr = str.substr(commaPos + 1).trim();

  unsigned width, height;
  if (widthStr.getAsInteger(10, width) || heightStr.getAsInteger(10, height)) {
    llvm::errs() << "Invalid " << optionName << " format '" << str
                 << "', values must be integers\n";
    return failure();
  }

  if (width > UINT16_MAX || height > UINT16_MAX) {
    llvm::errs() << "Invalid " << optionName << " values, must fit in uint16\n";
    return failure();
  }

  return Dimensions{static_cast<uint16_t>(width),
                    static_cast<uint16_t>(height)};
}

//===----------------------------------------------------------------------===//
// cl::parser<Dimensions>
//===----------------------------------------------------------------------===//

bool llvm::cl::parser<Dimensions>::parse(llvm::cl::Option &O,
                                         llvm::StringRef /*ArgName*/,
                                         llvm::StringRef Arg, Dimensions &Val) {
  auto result = parseDimensions(Arg, O.ArgStr);
  if (failed(result))
    return true; // error
  Val = *result;
  return false; // success
}

void llvm::cl::parser<Dimensions>::print(llvm::raw_ostream &os,
                                         const Dimensions &val) {
  if (val)
    os << val.width << "," << val.height;
}

void llvm::cl::parser<Dimensions>::printOptionDiff(const llvm::cl::Option &O,
                                                   const Dimensions &V,
                                                   const OptVal &Default,
                                                   size_t GlobalWidth) const {
  // Minimal implementation; MLIR pass option infrastructure doesn't depend on
  // a rich diff.
}

//===----------------------------------------------------------------------===//
// CslFlags implementation
//===----------------------------------------------------------------------===//

llvm::Error CslFlags::useSystemModuleSif(llvm::StringRef sifPath) {
  // The system modules are read from the SDK image, so an image that cannot
  // be read would only show up later as modules that are not found.
  auto sifFS = ::csl::sif::SifFileSystem::create(sifPath);
  if (!sifFS) {
    return llvm::createStringError(
        "cannot read the Cerebras SDK image '" + llvm::Twine(sifPath) +
        "': " + llvm::toString(sifFS.takeError()));
  }
  systemModuleSif = sifPath.str();
  if (systemImportPaths.empty())
    systemImportPaths.emplace_back("/cb/toolchains/cslang/*/*/csl-libs/");
  return llvm::Error::success();
}

void CslFlags::addInstalledLibraries(const char *argv0) {
  // The address of any function in the executable lets getMainExecutable
  // find it on platforms without /proc/self/exe.
  static int anchor;
  std::string executable = llvm::sys::fs::getMainExecutable(argv0, &anchor);
  if (executable.empty())
    return;
  llvm::SmallString<256> libraries(
      llvm::sys::path::parent_path(llvm::sys::path::parent_path(executable)));
  llvm::sys::path::append(libraries, "share", "csl-mlir", "libs");
  if (llvm::sys::fs::is_directory(libraries))
    importPaths.emplace_back(libraries.str());
}

llvm::Expected<CslFlags> CslFlags::fromCommandLine() {
  CslFlags flags;
  if (!clOptions.isConstructed())
    return llvm::createStringError("cl options not yet parsed");

  // Copy all flags from cl::opts (from .def)
#define CSL_FLAG(Type, name, key, def, desc) flags.name = clOptions->name;
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...)                         \
  flags.name = clOptions->name;
#define CSL_FLAG_LIST(ElemType, name, key, desc)                               \
  for (const auto &elem : clOptions->name)                                     \
    flags.name.push_back(elem);
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // Diagnostic flags (from .def)
#define CSL_FLAG_DIAG(name, diagKey, def, desc)                                \
  {                                                                            \
    llvm::StringRef levelStr = clOptions->diag_##name;                         \
    auto level = parseDiagLevel(levelStr);                                     \
    if (!level) {                                                              \
      return llvm::createStringError("Invalid -W" diagKey " value '" +         \
                                     llvm::Twine(levelStr) + "'");             \
    }                                                                          \
    flags.diagnostics.name = *level;                                           \
  }
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // These profiling scopes only have a meaning for the fiber backend: they
  // either keep its runtime out of line or give each @await_parallel site an
  // ordinary function that the backend can represent as a DWARF inline scope.
  // The stackless backend has no separable runtime -- its overhead lives in
  // the state machines the transform emits -- so accepting the flag there
  // would silently measure something else.
  if (flags.coroInstrument != CoroInstrument::None &&
      flags.coroMode != CoroMode::Fiber) {
    return llvm::createStringError(
        "--coro-instrument requires --coro-mode=fiber; the stackless backend "
        "has no separable runtime to attribute cycles to");
  }

  if (!flags.systemModuleSif.empty()) {
    std::string sifPath = flags.systemModuleSif;
    if (llvm::Error err = flags.useSystemModuleSif(sifPath))
      return err;
  }

  // Parse params option (may be specified multiple times, merge all)
  for (const auto &paramStr : clOptions->params) {
    auto parsedParams = ParsedParams::parseParams(paramStr);
    if (failed(parsedParams)) {
      return llvm::createStringError("Failed to parse --params option value '" +
                                     llvm::Twine(paramStr) + "'");
    }
    flags.addParams(*parsedParams);
  }

  // Parse colors option (may be specified multiple times, merge all)
  for (const auto &colorStr : clOptions->colors) {
    auto parsedColors = ParsedParams::parseColors(colorStr);
    if (failed(parsedColors)) {
      return llvm::createStringError("Failed to parse --colors option value '" +
                                     llvm::Twine(colorStr) + "'");
    }
    flags.addParams(*parsedColors);
  }

  return flags;
}

LogicalResult CslFlags::addParamsFromString(llvm::StringRef paramsStr) {
  auto parsedParams = ParsedParams::parseParams(paramsStr);
  if (failed(parsedParams))
    return failure();
  addParams(*parsedParams);
  return success();
}

LogicalResult CslFlags::addColorsFromString(llvm::StringRef colorsStr) {
  auto parsedColors = ParsedParams::parseColors(colorsStr);
  if (failed(parsedColors))
    return failure();
  addParams(*parsedColors);
  return success();
}

//===----------------------------------------------------------------------===//
// CslFlags::operator==
//===----------------------------------------------------------------------===//

bool CslFlags::operator==(const CslFlags &other) const {
  // Flags from .def
#define CSL_FLAG(Type, name, key, def, desc)                                   \
  if (!(name == other.name))                                                   \
    return false;
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...)                         \
  if (name != other.name)                                                      \
    return false;
#define CSL_FLAG_LIST(ElemType, name, key, desc)                               \
  if (name != other.name)                                                      \
    return false;
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // Diagnostic flags
  if (!(diagnostics == other.diagnostics))
    return false;

  // Manual fields
  if (!(params == other.params))
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
// CslFlags::configureContext
//===----------------------------------------------------------------------===//

std::unique_ptr<llvm::ThreadPoolInterface>
CslFlags::configureContext(MLIRContext *ctx) const {
  std::unique_ptr<llvm::ThreadPoolInterface> threadPool;

  if (numThreads != 1) {

    // Enable multithreading. When ThreadsRequested is 0, the thread pool will
    // auto-detect the number of threads to use.
    ctx->disableMultithreading();
    llvm::ThreadPoolStrategy strategy;
    strategy.ThreadsRequested = numThreads;
    threadPool = std::make_unique<llvm::DefaultThreadPool>(strategy);
    ctx->setThreadPool(*threadPool);
    ctx->enableMultithreading();
  } else {
    // Disable multithreading
    ctx->disableMultithreading(true);
  }

  ctx->printOpOnDiagnostic(false);
  return threadPool;
}

//===----------------------------------------------------------------------===//
// Flag print/parse helpers — overloaded for each type
//===----------------------------------------------------------------------===//

namespace {

/// Parse helpers — overloaded for each flag type.
llvm::Expected<bool> parseFlag(llvm::StringRef val, llvm::StringRef key,
                               std::string &out) {
  out = val.str();
  return true;
}

llvm::Expected<bool> parseFlag(llvm::StringRef val, llvm::StringRef key,
                               Dimensions &out) {
  auto dims = parseDimensions(val, key);
  if (failed(dims))
    return llvm::createStringError("Invalid " + llvm::Twine(key) +
                                   " value: " + llvm::Twine(val));
  out = *dims;
  return true;
}

llvm::Expected<bool> parseFlag(llvm::StringRef val, llvm::StringRef key,
                               bool &out) {
  if (val == "true")
    out = true;
  else if (val == "false")
    out = false;
  else
    return llvm::createStringError("Invalid " + llvm::Twine(key) +
                                   " value: " + llvm::Twine(val));
  return true;
}

llvm::Expected<bool> parseFlag(llvm::StringRef val, llvm::StringRef key,
                               unsigned &out) {
  if (val.getAsInteger(10, out))
    return llvm::createStringError("Invalid " + llvm::Twine(key) +
                                   " value: " + llvm::Twine(val));
  return true;
}

void parseFlagList(llvm::StringRef val, std::vector<std::string> &out) {
  llvm::SmallVector<llvm::StringRef> parts;
  val.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef p : parts)
    out.push_back(p.str());
}

/// Print helpers — overloaded for each flag type.
void printFlag(llvm::raw_ostream &os, llvm::StringRef key,
               const std::string &val, const std::string &def, bool &first) {
  if (val == def)
    return;
  if (!first)
    os << ';';
  os << key << '=' << val;
  first = false;
}

void printFlag(llvm::raw_ostream &os, llvm::StringRef key,
               const Dimensions &val, const Dimensions &def, bool &first) {
  if (val == def)
    return;
  if (!first)
    os << ';';
  os << key << '=' << val.width << ',' << val.height;
  first = false;
}

void printFlag(llvm::raw_ostream &os, llvm::StringRef key, bool val, bool def,
               bool &first) {
  if (val == def)
    return;
  if (!first)
    os << ';';
  os << key << '=' << (val ? "true" : "false");
  first = false;
}

void printFlag(llvm::raw_ostream &os, llvm::StringRef key, unsigned val,
               unsigned def, bool &first) {
  if (val == def)
    return;
  if (!first)
    os << ';';
  os << key << '=' << val;
  first = false;
}

void printFlagList(llvm::raw_ostream &os, llvm::StringRef key,
                   const std::vector<std::string> &val, bool &first) {
  if (val.empty())
    return;
  if (!first)
    os << ';';
  os << key << '=';
  for (size_t i = 0; i < val.size(); ++i) {
    if (i > 0)
      os << ',';
    os << val[i];
  }
  first = false;
}

} // namespace

void CslFlags::print(llvm::raw_ostream &os) const {
  CslFlags defaults;
  bool first = true;

  // Flags from .def
#define CSL_FLAG(Type, name, key, def, desc)                                   \
  printFlag(os, key, name, defaults.name, first);
#define CSL_ENUM_VAL(enumVal, cliName, cliDesc) {enumVal, cliName}
#define CSL_FLAG_ENUM(Type, name, key, def, desc, ...)                         \
  if (name != defaults.name) {                                                 \
    struct E {                                                                 \
      Type v;                                                                  \
      const char *s;                                                           \
    };                                                                         \
    E entries[] = {__VA_ARGS__};                                               \
    const char *valStr = "unknown";                                            \
    for (const auto &e : entries)                                              \
      if (e.v == name)                                                         \
        valStr = e.s;                                                          \
    printFlag(os, key, std::string(valStr), std::string(""), first);           \
  }
#define CSL_FLAG_LIST(ElemType, name, key, desc)                               \
  printFlagList(os, key, name, first);
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // Diagnostic flags (from .def)
#define CSL_FLAG_DIAG(name, diagKey, def, desc)                                \
  if (diagnostics.name != *parseDiagLevel(def)) {                              \
    if (!first)                                                                \
      os << ';';                                                               \
    os << "W" diagKey "=" << diagLevelToString(diagnostics.name);              \
    first = false;                                                             \
  }
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

  // Manual: params
  if (!params.values.empty()) {
    if (!first)
      os << ';';
    os << "params=" << params.toString();
    first = false;
  }
}

llvm::raw_ostream &mlir::csl::operator<<(llvm::raw_ostream &os,
                                         const CslFlags &flags) {
  flags.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// CslFlags::parseFromString
//===----------------------------------------------------------------------===//

llvm::Expected<CslFlags> CslFlags::parseFromString(llvm::StringRef str) {
  CslFlags flags;

  if (str.empty())
    return flags;

  llvm::SmallVector<llvm::StringRef> pairs;
  str.split(pairs, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  for (llvm::StringRef pair : pairs) {
    auto eqPos = pair.find('=');
    if (eqPos == llvm::StringRef::npos)
      return llvm::createStringError("Invalid CslFlags format, missing '=' "
                                     "in pair: '" +
                                     llvm::Twine(pair) + "'");

    llvm::StringRef key = pair.substr(0, eqPos).trim();
    llvm::StringRef val = pair.substr(eqPos + 1).trim();

    // Scalar, enum, and list flags (from .def)
    bool matched = false;
#define CSL_FLAG(Type, name, flagKey, def, desc)                               \
  if (!matched && key == flagKey) {                                            \
    auto parsed = parseFlag(val, flagKey, flags.name);                         \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    matched = true;                                                            \
  }
#define CSL_ENUM_VAL(enumVal, cliName, cliDesc) {enumVal, cliName}
#define CSL_FLAG_ENUM(Type, name, flagKey, def, desc, ...)                     \
  if (!matched && key == flagKey) {                                            \
    struct E {                                                                 \
      Type v;                                                                  \
      const char *s;                                                           \
    };                                                                         \
    E entries[] = {__VA_ARGS__};                                               \
    bool enumMatched = false;                                                  \
    for (const auto &e : entries)                                              \
      if (val == e.s) {                                                        \
        flags.name = e.v;                                                      \
        enumMatched = true;                                                    \
        break;                                                                 \
      }                                                                        \
    if (!enumMatched)                                                          \
      return llvm::createStringError("Invalid " + llvm::Twine(flagKey) +       \
                                     " value: " + llvm::Twine(val));           \
    matched = true;                                                            \
  }
#define CSL_FLAG_LIST(ElemType, name, flagKey, desc)                           \
  if (!matched && key == flagKey) {                                            \
    parseFlagList(val, flags.name);                                            \
    matched = true;                                                            \
  }
#define CSL_FLAG_DIAG(name, diagKey, def, desc)                                \
  if (!matched && key == "W" diagKey) {                                        \
    auto level = parseDiagLevel(val);                                          \
    if (!level)                                                                \
      return llvm::createStringError("Invalid W" diagKey " value: " +          \
                                     llvm::Twine(val));                        \
    flags.diagnostics.name = *level;                                           \
    matched = true;                                                            \
  }
#include "csl-mlir/Dialect/CSL/Utils/CslFlagsFields.def"

    // Manual: params
    if (!matched && key == "params") {
      auto parsedParams = ParsedParams::parseParams(val);
      if (failed(parsedParams))
        return llvm::createStringError("Invalid params value: " +
                                       llvm::Twine(val));
      flags.addParams(*parsedParams);
      matched = true;
    }

    if (!matched) {
      return llvm::createStringError("Unknown CslFlags key: '" +
                                     llvm::Twine(key) + "'");
    }
  }

  return flags;
}

//===----------------------------------------------------------------------===//
// cl::parser<CslFlags>
//===----------------------------------------------------------------------===//

bool llvm::cl::parser<CslFlags>::parse(llvm::cl::Option &O,
                                       llvm::StringRef /*ArgName*/,
                                       llvm::StringRef Arg, CslFlags &Val) {
  auto result = CslFlags::parseFromString(Arg);
  if (!result) {
    llvm::consumeError(result.takeError());
    return true; // error
  }
  Val = std::move(*result);
  return false; // success
}

void llvm::cl::parser<CslFlags>::print(llvm::raw_ostream &os,
                                       const CslFlags &val) {
  val.print(os);
}

void llvm::cl::parser<CslFlags>::printOptionDiff(const llvm::cl::Option &O,
                                                 const CslFlags &V,
                                                 const OptVal &Default,
                                                 size_t GlobalWidth) const {
  // Minimal implementation.
}
