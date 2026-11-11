//===- CompilationContext.cpp - Layout-Driven Compilation Context
//----------===//
//
// Manages a layout file and its referenced PE programs within a shared
// MLIRContext for the LSP server.
//
//===----------------------------------------------------------------------===//

#include "CompilationContext.h"
#include "LspUtils.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CslModuleResolver.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Target/CSL/ParseCSL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "csl-lsp-server"

// Avoid `using namespace mlir` to prevent ambiguity between mlir::Diagnostic /
// mlir::Location and the LSP equivalents.
namespace lsp = llvm::lsp;
using mlir::csl::lsp_utils::getRange;
using mlir::csl::lsp_utils::rangeSize;

/// Build a parameter signature string for a GenericFuncOp, e.g.
/// "(comptime T, x, y: anytype)".
static std::string
formatGenericFuncParams(mlir::parsed_csl::GenericFuncOp func) {
  auto comptimeParams = func.getComptimeParams();
  auto anytypeParams = func.getAnytypeParams();
  auto paramNamesAttr = func.getParamNames();
  int32_t numParams = func.getNumParams();

  llvm::SmallDenseSet<int32_t> comptimeSet(comptimeParams.begin(),
                                           comptimeParams.end());
  llvm::SmallDenseSet<int32_t> anytypeSet(anytypeParams.begin(),
                                          anytypeParams.end());

  std::string result = "(";
  for (int32_t i = 0; i < numParams; ++i) {
    if (i > 0)
      result += ", ";
    if (comptimeSet.contains(i))
      result += "comptime ";
    if (paramNamesAttr && static_cast<size_t>(i) < paramNamesAttr->size())
      result += mlir::cast<mlir::StringAttr>((*paramNamesAttr)[i]).getValue();
    else
      result += "_";
    if (anytypeSet.contains(i))
      result += ": anytype";
  }
  result += ")";
  return result;
}

//===----------------------------------------------------------------------===//
// FileState
//===----------------------------------------------------------------------===//

mlir::csl::DocumentView
mlir::csl::FileState::toView(llvm::StringRef filterFilePath) const {
  DocumentView view;
  view.compilationUnit = compilationUnit;
  view.analysis = analysis.get();
  view.filePath = filterFilePath.empty() ? filePath : filterFilePath.str();
  view.parsedSuccessfully = parsedSuccessfully;
  return view;
}

//===----------------------------------------------------------------------===//
// parseAndAnalyze
//===----------------------------------------------------------------------===//

void mlir::csl::parseAndAnalyze(
    mlir::MLIRContext *ctx, FileState &state, llvm::StringRef contents,
    const CslFlags &flags,
    const llvm::DenseMap<llvm::StringRef, csl::ParamBinding> &paramBindings,
    const llvm::StringMap<std::string> &overrideBuffers,
    std::shared_ptr<SystemModuleCache> systemModuleCache,
    std::function<bool(llvm::StringRef)> shouldEagerlyEvaluate) {

  // Set up module resolver.
  state.moduleResolver =
      systemModuleCache
          ? lsp_utils::createModuleResolver(state.filePath, flags,
                                            std::move(systemModuleCache))
          : lsp_utils::createModuleResolver(state.filePath, flags);
  state.moduleResolver->applyEditorOverlay(overrideBuffers);

  // Add the source buffer.
  unsigned bufferID = state.moduleResolver->addBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(contents, state.filePath));

  // Create module and compilation unit.
  state.module = mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx));
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(state.module->getBody());
  state.compilationUnit = csl::CompilationUnitOp::create(
      builder, mlir::FileLineColLoc::get(ctx, state.filePath, 0, 0));

  // Parse. In LSP mode, parseCSLIntoRegion generates partial IR even on
  // failure, so we always continue to comptime analysis.
  (void)csl::parseCSLIntoRegion(state.moduleResolver->getSourceMgr(), bufferID,
                                state.compilationUnit.getRegion(),
                                paramBindings, /*lineOffset=*/0, state.filePath,
                                /*currentModule=*/nullptr, /*lspMode=*/true,
                                flags);
  state.parsedSuccessfully = true;

  // Check for layout block.
  state.compilationUnit->walk([&](mlir::parsed_csl::LayoutBlockOp) {
    state.hasLayoutBlock = true;
    return mlir::WalkResult::interrupt();
  });

  // Run comptime analysis.
  mlir::parsed_csl::ComptimeEvalOptions evalOpts(flags);
  evalOpts.lspMode = true;
  evalOpts.shouldEagerlyEvaluate = std::move(shouldEagerlyEvaluate);
  state.analysis = std::make_unique<mlir::parsed_csl::ComptimeEvaluator>(
      state.compilationUnit, state.moduleResolver, evalOpts);
}

//===----------------------------------------------------------------------===//
// DocumentView
//===----------------------------------------------------------------------===//

/// Find the ModuleOp or CompilationUnitOp whose location filename matches
/// the target file. Only searches through top-level containers (ModuleOps
/// nested inside other ModuleOps or the CompilationUnitOp).
static mlir::Operation *
findContainerForFile(mlir::csl::CompilationUnitOp compUnit,
                     llvm::StringRef filePath) {
  // Check if the CompilationUnitOp itself matches.
  if (auto flc = mlir::dyn_cast<mlir::FileLineColRange>(compUnit->getLoc()))
    if (flc.getFilename().getValue() == filePath)
      return compUnit;

  // Search nested ModuleOps by source_file attribute, not location.
  // The ModuleOp's location is the @import_module call site, while
  // source_file holds the actual file path of the imported module.
  mlir::Operation *result = nullptr;
  mlir::parsed_csl::walkTopLevelDecls<mlir::parsed_csl::ModuleOp>(
      compUnit, [&](mlir::parsed_csl::ModuleOp mod) -> mlir::WalkResult {
        if (mod.getSourceFile() == filePath) {
          result = mod;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  return result;
}

mlir::Operation *mlir::csl::DocumentView::findOpMatching(
    const lsp::Position &pos,
    llvm::function_ref<bool(mlir::Operation *)> predicate) {
  if (!parsedSuccessfully || !compilationUnit)
    return nullptr;

  // Phase 1: Find the container (CompilationUnitOp or ModuleOp) for our file.
  mlir::Operation *container = findContainerForFile(compilationUnit, filePath);
  if (!container)
    return nullptr;

  // Phase 2: Walk within the container, pruning by location range.
  // Skip into nested ModuleOps from other files — their ops may have
  // overlapping line/column numbers but belong to a different file.
  mlir::Operation *best = nullptr;
  std::pair<int, int> bestSize = {INT_MAX, INT_MAX};
  container->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *op) -> mlir::WalkResult {
        // Skip GenericFuncOp bodies — they have no comptime analysis results.
        // The instantiated FuncOps (at the same location) have the data we
        // need.
        if (mlir::isa<mlir::parsed_csl::GenericFuncOp>(op))
          return mlir::WalkResult::skip();

        // Skip imported modules — they belong to different files.
        // But if the container itself is a ModuleOp for this file (e.g.
        // csl-sdk/system modules), we must still descend into it.
        if (mlir::isa<mlir::parsed_csl::ModuleOp>(op) && op != container)
          return mlir::WalkResult::skip();

        auto range = getRange(op->getLoc());
        if (!range || !range->contains(pos))
          return mlir::WalkResult::skip();
        if (predicate(op)) {
          auto size = rangeSize(*range);
          if (size <= bestSize) {
            best = op;
            bestSize = size;
          }
        }
        return mlir::WalkResult::advance();
      });
  return best;
}

mlir::Operation *
mlir::csl::DocumentView::findOpAtPosition(const lsp::Position &pos) {
  auto *result = findOpMatching(pos, [](mlir::Operation *) { return true; });
  // A read has the location of the expression it reads, which is what the
  // position refers to.
  while (auto readOp = mlir::dyn_cast_or_null<mlir::parsed_csl::ReadOp>(result)) {
    mlir::Operation *source = readOp.getSource().getDefiningOp();
    if (!source)
      break;
    result = source;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "findOpAtPosition(" << pos.line << ":" << pos.character
                 << ") -> ";
    if (result)
      llvm::dbgs() << result->getName() << " at " << result->getLoc() << "\n";
    else
      llvm::dbgs() << "(none)\n";
  });
  return result;
}

mlir::Operation *mlir::csl::DocumentView::resolveSymbol(llvm::StringRef name) {
  if (!compilationUnit)
    return nullptr;
  return mlir::SymbolTable::lookupSymbolIn(compilationUnit, name);
}

std::optional<std::string>
mlir::csl::DocumentView::formatHoverInfo(mlir::Operation *op) {
  using namespace mlir;
  using namespace mlir::csl;
  using namespace mlir::parsed_csl;

  if (!analysis)
    return std::nullopt;

  // LocOp is a transparent wrapper — forward to the inner value's defining op.
  if (auto locOp = mlir::dyn_cast<LocOp>(op)) {
    if (auto *innerOp = locOp.getValue().getDefiningOp()) {
      return formatHoverInfo(innerOp);
    } else if (auto blockArg = dyn_cast<BlockArgument>(locOp.getValue())) {
      return std::nullopt;
    }
    return std::nullopt;
  }

  // Handle ops without SSA results first (FuncOp, GenericFuncOp, GlobalOp).
  if (auto func = mlir::dyn_cast<FuncOp>(op)) {
    std::string info = "(function) " + func.getOriginalName().str() + ": " +
                       prettyPrintType(func.getFunctionType());
    if (auto doc = func.getDocString())
      info += "\n\n" + doc->str();
    return info;
  }

  if (auto genericFunc = mlir::dyn_cast<GenericFuncOp>(op)) {
    std::string info = "(generic function) " + genericFunc.getSymName().str() +
                       formatGenericFuncParams(genericFunc);
    if (auto doc = genericFunc.getDocString())
      info += "\n\n" + doc->str();
    return info;
  }

  if (auto global = mlir::dyn_cast<GlobalOp>(op)) {
    std::string label =
        global.getIsConst() ? "(global const) " : "(global var) ";
    label += global.getSymName().str();
    if (auto type = global.getInferredType())
      label += ": " + prettyPrintType(unwrapLValueType(*type));
    if (auto initAttr = global.getInferredInitializer())
      label += " = " + prettyPrintValue(*initAttr);
    return label;
  }

  if (auto varOp = mlir::dyn_cast<VariableOp>(op)) {
    std::string label = varOp.getIsConst() ? "(const) " : "(var) ";
    label += varOp.getName().str();
    auto inferredType = analysis->getInferredType(varOp.getResult());
    if (succeeded(inferredType))
      label += ": " + prettyPrintType(unwrapLValueType(*inferredType));
    auto rvalue = analysis->getOrLoadComptimeRValue(varOp.getResult());
    if (succeeded(rvalue))
      label += " = " + prettyPrintValue(*rvalue);
    return label;
  }

  // Handle ops with SSA results.
  if (op->getNumResults() > 0) {
    Value result = op->getResult(0);
    // Get inferred type
    auto inferredType = analysis->getInferredType(result);
    if (failed(inferredType))
      return std::nullopt;
    Type type = *inferredType;

    // Handle references to functions
    if (auto rvalue = analysis->getOrLoadComptimeRValue(result);
        succeeded(rvalue)) {
      // If the result is a FuncRefAttr, show the referenced function's info.
      if (auto funcAttr = mlir::dyn_cast<csl::FuncRefAttr>(*rvalue)) {
        auto func = parsed_csl::lookupFuncByRefAttr(funcAttr, compilationUnit);
        if (succeeded(func)) {
          if (std::holds_alternative<parsed_csl::FuncOp>(*func)) {
            auto funcOp = std::get<parsed_csl::FuncOp>(*func);
            std::string info = "(function) " + funcOp.getOriginalName().str() +
                               ": " + prettyPrintType(funcOp.getFunctionType());
            if (auto doc = funcOp.getDocString())
              info += "\n\n" + doc->str();
            return info;
          }
          if (std::holds_alternative<parsed_csl::GenericFuncOp>(*func)) {
            auto genericFunc = std::get<parsed_csl::GenericFuncOp>(*func);
            std::string info;
            // If the FuncRefAttr carries a concrete FunctionType, show it.
            if (auto funcType =
                    mlir::dyn_cast<FunctionType>(funcAttr.getType()))
              info = "(function) " + genericFunc.getSymName().str() + ": " +
                     prettyPrintType(funcType);
            else
              info = "(generic function) " + genericFunc.getSymName().str() +
                     formatGenericFuncParams(genericFunc);
            if (auto doc = genericFunc.getDocString())
              info += "\n\n" + doc->str();
            return info;
          }
        }
      }
    }

    // Try to get comptime value for this result.
    std::string valueStr;
    if (auto comptimeVal = analysis->getOrLoadComptimeRValue(result);
        succeeded(comptimeVal))
      valueStr = " = " + prettyPrintValue(*comptimeVal);

    // Handle references to globals and variables
    if (auto lvalue = analysis->getComptimeLValue(result); succeeded(lvalue)) {
      if (lvalue->isGlobalBased()) {
        return "(global) " + lvalue->getBaseGlobal().getSymName().str() + ": " +
               prettyPrintType(unwrapLValueType(type)) + valueStr;
      } else if (lvalue->isVariableBased()) {
        return "(variable) " + lvalue->getBaseVariable().getName().str() +
               ": " + prettyPrintType(unwrapLValueType(type)) + valueStr;
      }
    }

    if (auto fieldAccess = mlir::dyn_cast<FieldAccessOp>(op)) {
      std::string info = "(field) " + fieldAccess.getFieldName().str() + ": " +
                         prettyPrintType(unwrapLValueType(type));
      return info + valueStr;
    }

    if (auto builtinCall = mlir::dyn_cast<BuiltinCallOp>(op)) {
      return "(builtin) @" + builtinCall.getBuiltinName().str() + ": " +
             prettyPrintType(unwrapLValueType(type)) + valueStr;
    }

    // Generic fallback: show type and comptime value if available.
    if (!valueStr.empty())
      return prettyPrintType(type) + valueStr;
    return prettyPrintType(type);
  }
  return std::nullopt;
}

void mlir::csl::DocumentView::getLocationsOf(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    std::vector<lsp::Location> &locations) {
  LLVM_DEBUG(llvm::dbgs() << "getLocationsOf(" << pos.line << ":"
                          << pos.character << ")\n");
  mlir::Operation *op = findOpAtPosition(pos);
  if (!op) {
    LLVM_DEBUG(llvm::dbgs() << "  no op at position\n");
    return;
  }

  if (!analysis) {
    LLVM_DEBUG(llvm::dbgs() << "  no analysis available\n");
    return;
  }

  // LocOp is a transparent wrapper — navigate to the inner value's definition.
  if (auto locOp = mlir::dyn_cast<mlir::parsed_csl::LocOp>(op)) {
    if (auto *innerOp = locOp.getValue().getDefiningOp()) {
      if (auto range = getRange(innerOp->getLoc()))
        locations.emplace_back(uri, *range);
    } else if (auto blockArg = dyn_cast<BlockArgument>(locOp.getValue())) {
      if (auto range = getRange(blockArg.getLoc()))
        locations.emplace_back(uri, *range);
    }
    return;
  }

  if (op->getNumResults() > 0) {
    // Handle references to functions
    if (auto rvalue = analysis->getOrLoadComptimeRValue(op->getResult(0));
        succeeded(rvalue)) {
      LLVM_DEBUG(llvm::dbgs() << "  rvalue: " << *rvalue << "\n");
      // If the result is a FuncRefAttr, navigate to the referenced function.
      if (auto funcAttr = mlir::dyn_cast<csl::FuncRefAttr>(*rvalue)) {
        auto func = parsed_csl::lookupFuncByRefAttr(funcAttr, compilationUnit);
        if (succeeded(func)) {
          std::visit(
              [&](auto &&func) {
                if (auto loc = lsp_utils::getLocWithScheme(func.getLoc()))
                  locations.emplace_back(*loc);
              },
              *func);
          LLVM_DEBUG(llvm::dbgs() << "  -> func ref resolved, "
                                  << locations.size() << " location(s)\n");
          return;
        }
        LLVM_DEBUG(llvm::dbgs() << "  -> func ref lookup failed\n");
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  rvalue resolution failed\n");
    }

    // Handle references to globals and variables
    if (auto lvalue = analysis->getComptimeLValue(op->getResult(0));
        succeeded(lvalue)) {
      LLVM_DEBUG(llvm::dbgs() << "  lvalue: " << *lvalue << "\n");
      auto resolveLocAndAdd = [&](mlir::Location mlirLoc) {
        if (auto loc = lsp_utils::getLocWithScheme(mlirLoc))
          locations.emplace_back(*loc);
      };
      if (lvalue->isGlobalBased()) {
        resolveLocAndAdd(lvalue->getBaseGlobal()->getLoc());
        return;
      } else if (lvalue->isVariableBased()) {
        resolveLocAndAdd(lvalue->getBaseVariable()->getLoc());
        return;
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  lvalue resolution failed\n");
    }
  }

  // BuiltinCallOp -> jump to referenced file for import_module /
  // set_tile_code.
  if (auto builtinCall = mlir::dyn_cast<mlir::parsed_csl::BuiltinCallOp>(op)) {

    auto navigateToFile = [&](llvm::StringRef targetPath) {
      if (auto targetUri = lsp_utils::uriForPath(targetPath)) {
        lsp::Range fileStart(lsp::Position(0, 0), lsp::Position(0, 0));
        locations.emplace_back(*targetUri, fileStart);
      }
    };

    if (builtinCall.resolveBuiltin() == mlir::csl::BuiltinKind::ImportModule) {
      auto rvalue = analysis->getComptimeRValue(builtinCall.getResult());
      if (mlir::succeeded(rvalue)) {
        if (auto moduleRef =
                mlir::dyn_cast<mlir::parsed_csl::ModuleRefAttr>(*rvalue)) {
          uint64_t targetId = moduleRef.getModuleIdentifier();
          compilationUnit->walk([&](mlir::parsed_csl::ModuleOp moduleOp) {
            if (static_cast<uint64_t>(moduleOp.getModuleIdentifier()) ==
                targetId) {
              navigateToFile(moduleOp.getSourceFile());
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          });
        }
      }
    } else if (builtinCall.resolveBuiltin() ==
               mlir::csl::BuiltinKind::SetTileCode) {
      // @set_tile_code(x, y, "filename.csl", .{params}) — arg[2] is filename.
      if (builtinCall.getArgs().size() >= 3) {
        auto filenameVal =
            analysis->getComptimeRValue(builtinCall.getArgs()[2]);
        if (mlir::succeeded(filenameVal)) {
          if (auto strAttr =
                  mlir::dyn_cast<mlir::csl::ComptimeStringAttr>(*filenameVal)) {
            llvm::SmallString<256> absPath(
                llvm::sys::path::parent_path(filePath));
            llvm::sys::path::append(absPath, strAttr.getValue());
            llvm::sys::fs::make_absolute(absPath);
            llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
            navigateToFile(absPath);
          }
        }
      }
    }
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "  no resolution found, " << locations.size()
                          << " location(s)\n");
}

std::optional<lsp::Hover>
mlir::csl::DocumentView::findHover(const lsp::URIForFile &uri,
                                   const lsp::Position &pos) {
  mlir::Operation *op = findOpAtPosition(pos);
  if (!op)
    return std::nullopt;

  auto info = formatHoverInfo(op);
  if (!info)
    return std::nullopt;

  auto range = getRange(op->getLoc());
  if (!range)
    return std::nullopt;

  lsp::Hover hover(*range);
  hover.contents.kind = lsp::MarkupKind::Markdown;
  hover.contents.value = std::move(*info);
  return hover;
}

void mlir::csl::DocumentView::findDocumentSymbols(
    const lsp::URIForFile &uri, std::vector<lsp::DocumentSymbol> &symbols) {
  if (!parsedSuccessfully || !compilationUnit)
    return;

  for (mlir::Operation &op : compilationUnit.getBody()->getOperations()) {
    // Skip ops from other files (imported modules) when viewing a specific
    // file.
    if (auto flcRange = mlir::dyn_cast<mlir::FileLineColRange>(op.getLoc())) {
      if (flcRange.getFilename().getValue() != filePath)
        continue;
    }

    lsp::DocumentSymbol sym;

    if (auto genericFunc =
            mlir::dyn_cast<mlir::parsed_csl::GenericFuncOp>(op)) {
      sym.name = genericFunc.getSymName().str();
      sym.kind = lsp::SymbolKind::Function;
    } else if (auto func = mlir::dyn_cast<mlir::parsed_csl::FuncOp>(op)) {
      sym.name = func.getSymName().str();
      sym.kind = lsp::SymbolKind::Function;
    } else if (auto global = mlir::dyn_cast<mlir::parsed_csl::GlobalOp>(op)) {
      sym.name = global.getSymName().str();
      sym.kind = global.getIsConst() ? lsp::SymbolKind::Constant
                                     : lsp::SymbolKind::Variable;
    } else {
      continue;
    }

    auto range = getRange(op.getLoc());
    if (!range)
      continue;

    sym.range = *range;
    sym.selectionRange = *range;

    // Add type detail if available.
    if (auto genericFunc =
            mlir::dyn_cast<mlir::parsed_csl::GenericFuncOp>(op)) {
      sym.detail = formatGenericFuncParams(genericFunc);
    } else if (analysis) {
      for (auto result : op.getResults()) {
        auto inferredType = analysis->getInferredType(result);
        if (mlir::succeeded(inferredType)) {
          sym.detail = mlir::csl::prettyPrintType(
              mlir::parsed_csl::unwrapLValueType(*inferredType));
          break;
        }
      }
      if (auto func = mlir::dyn_cast<mlir::parsed_csl::FuncOp>(op))
        sym.detail = mlir::csl::prettyPrintType(func.getFunctionType());
    }

    symbols.push_back(std::move(sym));
  }
}

//===----------------------------------------------------------------------===//
// CompilationContext
//===----------------------------------------------------------------------===//

mlir::csl::CompilationContext::CompilationContext(
    mlir::MLIRContext &ctx, const CslFlags &flags, bool filterOverlappingErrors,
    std::shared_ptr<SystemModuleCache> systemModuleCache)
    : context(ctx), flags(flags),
      filterOverlappingErrors(filterOverlappingErrors),
      systemModuleCache(std::move(systemModuleCache)) {}

void mlir::csl::CompilationContext::analyzeLayout(
    llvm::StringRef filePath, llvm::StringRef contents,
    const llvm::StringMap<std::string> &overrideBuffers) {

  // Copy filePath before destroying old state — the caller may have obtained
  // it from getLayoutPath() which references layout->filePath.
  std::string filePathOwned = filePath.str();
  filePath = filePathOwned;

  // Reset storedDiagnostics: keep empty entries for previously known files
  // so that files that no longer have errors get their diagnostics cleared.
  {
    llvm::StringMap<std::vector<lsp::Diagnostic>> newDiags;
    for (const auto &[path, _] : storedDiagnostics)
      newDiags[path];
    storedDiagnostics = std::move(newDiags);
  }

  // Destroy old state.
  pePrograms.clear();
  importedModuleToOwner.clear();
  layout.reset();

  layout = std::make_unique<FileState>();
  layout->filePath = filePath.str();

  auto diagHandler = lsp_utils::createRoutingDiagHandler(
      &context, layout->filePath, storedDiagnostics, filterOverlappingErrors);

  auto paramBindings = flags.toBindings(&context);
  auto eagerPredicate =
      openFiles_.empty()
          ? std::function<bool(llvm::StringRef)>()
          : [this](llvm::StringRef path) { return openFiles_.count(path); };
  parseAndAnalyze(&context, *layout, contents, flags, paramBindings,
                  overrideBuffers, systemModuleCache, eagerPredicate);

  // Compile PE programs discovered from @set_tile_code.
  compilePEPrograms();
}

void mlir::csl::CompilationContext::compilePEPrograms() {
  if (!layout || !layout->analysis)
    return;

  const auto &globalCtx = layout->analysis->getGlobalContext();
  llvm::StringMap<csl::StructValueAttr> seenFiles;

  for (const auto &call : globalCtx.getRecordedBuiltinCalls()) {
    if (call.builtin != csl::BuiltinKind::SetTileCode)
      continue;

    // @set_tile_code(x, y, filename, params)
    if (call.argValues.size() < 4)
      continue;

    auto filenameAttr =
        mlir::dyn_cast<csl::ComptimeStringAttr>(call.argValues[2]);
    auto paramsAttr = mlir::dyn_cast<csl::StructValueAttr>(call.argValues[3]);
    if (!filenameAttr || !paramsAttr)
      continue;

    // Resolve to absolute path.
    llvm::SmallString<256> absPath(
        llvm::sys::path::parent_path(layout->filePath));
    llvm::sys::path::append(absPath, filenameAttr.getValue());
    (void)llvm::sys::fs::make_absolute(absPath);
    llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
    std::string resolved = std::string(absPath);

    // Take first occurrence per file (skip duplicates).
    if (seenFiles.count(resolved))
      continue;
    seenFiles[resolved] = paramsAttr;

    auto pe = std::make_unique<FileState>();
    pe->filePath = resolved;
    pe->params = paramsAttr;
    pe->paramsLoc = call.location;

    compileSinglePEProgram(*pe);

    pePrograms[resolved] = std::move(pe);
  }

  buildFileMapping();
}

void mlir::csl::CompilationContext::compileSinglePEProgram(FileState &pe) {

  auto diagHandler = lsp_utils::createRoutingDiagHandler(
      &context, pe.filePath, storedDiagnostics, filterOverlappingErrors);

  // Read PE program from disk.
  auto peSourceOrErr = llvm::MemoryBuffer::getFile(pe.filePath);
  if (!peSourceOrErr) {
    lsp::Diagnostic d;
    d.severity = lsp::DiagnosticSeverity::Error;
    d.source = "csl";
    d.message = "Failed to read PE program file: " + pe.filePath + ": " +
                peSourceOrErr.getError().message();
    storedDiagnostics[pe.filePath].push_back(std::move(d));
    return;
  }

  auto bindings = createParamBindings(
      pe.params, pe.paramsLoc.value_or(mlir::UnknownLoc::get(&context)));

  auto eagerPredicate =
      openFiles_.empty()
          ? std::function<bool(llvm::StringRef)>()
          : [this](llvm::StringRef path) { return openFiles_.count(path); };
  parseAndAnalyze(&context, pe, (*peSourceOrErr)->getBuffer(), flags, bindings,
                  /*overrideBuffers=*/{}, systemModuleCache, eagerPredicate);
}

void mlir::csl::CompilationContext::buildFileMapping() {
  importedModuleToOwner.clear();

  // Walk layout's imports.
  if (layout && layout->compilationUnit) {
    layout->compilationUnit->walk([&](mlir::parsed_csl::ModuleOp modOp) {
      llvm::StringRef sourceFile = modOp.getSourceFile();
      if (!sourceFile.empty())
        importedModuleToOwner[sourceFile] = layout->filePath;
    });
  }

  // Walk PE program imports.
  for (auto &[path, pe] : pePrograms) {
    if (!pe->compilationUnit)
      continue;
    pe->compilationUnit->walk([&](mlir::parsed_csl::ModuleOp modOp) {
      llvm::StringRef sourceFile = modOp.getSourceFile();
      if (!sourceFile.empty())
        importedModuleToOwner[sourceFile] = std::string(path);
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "buildFileMapping: " << importedModuleToOwner.size()
                 << " imported module(s):\n";
    for (const auto &[mod, owner] : importedModuleToOwner)
      llvm::dbgs() << "  '" << mod << "' -> owner '" << owner << "'\n";
  });
}

bool mlir::csl::CompilationContext::containsFile(
    llvm::StringRef absPath) const {
  if (layout && layout->filePath == absPath)
    return true;
  if (pePrograms.count(absPath))
    return true;
  if (importedModuleToOwner.count(absPath))
    return true;
  return false;
}

std::optional<mlir::csl::DocumentView>
mlir::csl::CompilationContext::getView(llvm::StringRef absPath) const {
  if (layout && layout->filePath == absPath)
    return layout->toView();

  auto peIt = pePrograms.find(absPath);
  if (peIt != pePrograms.end())
    return peIt->second->toView();

  // Imported module: use the owning compilation unit + analysis,
  // but with the module's file path for filtering.
  auto modIt = importedModuleToOwner.find(absPath);
  if (modIt != importedModuleToOwner.end()) {
    if (layout && modIt->second == layout->filePath)
      return layout->toView(absPath);

    auto ownerIt = pePrograms.find(modIt->second);
    if (ownerIt != pePrograms.end())
      return ownerIt->second->toView(absPath);
  }

  return std::nullopt;
}

void mlir::csl::CompilationContext::getKnownFilePaths(
    llvm::SmallVectorImpl<std::string> &paths) const {
  if (layout)
    paths.push_back(layout->filePath);
  for (const auto &[path, _] : pePrograms)
    paths.push_back(std::string(path));
  for (const auto &[path, _] : importedModuleToOwner)
    paths.push_back(std::string(path));
}

void mlir::csl::CompilationContext::reanalyzePEProgram(
    llvm::StringRef peAbsPath, llvm::StringRef newContents,
    const llvm::StringMap<std::string> &overrideBuffers) {

  auto peIt = pePrograms.find(peAbsPath);
  if (peIt == pePrograms.end())
    return;

  auto &pe = *peIt->second;

  // Pre-populate with empty entries for the PE program and all its currently
  // known imported modules. This ensures files that no longer have errors get
  // their diagnostics cleared.
  storedDiagnostics[pe.filePath].clear();
  for (const auto &[modPath, ownerPath] : importedModuleToOwner) {
    if (ownerPath == pe.filePath)
      storedDiagnostics[std::string(modPath)].clear();
  }

  // Destroy old analysis and compilation unit.
  pe.analysis.reset();
  pe.compilationUnit = nullptr;
  pe.module = nullptr;
  pe.parsedSuccessfully = false;

  auto diagHandler = lsp_utils::createRoutingDiagHandler(
      &context, pe.filePath, storedDiagnostics, filterOverlappingErrors);

  auto bindings = createParamBindings(
      pe.params, pe.paramsLoc.value_or(mlir::UnknownLoc::get(&context)));

  LLVM_DEBUG(llvm::dbgs() << "reanalyzePEProgram: peAbsPath=" << peAbsPath
                          << " overrideBuffers=" << overrideBuffers.size()
                          << "\n");

  auto eagerPredicate =
      openFiles_.empty()
          ? std::function<bool(llvm::StringRef)>()
          : [this](llvm::StringRef path) { return openFiles_.count(path); };
  parseAndAnalyze(&context, pe, newContents, flags, bindings, overrideBuffers,
                  systemModuleCache, eagerPredicate);

  // Rebuild file mapping since imports may have changed.
  buildFileMapping();
}

llvm::StringRef mlir::csl::CompilationContext::getLayoutPath() const {
  if (layout)
    return layout->filePath;
  return {};
}

llvm::StringRef mlir::csl::CompilationContext::getOwnerOfImportedModule(
    llvm::StringRef absPath) const {
  auto it = importedModuleToOwner.find(absPath);
  if (it != importedModuleToOwner.end())
    return it->second;
  return {};
}

llvm::DenseMap<llvm::StringRef, mlir::csl::ParamBinding>
mlir::csl::CompilationContext::createParamBindings(
    csl::StructValueAttr params, mlir::Location definitionLoc) {
  llvm::DenseMap<llvm::StringRef, csl::ParamBinding> bindings;

  if (params.isTuple())
    return bindings;

  for (size_t i = 0; i < params.getNumFields(); ++i) {
    auto nameResult = params.getFieldName(i);
    if (mlir::failed(nameResult))
      continue;
    llvm::StringRef name = *nameResult;
    mlir::TypedAttr value = params.getField(i);
    bindings.try_emplace(name, csl::ParamBinding{value, definitionLoc});
  }

  return bindings;
}
