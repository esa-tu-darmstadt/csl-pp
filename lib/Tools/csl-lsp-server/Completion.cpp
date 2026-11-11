//===- Completion.cpp - Code Completion for CSL LSP Server ----------------===//
//
// Implements textDocument/completion: identifier completion (visible symbols),
// dot completion (module members, struct fields, enum elements) and builtin
// completion (after @).
//
//===----------------------------------------------------------------------===//

#include "CompilationContext.h"
#include "LspUtils.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinTypes.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LSP/Protocol.h"

#define DEBUG_TYPE "csl-lsp-server"

namespace lsp = llvm::lsp;

using lsp::CompletionItem;
using lsp::CompletionItemKind;
using lsp::CompletionList;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Get the character at an offset relative to the cursor position.
static char getCharAt(const lsp::Position &pos, int charOffset,
                      llvm::StringRef source) {
  unsigned line = 0, offset = 0;
  for (; offset < source.size() && line < static_cast<unsigned>(pos.line);
       ++offset)
    if (source[offset] == '\n')
      ++line;
  int targetCol = pos.character + charOffset;
  if (targetCol < 0)
    return '\0';
  offset += targetCol;
  if (offset < source.size())
    return source[offset];
  return '\0';
}

/// Check if LSP position \p a comes before position \p b.
static bool isBefore(const lsp::Position &a, const lsp::Position &b) {
  return a.line < b.line || (a.line == b.line && a.character < b.character);
}

//===----------------------------------------------------------------------===//
// Completion Item Factories
//===----------------------------------------------------------------------===//

static CompletionItem makeGlobalItem(mlir::parsed_csl::GlobalOp globalOp) {
  CompletionItemKind kind = globalOp.getIsConst()
                                ? CompletionItemKind::Constant
                                : CompletionItemKind::Variable;
  if (globalOp.getInferredType() &&
      mlir::isa<mlir::csl::TypeType>(*(globalOp.getInferredType()))) {
    kind = CompletionItemKind::Keyword;
  }

  CompletionItem item(globalOp.getSymName(), kind);
  if (auto inferredType = globalOp.getInferredType()) {
    mlir::Type type = mlir::parsed_csl::unwrapLValueType(*inferredType);
    item.detail = mlir::csl::prettyPrintType(type);
  }
  return item;
}

static CompletionItem makeFuncItem(mlir::parsed_csl::FuncOp funcOp) {
  CompletionItem item(funcOp.getSymName(), CompletionItemKind::Function);
  item.detail = mlir::csl::prettyPrintType(funcOp.getFunctionType());
  return item;
}

static CompletionItem
makeGenericFuncItem(mlir::parsed_csl::GenericFuncOp gfOp) {
  CompletionItem item(gfOp.getSymName(), CompletionItemKind::Function);
  item.detail = "(generic)";
  return item;
}

static CompletionItem
makeVariableItem(mlir::parsed_csl::VariableOp varOp,
                 mlir::parsed_csl::ComptimeEvaluator *analysis) {
  CompletionItem item(varOp.getName(), varOp.getIsConst()
                                           ? CompletionItemKind::Constant
                                           : CompletionItemKind::Variable);
  if (analysis) {
    auto inferredType = analysis->getInferredType(varOp.getResult());
    if (mlir::succeeded(inferredType)) {
      mlir::Type type = mlir::parsed_csl::unwrapLValueType(*inferredType);
      item.detail = mlir::csl::prettyPrintType(type);
    }
  }
  return item;
}

static CompletionItem makeParamItem(llvm::StringRef name, mlir::Type type) {
  CompletionItem item(name, CompletionItemKind::Variable);
  if (type)
    item.detail = mlir::csl::prettyPrintType(type);
  return item;
}

//===----------------------------------------------------------------------===//
// Dot Completion Helpers
//===----------------------------------------------------------------------===//

/// Add completion items for members of a module.
static void addModuleCompletions(mlir::parsed_csl::ModuleOp moduleOp,
                                 CompletionList &result) {
  for (auto &op : moduleOp.getBody()->getOperations()) {
    if (auto globalOp = mlir::dyn_cast<mlir::parsed_csl::GlobalOp>(op))
      result.items.push_back(makeGlobalItem(globalOp));
    // else if (auto funcOp = mlir::dyn_cast<mlir::parsed_csl::FuncOp>(op))
    //   result.items.push_back(makeFuncItem(funcOp));
    else if (auto gfOp = mlir::dyn_cast<mlir::parsed_csl::GenericFuncOp>(op))
      result.items.push_back(makeGenericFuncItem(gfOp));
  }
}

/// Add completion items for fields of a struct type.
static void addStructCompletions(mlir::csl::StructType structType,
                                 CompletionList &result) {
  if (structType.isTuple())
    return;
  auto names = structType.getFieldNames();
  auto types = structType.getFieldTypes();
  for (size_t i = 0; i < names.size(); ++i) {
    CompletionItem item(names[i].getValue(), CompletionItemKind::Field);
    item.detail = mlir::csl::prettyPrintType(types[i]);
    result.items.push_back(std::move(item));
  }
}

/// Add completion items for elements of an enum type.
static void addEnumCompletions(mlir::csl::EnumType enumType,
                               CompletionList &result) {
  for (auto name : enumType.getElementNames()) {
    CompletionItem item(name.getValue(), CompletionItemKind::EnumMember);
    result.items.push_back(std::move(item));
  }
}

//===----------------------------------------------------------------------===//
// Dot Completion
//===----------------------------------------------------------------------===//

static void getDotCompletions(mlir::csl::DocumentView &view,
                              const lsp::Position &pos,
                              llvm::StringRef sourceText,
                              CompletionList &result) {
  LLVM_DEBUG(llvm::dbgs() << "getDotCompletions(" << pos.line << ":"
                          << pos.character << ")\n");
  if (!view.analysis) {
    LLVM_DEBUG(llvm::dbgs() << "  no analysis available\n");
    return;
  }

  // The cursor is after the dot. Find the base expression at the position
  // before the dot.
  lsp::Position basePos = {pos.line, std::max(0, pos.character - 2)};
  mlir::Operation *op = view.findOpAtPosition(basePos);
  if (!op) {
    LLVM_DEBUG(llvm::dbgs() << "  no op at base position\n");
    return;
  }

  // Unwrap LocOp (same pattern as formatHoverInfo).
  if (auto locOp = mlir::dyn_cast<mlir::parsed_csl::LocOp>(op)) {
    if (auto *innerOp = locOp.getValue().getDefiningOp())
      op = innerOp;
    else
      return;
  }

  LLVM_DEBUG(llvm::dbgs() << "  base op: " << op->getName() << " at "
                          << op->getLoc() << "\n");

  if (op->getNumResults() == 0)
    return;

  mlir::Value baseValue = op->getResult(0);

  // Try module-based lvalue first.
  if (auto lvalue = view.analysis->getComptimeLValue(baseValue);
      mlir::succeeded(lvalue)) {
    LLVM_DEBUG(llvm::dbgs() << "  lvalue: " << *lvalue << "\n");
    if (lvalue->isModuleBased()) {
      addModuleCompletions(lvalue->getBaseModule(), result);
      LLVM_DEBUG(llvm::dbgs()
                 << "  -> " << result.items.size() << " module completions\n");
      return;
    }
  }

  // Try inferred type for struct/enum fields.
  auto inferredType = view.analysis->getInferredType(baseValue);
  if (mlir::failed(inferredType)) {
    LLVM_DEBUG(llvm::dbgs() << "  inferred type failed\n");
    return;
  }

  mlir::Type type = mlir::parsed_csl::unwrapLValueType(*inferredType);
  LLVM_DEBUG(llvm::dbgs() << "  inferred type: " << type << "\n");

  if (auto structType = mlir::dyn_cast<mlir::csl::StructType>(type)) {
    addStructCompletions(structType, result);
  } else if (auto enumType = mlir::dyn_cast<mlir::csl::EnumType>(type)) {
    addEnumCompletions(enumType, result);
  }
  LLVM_DEBUG(llvm::dbgs() << "  -> " << result.items.size()
                          << " completions\n");
}

//===----------------------------------------------------------------------===//
// Builtin Completion
//===----------------------------------------------------------------------===//

static void getBuiltinCompletions(CompletionList &result) {
  unsigned maxVal = mlir::csl::getMaxEnumValForBuiltinKind();
  for (unsigned i = 0; i <= maxVal; ++i) {
    auto kind = mlir::csl::symbolizeBuiltinKind(i);
    if (!kind)
      continue;
    // Skip internal builtins.
    llvm::StringRef name = mlir::csl::stringifyBuiltinKind(*kind);
    if (name.starts_with("__"))
      continue;
    CompletionItem item("@" + name.str(), CompletionItemKind::Function);
    item.insertText = name.str();
    item.filterText = name.str();
    result.items.push_back(std::move(item));
  }
}

//===----------------------------------------------------------------------===//
// Identifier Completion
//===----------------------------------------------------------------------===//

/// Add function parameters to the completion list.
static void addFuncParams(mlir::parsed_csl::FuncOp funcOp,
                          CompletionList &result) {
  auto paramNames = funcOp.getParamNames();
  if (!paramNames)
    return;
  auto paramTypes = funcOp.getParamTypes();
  for (size_t i = 0; i < paramNames->size(); ++i) {
    auto name = mlir::cast<mlir::StringAttr>((*paramNames)[i]).getValue();
    mlir::Type type = (i < paramTypes.size()) ? paramTypes[i] : nullptr;
    result.items.push_back(makeParamItem(name, type));
  }
}

/// Add generic function parameters to the completion list (no types available).
static void addGenericFuncParams(mlir::parsed_csl::GenericFuncOp gfOp,
                                 CompletionList &result) {
  auto paramNames = gfOp.getParamNames();
  if (!paramNames)
    return;
  for (auto nameAttr : *paramNames) {
    auto name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    result.items.push_back(makeParamItem(name, nullptr));
  }
}

static void getIdentifierCompletions(mlir::csl::DocumentView &view,
                                     const lsp::Position &pos,
                                     CompletionList &result) {
  if (!view.compilationUnit)
    return;

  // Add builtin type names.
  llvm::SmallVector<llvm::StringRef> typeNames;
  mlir::csl::getAllBuiltinTypeNames(typeNames);
  for (auto name : typeNames) {
    CompletionItem item(name, CompletionItemKind::Keyword);
    item.detail = "(type)";
    result.items.push_back(std::move(item));
  }

  // Add language keywords.
  static constexpr llvm::StringLiteral keywords[] = {
      "align",    "and",      "anytype", "asm",      "break",  "comptime",
      "const",    "continue", "else",    "enum",     "export", "extern",
      "false",    "fn",       "for",     "if",       "inline", "layout",
      "linkname", "or",       "param",   "return",   "struct", "switch",
      "task",     "true",     "var",     "volatile", "while",
  };
  for (auto kw : keywords) {
    CompletionItem item(kw, CompletionItemKind::Keyword);
    result.items.push_back(std::move(item));
  }

  // Add top-level symbols from the compilation unit.
  for (auto &op : view.compilationUnit.getBody()->getOperations()) {
    // Skip ops from other files (imported modules).
    if (auto flcRange = mlir::dyn_cast<mlir::FileLineColRange>(op.getLoc()))
      if (flcRange.getFilename().getValue() != view.filePath)
        continue;

    if (auto globalOp = mlir::dyn_cast<mlir::parsed_csl::GlobalOp>(op))
      result.items.push_back(makeGlobalItem(globalOp));
    // else if (auto funcOp = mlir::dyn_cast<mlir::parsed_csl::FuncOp>(op))
    //   result.items.push_back(makeFuncItem(funcOp));
    else if (auto gfOp = mlir::dyn_cast<mlir::parsed_csl::GenericFuncOp>(op))
      result.items.push_back(makeGenericFuncItem(gfOp));
  }

  // Find the enclosing function for local completions.
  mlir::Operation *opAtPos = view.findOpAtPosition(pos);
  if (!opAtPos)
    return;

  mlir::Operation *funcOp = opAtPos;
  while (funcOp &&
         !mlir::isa<mlir::parsed_csl::FuncOp, mlir::parsed_csl::GenericFuncOp>(
             funcOp))
    funcOp = funcOp->getParentOp();
  if (!funcOp)
    return;

  // Add function parameters.
  if (auto func = mlir::dyn_cast<mlir::parsed_csl::FuncOp>(funcOp))
    addFuncParams(func, result);
  else if (auto gf = mlir::dyn_cast<mlir::parsed_csl::GenericFuncOp>(funcOp))
    addGenericFuncParams(gf, result);

  // Add local variables declared before the cursor.
  funcOp->walk([&](mlir::parsed_csl::VariableOp varOp) {
    auto range = mlir::csl::lsp_utils::getRange(varOp->getLoc());
    if (range && isBefore(range->start, pos))
      result.items.push_back(makeVariableItem(varOp, view.analysis));
  });
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

lsp::CompletionList
mlir::csl::DocumentView::getCompletions(const lsp::Position &pos,
                                        const lsp::CompletionContext &ctx,
                                        llvm::StringRef sourceText) {
  CompletionList result;

  LLVM_DEBUG(llvm::dbgs() << "getCompletions(" << pos.line << ":"
                          << pos.character << ") trigger='"
                          << ctx.triggerCharacter << "'\n");

  if (ctx.triggerCharacter == ".") {
    getDotCompletions(*this, pos, sourceText, result);
  } else if (ctx.triggerCharacter == "@") {
    getBuiltinCompletions(result);
  } else {
    // Manual invocation or regular typing: check char before cursor.
    char ch = getCharAt(pos, -1, sourceText);
    if (ch == '.')
      getDotCompletions(*this, pos, sourceText, result);
    else if (ch == '@')
      getBuiltinCompletions(result);
    else
      getIdentifierCompletions(*this, pos, result);
  }

  LLVM_DEBUG(llvm::dbgs() << "  returning " << result.items.size()
                          << " completion(s)\n");
  return result;
}
