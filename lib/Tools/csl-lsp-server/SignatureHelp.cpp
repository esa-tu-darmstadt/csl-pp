//===- SignatureHelp.cpp - Signature Help for CSL LSP Server
//---------------===//
//
// Implements textDocument/signatureHelp: shows function parameter names/types
// and highlights the active parameter based on cursor position within a call.
//
//===----------------------------------------------------------------------===//

#include "CompilationContext.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvaluator.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LSP/Protocol.h"

#define DEBUG_TYPE "csl-lsp-server"

namespace lsp = llvm::lsp;

using lsp::ParameterInformation;
using lsp::SignatureHelp;
using lsp::SignatureInformation;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Convert an LSP position (line, character) to a byte offset in the source.
/// Returns std::nullopt if the position is out of range.
static std::optional<unsigned> positionToOffset(llvm::StringRef source,
                                                const lsp::Position &pos) {
  unsigned line = 0, offset = 0;
  for (; offset < source.size() && line < static_cast<unsigned>(pos.line);
       ++offset)
    if (source[offset] == '\n')
      ++line;
  if (line != static_cast<unsigned>(pos.line))
    return std::nullopt;
  unsigned col = static_cast<unsigned>(pos.character);
  if (offset + col > source.size())
    return std::nullopt;
  return offset + col;
}

/// Scan backwards from the cursor to find the opening '(' of the enclosing
/// call. Returns the offset of '(' and sets `commaCount` to the number of
/// commas at nesting depth 0 (i.e. between the call's '(' and cursor).
/// Returns std::nullopt if no enclosing '(' is found.
static std::optional<unsigned> findEnclosingParen(llvm::StringRef source,
                                                  unsigned cursorOffset,
                                                  int &commaCount) {
  commaCount = 0;
  int depth = 0;
  bool inString = false;

  for (unsigned i = cursorOffset; i > 0; --i) {
    char c = source[i - 1];

    // Track string literals (simplified — no escape handling needed since we
    // scan backwards and CSL strings don't span lines).
    if (c == '"') {
      inString = !inString;
      continue;
    }
    if (inString)
      continue;

    if (c == ')' || c == ']' || c == '}') {
      ++depth;
    } else if (c == '(' || c == '[' || c == '{') {
      if (depth == 0) {
        if (c == '(')
          return i - 1;
        // Hit an unmatched '[' or '{' — not inside a call.
        return std::nullopt;
      }
      --depth;
    } else if (c == ',' && depth == 0) {
      ++commaCount;
    }
  }
  return std::nullopt;
}

/// Build a SignatureInformation from a function name, its FunctionType, and
/// optional parameter names.
static SignatureInformation
buildSignatureFromFunc(llvm::StringRef funcName, mlir::FunctionType funcType,
                       std::optional<mlir::ArrayAttr> paramNames) {
  SignatureInformation sig;

  std::string label = funcName.str() + "(";
  std::vector<ParameterInformation> params;

  unsigned numInputs = funcType.getNumInputs();
  for (unsigned i = 0; i < numInputs; ++i) {
    if (i > 0)
      label += ", ";

    ParameterInformation param;
    unsigned paramStart = label.size();

    // Parameter name.
    if (paramNames && i < paramNames->size()) {
      llvm::StringRef name =
          mlir::cast<mlir::StringAttr>((*paramNames)[i]).getValue();
      label += name.str();
      label += ": ";
    }

    // Parameter type.
    label += mlir::csl::prettyPrintType(funcType.getInput(i));

    unsigned paramEnd = label.size();
    param.labelOffsets = std::make_pair(paramStart, paramEnd);
    params.push_back(std::move(param));
  }

  label += ")";

  // Return type.
  if (funcType.getNumResults() > 0) {
    mlir::Type retType = funcType.getResult(0);
    if (!mlir::isa<mlir::csl::VoidType>(retType))
      label += " " + mlir::csl::prettyPrintType(retType);
  }

  sig.label = std::move(label);
  sig.parameters = std::move(params);
  return sig;
}

/// Build a SignatureInformation for a GenericFuncOp that hasn't been
/// instantiated (no concrete FunctionType available). Shows comptime/anytype
/// annotations.
static SignatureInformation
buildSignatureFromGenericFunc(mlir::parsed_csl::GenericFuncOp genericFunc) {
  SignatureInformation sig;

  auto comptimeParams = genericFunc.getComptimeParams();
  auto anytypeParams = genericFunc.getAnytypeParams();
  auto paramNamesAttr = genericFunc.getParamNames();
  int32_t numParams = genericFunc.getNumParams();

  llvm::SmallDenseSet<int32_t> comptimeSet(comptimeParams.begin(),
                                           comptimeParams.end());
  llvm::SmallDenseSet<int32_t> anytypeSet(anytypeParams.begin(),
                                          anytypeParams.end());

  std::string label = genericFunc.getSymName().str() + "(";
  std::vector<ParameterInformation> params;

  for (int32_t i = 0; i < numParams; ++i) {
    if (i > 0)
      label += ", ";

    ParameterInformation param;
    unsigned paramStart = label.size();

    if (comptimeSet.contains(i))
      label += "comptime ";

    if (paramNamesAttr && static_cast<size_t>(i) < paramNamesAttr->size())
      label +=
          mlir::cast<mlir::StringAttr>((*paramNamesAttr)[i]).getValue().str();
    else
      label += "_";

    if (anytypeSet.contains(i))
      label += ": anytype";

    unsigned paramEnd = label.size();
    param.labelOffsets = std::make_pair(paramStart, paramEnd);
    params.push_back(std::move(param));
  }

  label += ")";

  sig.label = std::move(label);
  sig.parameters = std::move(params);
  return sig;
}

/// Build a SignatureInformation for a resolved GenericFuncOp with a concrete
/// FunctionType. Skips comptime parameters from both the label and params list,
/// since they don't appear at the call site after instantiation.
static SignatureInformation buildSignatureFromResolvedGenericFunc(
    mlir::parsed_csl::GenericFuncOp genericFunc, mlir::FunctionType funcType) {
  SignatureInformation sig;

  auto comptimeParams = genericFunc.getComptimeParams();
  auto paramNamesAttr = genericFunc.getParamNames();
  int32_t numParams = genericFunc.getNumParams();

  llvm::SmallDenseSet<int32_t> comptimeSet(comptimeParams.begin(),
                                           comptimeParams.end());

  // The FunctionType only has runtime parameters (comptime params are removed).
  // Build a mapping from runtime param index to GenericFuncOp param index.
  std::string label = genericFunc.getSymName().str() + "(";
  std::vector<ParameterInformation> params;

  unsigned runtimeIdx = 0;
  for (int32_t i = 0; i < numParams; ++i) {
    if (comptimeSet.contains(i))
      continue;

    if (runtimeIdx > 0)
      label += ", ";

    ParameterInformation param;
    unsigned paramStart = label.size();

    // Parameter name from the GenericFuncOp.
    if (paramNamesAttr && static_cast<size_t>(i) < paramNamesAttr->size()) {
      llvm::StringRef name =
          mlir::cast<mlir::StringAttr>((*paramNamesAttr)[i]).getValue();
      label += name.str();
      label += ": ";
    }

    // Parameter type from the concrete FunctionType.
    if (runtimeIdx < funcType.getNumInputs())
      label += mlir::csl::prettyPrintType(funcType.getInput(runtimeIdx));

    unsigned paramEnd = label.size();
    param.labelOffsets = std::make_pair(paramStart, paramEnd);
    params.push_back(std::move(param));
    ++runtimeIdx;
  }

  label += ")";

  // Return type.
  if (funcType.getNumResults() > 0) {
    mlir::Type retType = funcType.getResult(0);
    if (!mlir::isa<mlir::csl::VoidType>(retType))
      label += " " + mlir::csl::prettyPrintType(retType);
  }

  sig.label = std::move(label);
  sig.parameters = std::move(params);
  return sig;
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

SignatureHelp
mlir::csl::DocumentView::getSignatureHelp(const lsp::Position &pos,
                                          llvm::StringRef sourceText) {
  SignatureHelp result;

  LLVM_DEBUG(llvm::dbgs() << "getSignatureHelp(" << pos.line << ":"
                          << pos.character << ")\n");

  if (sourceText.empty() || !analysis || !parsedSuccessfully) {
    LLVM_DEBUG(llvm::dbgs() << "  early exit: empty=" << sourceText.empty()
                            << " analysis=" << (analysis != nullptr)
                            << " parsed=" << parsedSuccessfully << "\n");
    return result;
  }

  // 1. Convert cursor position to byte offset.
  auto cursorOffset = positionToOffset(sourceText, pos);
  if (!cursorOffset)
    return result;

  // 2. Scan backwards to find the enclosing '(' and count commas.
  int commaCount = 0;
  auto parenOffset = findEnclosingParen(sourceText, *cursorOffset, commaCount);
  if (!parenOffset) {
    LLVM_DEBUG(llvm::dbgs() << "  no enclosing paren found\n");
    return result;
  }
  LLVM_DEBUG(llvm::dbgs() << "  paren at offset " << *parenOffset
                          << ", commaCount=" << commaCount << "\n");

  // 3. Find the innermost CallOp whose location contains the cursor.
  //    CallOps are not parents of their argument ops (they're siblings in the
  //    same block), so we search by location range rather than walking up.
  auto callOp = findOpAtPosition<mlir::parsed_csl::CallOp>(pos);
  if (!callOp) {
    LLVM_DEBUG(llvm::dbgs() << "  no enclosing CallOp found\n");
    return result;
  }
  LLVM_DEBUG(llvm::dbgs() << "  found CallOp at " << callOp->getLoc() << "\n");

  // 4. Resolve callee to a function.
  auto rvalue = analysis->getOrLoadComptimeRValue(callOp.getCallee());
  if (mlir::failed(rvalue)) {
    LLVM_DEBUG(llvm::dbgs() << "  callee rvalue resolution failed\n");
    return result;
  }

  auto funcAttr = mlir::dyn_cast<csl::FuncRefAttr>(*rvalue);
  if (!funcAttr) {
    LLVM_DEBUG(llvm::dbgs()
               << "  callee is not a FuncRefAttr: " << *rvalue << "\n");
    return result;
  }

  auto funcResult = parsed_csl::lookupFuncByRefAttr(funcAttr, compilationUnit);
  if (mlir::failed(funcResult)) {
    LLVM_DEBUG(llvm::dbgs() << "  func lookup failed for " << funcAttr << "\n");
    return result;
  }
  LLVM_DEBUG(llvm::dbgs() << "  resolved to " << funcAttr << "\n");

  // 5. Build SignatureInformation based on function kind.
  SignatureInformation sig;

  if (std::holds_alternative<parsed_csl::FuncOp>(*funcResult)) {
    auto funcOp = std::get<parsed_csl::FuncOp>(*funcResult);
    sig = buildSignatureFromFunc(funcOp.getOriginalName(),
                                 funcOp.getFunctionType(),
                                 funcOp.getParamNames());
    if (auto doc = funcOp.getDocString())
      sig.documentation = doc->str();
  } else if (std::holds_alternative<parsed_csl::GenericFuncOp>(*funcResult)) {
    auto genericFunc = std::get<parsed_csl::GenericFuncOp>(*funcResult);

    // If the FuncRefAttr carries a concrete FunctionType, use it.
    if (auto funcType = mlir::dyn_cast<mlir::FunctionType>(funcAttr.getType()))
      sig = buildSignatureFromResolvedGenericFunc(genericFunc, funcType);
    else
      sig = buildSignatureFromGenericFunc(genericFunc);
    if (auto doc = genericFunc.getDocString())
      sig.documentation = doc->str();
  }

  if (sig.label.empty())
    return result;

  result.activeParameter = commaCount;
  result.signatures.push_back(std::move(sig));
  return result;
}
