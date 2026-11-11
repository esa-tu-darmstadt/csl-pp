//===- ComptimeInterfaces.cpp - Comptime Evaluation Interfaces -----------===//
//
// Implementation of comptime evaluation interfaces.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.h"
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeEvalContext.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"

using namespace mlir;
using namespace mlir::parsed_csl;

//===----------------------------------------------------------------------===//
// ComptimeError Constructor
//===----------------------------------------------------------------------===//

ComptimeError::ComptimeError(ErrorKind kind, Operation *op,
                             ComptimeEvalContext &context)
    : kind_(kind), failedOp_(op), inferredType_(Type()) {
  // Create an error diagnostic for this operation
  InFlightDiagnostic diag = op->emitError();

  // Steal the diagnostic and store it in a shared_ptr
  Diagnostic &underlyingDiag = *diag.getUnderlyingDiagnostic();
  reason_ = std::make_shared<Diagnostic>(std::move(underlyingDiag));
  diag.abandon();
}

//===----------------------------------------------------------------------===//
// LValueDescriptor::resolveModuleIfNeeded Implementation
//===----------------------------------------------------------------------===//

LValueDescriptor
LValueDescriptor::resolveModuleIfNeeded(ComptimeEvalContext &context) const {
  if (!isGlobalBased())
    return *this;

  // Accesses into modules should be resolved during extension
  assert(accessPath_.empty());

  // Check if the value type is a module type
  if (!mlir::isa<csl::ImportedModuleType>(valueType_))
    return *this;

  // Get the module reference value from the global
  GlobalOp globalOp = getBaseGlobal();
  auto initValue = globalOp.getInferredInitializer();
  if (!initValue || !mlir::isa<ModuleRefAttr>(*initValue))
    return *this;

  // Find the ModuleOp with the matching instance ID.
  auto moduleRefAttr = mlir::cast<ModuleRefAttr>(*initValue);
  csl::CompilationUnitOp compUnit =
      globalOp->getParentOfType<csl::CompilationUnitOp>();

  ModuleOp moduleOp =
      findModuleById(compUnit, moduleRefAttr.getModuleIdentifier());
  if (!moduleOp)
    return *this; // Module not found, return unchanged

  // Return a module-based lvalue descriptor
  return LValueDescriptor(moduleOp);
}

//===----------------------------------------------------------------------===//
// LValueDescriptor::extendWithField Implementation
//===----------------------------------------------------------------------===//

LValueDescriptor
LValueDescriptor::extendWithField(StringRef fieldName, Type newType,
                                  ComptimeEvalContext &context) const {
  // LValueDescriptors are always resolved (module refs unpacked at creation)
  // so we can directly check if this is module-based.
  if (isModuleBased()) {
    assert(accessPath_.empty() &&
           "Module-based lvalue with non-empty access path");
    ModuleOp moduleOp = getBaseModule();

    FailureOr<GlobalOp> memberGlobalOp = moduleOp.getMemberVariable(fieldName);
    assert(succeeded(memberGlobalOp) &&
           "Module field not found during lvalue extension");

    // Get the member's value type
    GlobalOp memberOp = *memberGlobalOp;
    FailureOr<Type> memberType = context.getGlobalValueType(memberOp);
    assert(succeeded(memberType) && "Failed to get global value type");

    // Create LValueDescriptor for the member and resolve if it's a module
    LValueDescriptor memberLValue(memberOp, {}, *memberType);
    return memberLValue.resolveModuleIfNeeded(context);
  }

  // Normal case: extend the access path
  SmallVector<AccessPathElement> newPath(accessPath_);
  newPath.push_back(FieldAccess(fieldName));
  return LValueDescriptor(base_, std::move(newPath), newType);
}

//===----------------------------------------------------------------------===//
// LValueDescriptor::toGlobalRefAttr Implementation
//===----------------------------------------------------------------------===//

std::optional<csl::GlobalRefAttr> LValueDescriptor::toGlobalRefAttr() const {
  // Can only convert direct global (no access path)
  if (!isDirectReference() || !isGlobalBased())
    return std::nullopt;

  GlobalOp globalOp = getBaseGlobal();

  // Determine module ID if the global is inside a module
  std::optional<uint64_t> moduleId;
  if (auto parentModule = globalOp->getParentOfType<parsed_csl::ModuleOp>()) {
    moduleId = static_cast<uint64_t>(parentModule.getModuleIdentifier());
  }

  // Create LValueType from the value type and global's const property
  // valueType_ is the value type stored in the global, not wrapped in
  // LValueType
  csl::LValueType lvalueType =
      csl::LValueType::get(valueType_, globalOp.getIsConst());

  return csl::GlobalRefAttr::get(
      globalOp->getContext(), moduleId,
      FlatSymbolRefAttr::get(globalOp.getSymNameAttr()), lvalueType);
}

std::optional<ModuleRefAttr> LValueDescriptor::toModuleRefAttr() const {
  // Can only convert direct module references (no access path)
  if (!isDirectReference() || !isModuleBased())
    return std::nullopt;

  ModuleOp moduleOp = getBaseModule();

  return ModuleRefAttr::get(moduleOp.getContext(),
                            moduleOp.getModuleIdentifier());
}

std::optional<TypedAttr> LValueDescriptor::toAttr() const {
  // Try to convert to GlobalRefAttr
  if (std::optional<csl::GlobalRefAttr> globalRef = toGlobalRefAttr()) {
    return *globalRef;
  }

  // Try to convert to ModuleRefAttr
  if (std::optional<ModuleRefAttr> moduleRef = toModuleRefAttr()) {
    return *moduleRef;
  }

  return std::nullopt;
}

/// Include the generated interface method definitions.
#include "csl-mlir/Dialect/ParsedCSL/Analysis/ComptimeEval/ComptimeInterfaces.cpp.inc"
