//===- IndirectCallTargets.cpp - Possible callees of indirect calls -------===//

#include "csl-mlir/Dialect/TypedCSL/Analysis/IndirectCallTargets.h"

#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::typed_csl;

/// Whether `constant` is a function reference that is only passed to
/// builtins that do not give the program a pointer to the function.
static bool isOnlyBoundOrExported(ConstantOp constant) {
  if (!mlir::isa<csl::FuncRefAttr>(constant.getValue()) ||
      constant->use_empty())
    return false;
  return llvm::all_of(constant->getUsers(), [](Operation *user) {
    auto builtinCall = mlir::dyn_cast<BuiltinCallOp>(user);
    if (!builtinCall)
      return false;
    switch (builtinCall.getBuiltin()) {
    case csl::BuiltinKind::BindLocalTask:
    case csl::BuiltinKind::BindDataTask:
    case csl::BuiltinKind::BindControlTask:
    case csl::BuiltinKind::ExportSymbol:
      return true;
    default:
      return false;
    }
  });
}

/// The function referenced by a constant callee of an indirect call, or null
/// if the callee is not a constant.
static csl::FuncRefAttr getConstantCallee(IndirectCallOp call) {
  auto constant = call.getCallee().getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  if (auto funcAddr = mlir::dyn_cast<csl::FuncAddrAttr>(constant.getValue()))
    return funcAddr.getFuncRef();
  return mlir::dyn_cast<csl::FuncRefAttr>(constant.getValue());
}

IndirectCallTargets::IndirectCallTargets(Operation *op) {
  SymbolTableCollection symbolTable;
  auto lookup = [&](Operation *from, csl::FuncRefAttr ref) -> FuncOp {
    return symbolTable.lookupNearestSymbolFrom<FuncOp>(from,
                                                       ref.getSymbolName());
  };

  op->walk([&](Operation *nested) {
    if (auto constant = mlir::dyn_cast<ConstantOp>(nested);
        constant && isOnlyBoundOrExported(constant))
      return;
    for (NamedAttribute attr : nested->getAttrs()) {
      attr.getValue().walk([&](csl::FuncRefAttr ref) {
        if (FuncOp func = lookup(nested, ref))
          addressTaken.insert(func);
      });
    }
  });

  op->walk([&](FuncOp func) {
    if (addressTaken.contains(func))
      addressTakenByType[func.getFunctionType()].push_back(func);
  });

  op->walk([&](IndirectCallOp call) {
    SmallVector<FuncOp> &targets = callees[call];
    if (csl::FuncRefAttr ref = getConstantCallee(call)) {
      if (FuncOp func = lookup(call, ref))
        targets.push_back(func);
      return;
    }
    auto it = addressTakenByType.find(call.getCalleeType());
    if (it != addressTakenByType.end())
      targets = it->second;
  });
}

ArrayRef<FuncOp>
IndirectCallTargets::getPossibleCallees(IndirectCallOp call) const {
  auto it = callees.find(call);
  if (it == callees.end())
    return {};
  return it->second;
}

bool IndirectCallTargets::isAddressTaken(FuncOp func) const {
  return addressTaken.contains(func);
}
