//===- TypedCSLInterfaces.td - TypedCSL Op Interfaces ----*- tablegen -*-===//
//
// Op interface definitions for the TypedCSL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TYPED_CSL_INTERFACES
#define TYPED_CSL_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BreakingTerminatorOpInterface
//===----------------------------------------------------------------------===//

def TypedCSL_BreakingTerminatorOpInterface
    : OpInterface<"BreakingTerminatorOpInterface"> {
  let cppNamespace = "::mlir::typed_csl";

  let description = [{
    Interface for terminators that exit a fixed number of enclosing regions
    (break, continue, return, unreachable). The number of breaking regions N
    addresses the op that receives control:

    - N = 1 exits the immediately enclosing region and transfers control to
      the parent op.
    - N = K exits K regions and transfers control to the K-th ancestor, which
      must implement `HasBreakingControlFlowOpInterface` and accept the
      terminator. The K-1 ops in between must carry
      `PropagateControlFlowBreak`.

    Implementing ops must also declare `RegionExitTerminatorOpInterface`; this
    interface provides its `getPotentialTargets` method.
  }];

  let methods = [
    InterfaceMethod<
      "Return the number of enclosing regions this terminator exits.",
      "int32_t", "getNumBreakingRegions", (ins)
    >,
    InterfaceMethod<
      "Set the number of enclosing regions this terminator exits.",
      "void", "setNumBreakingRegions", (ins "int32_t":$numBreakingRegions)
    >
  ];

  let extraSharedClassDeclaration = [{
    /// Return the op that receives control from this terminator, or nullptr
    /// if the terminator is nested in fewer ops than it exits.
    ::mlir::Operation *getTargetOp() {
      int32_t numBreakingRegions = $_op.getNumBreakingRegions();
      if (numBreakingRegions < 1)
        return nullptr;
      ::mlir::Operation *target = $_op.getOperation();
      for (int32_t i = 0; i < numBreakingRegions && target; ++i)
        target = target->getParentOp();
      return target;
    }

    /// RegionExitTerminatorOpInterface: the addressed op is the only
    /// potential target.
    ::llvm::SmallVector<::mlir::Operation *> getPotentialTargets() {
      if (::mlir::Operation *target = getTargetOp())
        return {target};
      return {};
    }
  }];

  let verify = [{
    return ::mlir::typed_csl::detail::verifyBreakingTerminator($_op);
  }];
}

#endif // TYPED_CSL_INTERFACES
