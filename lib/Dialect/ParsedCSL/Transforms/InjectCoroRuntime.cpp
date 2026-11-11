//===- InjectCoroRuntime.cpp - Import the coroutine runtime ---------------===//
//
// Adds the coroutine runtime module imports to PE programs that need them.
//
// The runtime is part of the compiler, not a library the user depends on: the
// lowering passes reach into it by symbol name (`__fiber_runtime__*`,
// `__alloc_runtime__*`). Before this pass existed, every program that used
// coroutines had to import it by hand, by absolute path into a checkout of
// this repository -- which broke the moment the sources moved.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CoroRuntimeSources.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h"

#include "mlir/IR/Builders.h"

namespace mlir::parsed_csl {
#define GEN_PASS_DEF_INJECTCORORUNTIMEPASS
#include "csl-mlir/Dialect/ParsedCSL/Transforms/Passes.h.inc"
} // namespace mlir::parsed_csl

using namespace mlir;
using namespace mlir::parsed_csl;

namespace {

/// Emit `const <moduleName> = @import_module("<importPath>");` at the end of
/// the compilation unit. Parameters are left at their defaults; the runtime's
/// only parameter is a debug switch, which belongs to the compiler rather than
/// to the program being compiled.
void injectImport(csl::CompilationUnitOp unit, OpBuilder &builder,
                  StringRef moduleName, StringRef importPath) {
  // An existing declaration means a program written against the old
  // hand-imported convention. Leave it alone rather than colliding with it.
  if (unit.lookupSymbol(moduleName))
    return;

  Location loc = unit.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(unit.getBody());

  auto globalOp = GlobalOp::create(
      builder, loc, moduleName, /*dynamicType=*/Value(), /*is_const=*/true,
      /*is_export=*/false, /*is_extern=*/false, /*is_unbound_param=*/false,
      /*linkname=*/nullptr, /*align=*/nullptr, /*inferred_type=*/nullptr,
      /*inferred_initializer=*/nullptr);

  Block &initBlock = globalOp.getInitRegion().emplaceBlock();
  builder.setInsertionPointToStart(&initBlock);

  // The angle brackets mark this as a system import; the resolver recognises
  // the path and serves the source from the compiler binary.
  Value path = ConstantOp::create(
      builder, loc,
      cast<TypedAttr>(csl::ComptimeStringAttr::get(
          builder.getContext(), ("<" + importPath + ">").str())));
  Value module =
      BuiltinCallOp::create(
          builder, loc,
          csl::stringifyBuiltinKind(csl::BuiltinKind::ImportModule),
          ValueRange{path}, /*inferred_type=*/mlir::TypeAttr())
          .getResult();
  YieldOp::create(builder, loc, module);
}

struct InjectCoroRuntimePass
    : public parsed_csl::impl::InjectCoroRuntimePassBase<
          InjectCoroRuntimePass> {
  using InjectCoroRuntimePassBase::InjectCoroRuntimePassBase;

  void runOnOperation() override {
    csl::CompilationUnitOp unit = getOperation();
    OpBuilder builder(unit.getContext());

    // Both modules are imported unconditionally.
    //
    // Whether a program needs them cannot be decided here. This pass has to
    // run before comptime evaluation, because that is what resolves imports
    // and specializes the runtime -- but it is also comptime evaluation that
    // pulls in the imported modules, so at this point the only thing visible
    // is the entry file. The constructs that need the runtime routinely sit
    // in an imported library instead (`benchmark-libs/stencil_3d_7pts` awaits
    // in parallel, for one), and gating on the entry file's syntax silently
    // misses them.
    //
    // Importing unconditionally is safe in the direction that matters: an
    // unused module costs compile time, a missing one is a hard error late in
    // lowering. The runtime's symbols are marked export so they survive DCE
    // until the passes that reference them have run; StripRuntimeExportsPass
    // then drops those markers and stage 3's SymbolDCE removes whatever went
    // unreferenced, so nothing unused reaches the output.
    injectImport(unit, builder, csl::kFiberRuntimeModuleName,
                 csl::kFiberRuntimeImportPath);
    // The fiber runtime itself calls @release_local_task, so it depends on
    // the allocator; the two are never imported separately.
    injectImport(unit, builder, csl::kAllocRuntimeModuleName,
                 csl::kAllocRuntimeImportPath);
  }
};

} // namespace
