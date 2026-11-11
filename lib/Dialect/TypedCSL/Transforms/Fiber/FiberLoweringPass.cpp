//===- FiberLoweringPass.cpp - Fiber-based cooperative multitasking
//--------===//
//
// Lowers async functions using fiber-based stack switching. Uses a runtime
// library (wse3.csl) for set_sp_and_jmp, suspend, and complete primitives.
// At suspend points, the calling convention handles register preservation.
// At fresh start, args and completion handler address are pre-written to the
// fiber stack's arg area and loaded by set_sp_and_jmp into r2-r5/r15.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/Utils/CommandLineOptions.h"
#include "csl-mlir/Dialect/TypedCSL/Analysis/IndirectCallTargets.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizeSource.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberSizingContext.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/FiberValidation.h"
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include <atomic>
#include <optional>

#define DEBUG_TYPE "fiber-lowering"

namespace mlir::typed_csl {
#define GEN_PASS_DEF_FIBERLOWERINGPASS
#include "csl-mlir/Dialect/TypedCSL/Transforms/Passes.h.inc"
} // namespace mlir::typed_csl

using namespace mlir;
using namespace mlir::typed_csl;

namespace {

//===----------------------------------------------------------------------===//
// Constants — the shared layout values (also queried by the resolver) live
// in FiberSizingContext.h.  See `lowerSingleAwaitParallel` for the chain
// math each constant participates in.
//===----------------------------------------------------------------------===//

using mlir::typed_csl::kArgAreaU16Slots;

// Local short alias to match prior code's name.
static constexpr unsigned kArgAreaSlots = kArgAreaU16Slots;

/// Assigns the arguments of a worker callee to its four 16-bit argument
/// registers r2-r5, which set_sp_and_jmp loads from arg-area slots 0-3, the
/// way cslc does: a 32-bit argument takes the first free even-aligned pair
/// (r2:r3 or r4:r5), a 16-bit argument the first free register, including
/// one that the alignment of a 32-bit argument skipped.
struct ArgRegisterAllocator {
  bool used[4] = {false, false, false, false};

  /// The arg-area slot of the first register for an argument of
  /// `sizeBytes`, or std::nullopt if it does not fit.
  std::optional<unsigned> allocate(uint32_t sizeBytes) {
    if (sizeBytes <= 2) {
      for (unsigned slot = 0; slot < 4; ++slot) {
        if (!used[slot]) {
          used[slot] = true;
          return slot;
        }
      }
      return std::nullopt;
    }
    if (sizeBytes == 4) {
      for (unsigned slot = 0; slot < 4; slot += 2) {
        if (!used[slot] && !used[slot + 1]) {
          used[slot] = used[slot + 1] = true;
          return slot;
        }
      }
    }
    return std::nullopt;
  }

  /// Whether arguments of `types` fit into the registers.
  static bool fits(TypeRange types) {
    ArgRegisterAllocator allocator;
    for (Type type : types) {
      std::optional<uint32_t> size = csl::getSizeInBytes(type);
      if (!size || !allocator.allocate(*size))
        return false;
    }
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Fiber Pool Info (computed during infrastructure generation)
//===----------------------------------------------------------------------===//

struct FiberPoolInfo {
  /// One per-field entry; the runtime stores the pool struct-of-arrays so
  /// each field has its own top-level `__fiber_runtime__pool_<field>` global.
  struct Field {
    csl::GlobalRefAttr globalRef; ///< `__fiber_runtime__pool_<field>` global
    csl::ArrayType arrayTy;       ///< `[pool_size]<elementTy>`
    Type elementTy;               ///< element type, same as the old field type
  };
  llvm::StringMap<Field> fields;
  uint32_t poolSize;

  /// Convenience accessors mirroring the old API (used at many call sites).
  /// Both SDK 2.10 runtimes store hardware word addresses as u16.
  Type spFieldTy() const { return fields.lookup("sp").elementTy; }
  Type pcFieldTy() const { return fields.lookup("pc").elementTy; }
  Type joinCounterPtrFieldTy() const {
    return fields.lookup("join_counter_ptr").elementTy;
  } ///< *u16
};

//===----------------------------------------------------------------------===//
// Inline Asm Helpers
//===----------------------------------------------------------------------===//

/// Emit volatile asm to read the hardware SP as a u16 word address.
static Value emitReadSP(OpBuilder &b, Location loc,
                        csl::TargetArch targetArch) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto scratchLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
  static std::atomic<unsigned> spUidCounter{0};
  std::string name = "__sp_read_" + std::to_string(spUidCounter++);
  Value scratch =
      VariableOp::create(b, loc, scratchLvalTy, name, Value()).getResult();

  StringRef asmStr = targetArch == csl::TargetArch::WSE3
                         ? "fmov16.nf %[out] = sp"
                         : "mov16 %[out] = r7";

  InlineAsmOp::create(b, loc,
                      /*outputs=*/ValueRange{scratch},
                      /*inputs=*/ValueRange{}, b.getStringAttr(asmStr),
                      /*output_names=*/b.getStrArrayAttr({"out"}),
                      /*output_constraints=*/b.getStrArrayAttr({"=r"}),
                      /*input_names=*/b.getStrArrayAttr({}),
                      /*input_constraints=*/b.getStrArrayAttr({}),
                      /*clobbers=*/b.getStrArrayAttr({}),
                      /*is_volatile=*/UnitAttr::get(ctx));

  return LoadOp::create(b, loc, scratch);
}

/// Emit non-volatile asm to bitcast a typed value to u16 (raw address
/// extraction). Used to extract raw addresses from typed pointers for
/// writing to save-area slots.
static std::atomic<unsigned> bitcastUidCounter{0};
static Value emitBitcastToU16(OpBuilder &b, Location loc, Value typedVal,
                              csl::TargetArch targetArch) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto scratchLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
  std::string name = "__asm_scratch_" + std::to_string(bitcastUidCounter++);
  Value scratch =
      VariableOp::create(b, loc, scratchLvalTy, name, Value()).getResult();

  StringRef asmStr = targetArch == csl::TargetArch::WSE3
                         ? "fmov16.nf %[out] = %[inp]"
                         : "mov16 %[out] = %[inp]";

  InlineAsmOp::create(b, loc,
                      /*outputs=*/ValueRange{scratch},
                      /*inputs=*/ValueRange{typedVal}, b.getStringAttr(asmStr),
                      /*output_names=*/b.getStrArrayAttr({"out"}),
                      /*output_constraints=*/b.getStrArrayAttr({"=r"}),
                      /*input_names=*/b.getStrArrayAttr({"inp"}),
                      /*input_constraints=*/b.getStrArrayAttr({"r"}),
                      /*clobbers=*/b.getStrArrayAttr({}),
                      /*is_volatile=*/UnitAttr()); // NOT volatile

  return LoadOp::create(b, loc, scratch);
}

/// Convert an SDK 2.10 CSL byte address to the hardware u16-word address used
/// by WSE-2/WSE-3 SP and indirect jump targets.
static Value emitHardwareWordAddress(OpBuilder &b, Location loc,
                                     Value byteAddress,
                                     csl::TargetArch targetArch) {
  if (targetArch != csl::TargetArch::WSE2 &&
      targetArch != csl::TargetArch::WSE3)
    return byteAddress;

  auto ui16Ty = IntegerType::get(b.getContext(), 16, IntegerType::Unsigned);
  Value one = ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, 1));
  return BinaryOp::create(b, loc, ui16Ty, csl::BinaryOpKind::Shr, byteAddress,
                          one);
}

/// Emit non-volatile asm to convert a hardware u16-word address to a [*]u16
/// many-pointer used for compiler-generated arg-area writes. SDK 2.10 CSL
/// pointers are byte-addressed on both supported architectures, so the
/// numeric hardware word address must be doubled.
static Value emitU16ToManyPtr(OpBuilder &b, Location loc, Value u16Val,
                              csl::TargetArch targetArch) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto u16ManyPtrTy =
      csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/true);
  auto scratchLvalTy = csl::LValueType::get(u16ManyPtrTy, /*isConst=*/false);
  std::string name = "__asm_scratch_" + std::to_string(bitcastUidCounter++);
  Value scratch =
      VariableOp::create(b, loc, scratchLvalTy, name, Value()).getResult();

  Value byteAddress = u16Val;
  StringRef asmStr = "mov16 %[out] = %[inp]";
  if (targetArch == csl::TargetArch::WSE3) {
    // WSE-3 permits the pointer result register as an ALU destination.
    asmStr = "sll16.nf %[out] = %[inp], 1";
  } else {
    // WSE-2 materializes pointer results in its special r6 addressing
    // register, which cannot be an sll16 destination. Shift as an ordinary
    // u16 first, then use mov16 only for the integer-to-pointer bitcast.
    Value one = ConstantOp::create(b, loc, IntegerAttr::get(ui16Ty, 1));
    byteAddress =
        BinaryOp::create(b, loc, ui16Ty, csl::BinaryOpKind::Shl, u16Val, one);
  }

  InlineAsmOp::create(b, loc,
                      /*outputs=*/ValueRange{scratch},
                      /*inputs=*/ValueRange{byteAddress},
                      b.getStringAttr(asmStr),
                      /*output_names=*/b.getStrArrayAttr({"out"}),
                      /*output_constraints=*/b.getStrArrayAttr({"=r"}),
                      /*input_names=*/b.getStrArrayAttr({"inp"}),
                      /*input_constraints=*/b.getStrArrayAttr({"r"}),
                      /*clobbers=*/b.getStrArrayAttr({}),
                      /*is_volatile=*/UnitAttr()); // NOT volatile

  return LoadOp::create(b, loc, scratch);
}

/// Get a [*]u16 pointer to a named global.
/// WSE2: uses movr6i asm to load the symbol address directly as [*]u16.
/// WSE3: uses GlobalAddrAttr + ptrcast to [*]u16.
static std::atomic<unsigned> addrOfLabelUidCounter{0};
static Value getGlobalAsU16ManyPtr(OpBuilder &b, Location loc,
                                   StringRef globalName, Type globalTy,
                                   csl::TargetArch targetArch) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto u16ManyPtrTy =
      csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/true);

  if (targetArch == csl::TargetArch::WSE3) {
    auto lvalTy = csl::LValueType::get(globalTy, /*isConst=*/false);
    auto globalRef = csl::GlobalRefAttr::get(
        ctx, /*module_identifier=*/std::nullopt,
        FlatSymbolRefAttr::get(ctx, globalName), lvalTy);
    auto globalAddr = csl::GlobalAddrAttr::get(globalRef);
    Value ptr = ConstantOp::create(b, loc, globalAddr);
    return CastOp::create(b, loc, u16ManyPtrTy, ptr).getResult();
  }

  // WSE2: load symbol address via movr6i, output directly as [*]u16.
  auto scratchLvalTy = csl::LValueType::get(u16ManyPtrTy, /*isConst=*/false);
  std::string name = "__asm_addr_" + std::to_string(addrOfLabelUidCounter++);
  Value scratch =
      VariableOp::create(b, loc, scratchLvalTy, name, Value()).getResult();

  InlineAsmOp::create(b, loc,
                      /*outputs=*/ValueRange{scratch},
                      /*inputs=*/ValueRange{},
                      b.getStringAttr("movr6i r6 = " + globalName.str()),
                      /*output_names=*/b.getStrArrayAttr({"out"}),
                      /*output_constraints=*/b.getStrArrayAttr({"={r6}"}),
                      /*input_names=*/b.getStrArrayAttr({}),
                      /*input_constraints=*/b.getStrArrayAttr({}),
                      /*clobbers=*/b.getStrArrayAttr({}),
                      /*is_volatile=*/UnitAttr()); // NOT volatile

  return LoadOp::create(b, loc, scratch);
}

/// Write a u16 value to a [*]u16 pointer at a given u16-word offset.
static void storeToU16Ptr(OpBuilder &b, Location loc, Value u16ManyPtr,
                          Value offsetVal, Value value) {
  auto ctx = b.getContext();
  auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
  auto lvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
  Value lval = PtrIndexAccessOp::create(b, loc, lvalTy, u16ManyPtr, offsetVal);
  AssignOp::create(b, loc, lval, value);
}

//===----------------------------------------------------------------------===//
// Pool Access Helpers
//===----------------------------------------------------------------------===//

/// Get an lvalue to `__fiber_runtime__pool_<fieldName>[idx]`.
///
/// With the runtime's struct-of-arrays pool layout, this is a single
/// `ArrayIndexAccess` against the field's own global (each element is 2
/// bytes, so the back-end emits `base + i*2`), instead of the old
/// `ArrayIndexAccess + FieldAccess` against a struct-of-10-bytes pool.
static Value getPoolFieldLValue(OpBuilder &b, Location loc,
                                const FiberPoolInfo &pool, Value idx,
                                StringRef fieldName, Type fieldTy) {
  auto it = pool.fields.find(fieldName);
  assert(it != pool.fields.end() && "unknown pool field");
  const auto &field = it->second;
  auto arrayLvalTy = csl::LValueType::get(field.arrayTy, /*isConst=*/false);
  Value arrayLval =
      createGlobalRef(b, loc, arrayLvalTy, field.globalRef.getSymbolName());
  auto elemLvalTy = csl::LValueType::get(fieldTy, /*isConst=*/false);
  return ArrayIndexAccessOp::create(b, loc, elemLvalTy, arrayLval, idx);
}

/// Get a *u16 pointer to pool[idx].field (for passing to asm or runtime).
static Value getPoolFieldPtr(OpBuilder &b, Location loc,
                             const FiberPoolInfo &pool, Value idx,
                             StringRef fieldName, Type fieldTy) {
  Value lval = getPoolFieldLValue(b, loc, pool, idx, fieldName, fieldTy);
  auto ptrTy =
      csl::PointerType::get(fieldTy, /*isConst=*/false, /*isArray=*/false);
  return AddressOfOp::create(b, loc, ptrTy, lval);
}

//===----------------------------------------------------------------------===//
// Binding Info (replaces FiberAnalysis)
//===----------------------------------------------------------------------===//

/// Information about a single async task entry point.
struct FiberBinding {
  FuncOp funcOp;
  BuiltinCallOp bindOp;      ///< the @bind_*_task call
  uint16_t taskId;           ///< task ID from the bind call
  csl::BuiltinKind bindKind; ///< original bind variant
};

/// Extract a uint16_t task ID from a ConstantOp operand.
static std::optional<uint16_t> extractTaskIdValue(ConstantOp constOp) {
  TypedAttr val = constOp.getValue();
  if (auto localAttr = mlir::dyn_cast<csl::LocalTaskIdAttr>(val))
    return localAttr.getValue();
  if (auto controlAttr = mlir::dyn_cast<csl::ControlTaskIdAttr>(val))
    return controlAttr.getValue();
  return std::nullopt;
}

/// Collect @bind_*_task calls that bind async functions.
static LogicalResult
collectAsyncBindings(csl::CompilationUnitOp unit,
                     SmallVectorImpl<FiberBinding> &bindings) {
  auto isBindKind = [](csl::BuiltinKind k) {
    return k == csl::BuiltinKind::BindLocalTask ||
           k == csl::BuiltinKind::BindControlTask ||
           k == csl::BuiltinKind::BindDataTask;
  };

  LogicalResult result = success();
  unit.walk([&](BuiltinCallOp bindOp) {
    csl::BuiltinKind kind = bindOp.getBuiltin();
    if (!isBindKind(kind))
      return;
    if (bindOp.getArgs().size() < 2)
      return;

    csl::FuncRefAttr funcRef = getFuncRef(bindOp.getArgs()[0]);
    FuncOp funcOp;
    if (funcRef)
      funcOp = unit.lookupSymbol<FuncOp>(funcRef.getSymbolName());
    if (!funcOp || !funcOp.getIsAsync())
      return;

    auto constOp = bindOp.getArgs()[1].getDefiningOp<ConstantOp>();
    std::optional<uint16_t> taskId;
    if (constOp)
      taskId = extractTaskIdValue(constOp);
    if (!taskId) {
      bindOp.emitError("async task '")
          << funcOp.getSymName()
          << "' bound with @bind_*_task but task ID is not comptime-known";
      result = failure();
      return;
    }

    bindings.push_back(FiberBinding{funcOp, bindOp, *taskId, kind});

    LLVM_DEBUG(llvm::dbgs() << "  binding: " << funcOp.getSymName()
                            << " task_id=" << *taskId << "\n");
  });

  return result;
}

//===----------------------------------------------------------------------===//
// FiberLoweringPass
//===----------------------------------------------------------------------===//

struct FiberLoweringPass
    : public typed_csl::impl::FiberLoweringPassBase<FiberLoweringPass> {
  using FiberLoweringPassBase::FiberLoweringPassBase;

  FiberLoweringPass(const typed_csl::FiberLoweringPassOptions &opts,
                    typed_csl::FiberSizeSource *src)
      : FiberLoweringPassBase(opts), externalSizeSource(src) {}

  FiberLoweringPass(const FiberLoweringPass &other)
      : FiberLoweringPassBase(other),
        externalSizeSource(other.externalSizeSource) {}

  /// External per-callee size oracle (nullptr → fall back to
  /// `defaultSource`, the round-1 placeholder behavior).  Pass 2 of the
  /// two-round compile threads a `ResolvedSizeSource` in here.
  typed_csl::FiberSizeSource *externalSizeSource = nullptr;
  typed_csl::PlaceholderSizeSource defaultSource;

  typed_csl::FiberSizeSource &source() {
    return externalSizeSource ? *externalSizeSource : defaultSource;
  }

  /// Fiber pool info (populated in phase 2).
  FiberPoolInfo pool;

  /// Collected async task bindings.
  SmallVector<FiberBinding> bindings;

  /// For each @await_parallel with a result: per callee, the caller's
  /// variable that the worker stores the callee result into, or a null value
  /// for the first callee and for callees without a result.
  DenseMap<Operation *, SmallVector<Value>> resultSlots;

  /// Runtime functions (looked up from the inlined runtime module).
  FuncOp rtSetSpAndJmp;
  FuncOp rtSuspend;
  FuncOp rtCallerJoin;
  FuncOp rtCompletionHandler;
  FuncOp rtRootDone;
  FuncOp rtWorkerEntry;
  FuncOp rtInit;

  /// Look up all runtime functions by prefixed name. When hasAsync is true,
  /// all functions are required and a missing one emits an error and returns
  /// failure. Export stripping is handled by StripRuntimeExportsPass.
  LogicalResult findRuntimeFunctions(csl::CompilationUnitOp unit,
                                     bool hasAsync) {
    auto lookup = [&](StringRef suffix, bool required) -> FuncOp {
      auto name =
          (Twine(csl::CSLDialect::getFiberRuntimePrefix()) + suffix).str();
      auto funcOp = unit.lookupSymbol<FuncOp>(name);
      if (!funcOp && required) {
        unit.emitError("internal error: the async runtime has no function '")
            << suffix
            << "'; the runtime is imported automatically, please report "
               "this";
      }
      return funcOp;
    };
    rtSetSpAndJmp = lookup("set_sp_and_jmp", hasAsync);
    rtSuspend = lookup("suspend", hasAsync);
    rtCallerJoin = lookup("caller_join", hasAsync);
    rtCompletionHandler = lookup("completion_handler", hasAsync);
    rtRootDone = lookup("root_done", hasAsync);
    rtWorkerEntry = lookup("worker_entry", hasAsync);
    rtInit = lookup("init", hasAsync);
    if (hasAsync &&
        (!rtSetSpAndJmp || !rtSuspend || !rtCallerJoin ||
         !rtCompletionHandler || !rtRootDone || !rtWorkerEntry || !rtInit))
      return failure();

    if (rtWorkerEntry)
      rtWorkerEntry.setIsTask(true);
    return success();
  }

  /// Emit @current_task() → @get_int() to get the current task ID as u16.
  Value emitGetCurrentTaskId(OpBuilder &b, Location loc) {
    auto *ctx = b.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);
    Value currentTask =
        BuiltinCallOp::create(b, loc, localTaskIdTy,
                              csl::BuiltinKind::CurrentTask, ValueRange{})
            .getResult();
    return BuiltinCallOp::create(b, loc, ui16Ty, csl::BuiltinKind::GetInt,
                                 ValueRange{currentTask})
        .getResult();
  }

  void runOnOperation() override {
    auto unit = getOperation();
    IRRewriter rewriter(&getContext());

    // Phase 1: Collect async task bindings.
    bindings.clear();
    if (failed(collectAsyncBindings(unit, bindings)))
      return signalPassFailure();

    bool hasAsync = !bindings.empty();

    // Phase 1b: Validate that every async-reachable function has a stack
    // reach the resolver can statically bound.  Runs before the runtime
    // lookup so user-source errors fire even in fixtures that don't pull
    // in the runtime module.
    const IndirectCallTargets &indirectTargets =
        getAnalysis<IndirectCallTargets>();
    {
      FiberReachableFuncs reachable =
          computeFiberReachable(unit, indirectTargets);
      if (failed(validateFiberSources(unit, reachable, indirectTargets)))
        return signalPassFailure();
      SmallVector<FuncOp> rootTasks;
      for (const FiberBinding &binding : bindings)
        rootTasks.push_back(binding.funcOp);
      checkWorkerTasks(unit, reachable, indirectTargets, rootTasks,
                       targetArch);
    }

    if (failed(findRuntimeFunctions(unit, hasAsync)))
      return signalPassFailure();

    SmallVector<BuiltinCallOp> coroInitOps;
    unit.walk([&](BuiltinCallOp op) {
      if (op.getBuiltin() == csl::BuiltinKind::CoroRuntimeInit)
        coroInitOps.push_back(op);
    });

    if (!hasAsync) {
      // Nothing roots a fiber, so there is no runtime state to set up and
      // @coro_runtime_init() has nothing to do. Point at the call the user
      // can delete rather than silently accepting it.
      for (BuiltinCallOp op : coroInitOps) {
        op.emitRemark("@coro_runtime_init() is not needed here: no async "
                      "function is bound to a task, so no async runtime is "
                      "initialized; remove the call");
        rewriter.eraseOp(op);
      }
      LLVM_DEBUG(llvm::dbgs() << "No async task bindings — nothing to do\n");
      return;
    }

    // Without the runtime's init(), no async task has a stack to run on.
    if (coroInitOps.empty()) {
      FiberBinding first = bindings.front();
      first.bindOp.emitError("async task '")
          << first.funcOp.getSymName()
          << "' is bound to a task, but the program never calls "
             "@coro_runtime_init(); call it once before the first async "
             "task is activated, for example in the function the host "
             "launches";
      return signalPassFailure();
    }

    // Lower @coro_runtime_init() to a call to the runtime's init(). The body
    // of that function is generated by generateInitFunction below.
    for (BuiltinCallOp op : coroInitOps) {
      rewriter.setInsertionPoint(op);
      CallOp::create(rewriter, op.getLoc(), TypeRange{},
                     FlatSymbolRefAttr::get(unit.getContext(),
                                            rtInit.getSymName()),
                     ValueRange{}, ArrayAttr(), ArrayAttr());
      rewriter.eraseOp(op);
    }

    // Instrumentation (opt-in): give every @await_parallel site its own
    // function so a profiler can charge the site's cost to it.  Must run
    // BEFORE the capture below, so the wrapper is an ordinary sync callee in
    // the captured graph and the stack resolver sizes its frame like any
    // other.  A wrapper introduced after the capture would be invisible to
    // the resolver and the root fiber stack would come out under-sized.
    if (failed(outlineAwaitParallelSites(unit, rewriter)))
      return signalPassFailure();

    // Route the results of @await_parallel workers through wrappers. Must
    // run before the capture below so the stack resolver sizes the wrappers.
    if (failed(prepareAwaitParallelResults(unit, rewriter)))
      return signalPassFailure();

    // Capture the call graph onto each FuncOp as discardable attributes
    // before subsequent phases rewrite @await_parallel / @await away.
    // ProcessTileCodePass extracts these after stage A to populate the
    // FiberSizingContext for the resolver.
    captureCallGraphAsAttributes(unit, rewriter, indirectTargets);

    // Unwrap FutureType from async function results.
    rewriteAsyncResults(unit, rewriter);

    // Phase 2: Generate fiber infrastructure (pool, stacks, init function).
    if (failed(generateInfrastructure(unit, rewriter)))
      return signalPassFailure();

    // Phase 3: Fuse sequential @await (call + await → direct call).
    if (failed(fuseSequentialAwaits(unit, rewriter)))
      return signalPassFailure();

    // Phase 4: Lower @await_parallel with dynamic task allocation.
    if (failed(lowerAwaitParallel(unit, rewriter)))
      return signalPassFailure();

    // Phase 5: Lower SuspendOp → call to runtime suspend.
    if (failed(lowerSuspendOps(unit, rewriter)))
      return signalPassFailure();

    // Phase 6: Cleanup — remove async/coro markers.
    cleanupMarkers(unit, rewriter);
  }

  //===--------------------------------------------------------------------===//
  // Instrumentation: outline @await_parallel sites (opt-in)
  //===--------------------------------------------------------------------===//

  /// Move each `@await_parallel` site into its own ordinary async fn, awaited
  /// at the original location, so the LLVM backend can give the site a DWARF
  /// inline scope. Without this the whole lowered sequence — claim, bind,
  /// arg-area writes, activate, join — has only the enclosing user function's
  /// debug scope.
  ///
  /// Deliberately set neither `inline` nor `noinline`: frontend inlining would
  /// erase the function-level debug scope, while noinline is the distortion
  /// this profiling mode exists to avoid. The backend is then free to inline
  /// the wrapper and describe that decision with DW_TAG_inlined_subroutine.
  ///
  /// The site's synchronous callee moves into the wrapper too. The profiler
  /// takes only the unbroken suffix of coroutine-generated DWARF frames, so a
  /// user callee ends the wrapper's exclusive range even if a future backend
  /// configuration inlines that callee as well.
  LogicalResult outlineAwaitParallelSites(csl::CompilationUnitOp unit,
                                          IRRewriter &rewriter) {
    if (!outlineAwaitParallel)
      return success();

    SmallVector<AwaitParallelOp> apOps;
    unit.walk([&](AwaitParallelOp op) { apOps.push_back(op); });

    unsigned siteIdx = 0;
    for (AwaitParallelOp apOp : apOps)
      if (failed(outlineOneAwaitParallelSite(unit, rewriter, apOp, siteIdx++)))
        return failure();
    return success();
  }

  LogicalResult outlineOneAwaitParallelSite(csl::CompilationUnitOp unit,
                                            IRRewriter &rewriter,
                                            AwaitParallelOp apOp,
                                            unsigned siteIdx) {
    auto *ctx = unit.getContext();
    Location loc = apOp.getLoc();

    // The ops that move: each future's defining call, then the await itself.
    SmallVector<CallOp> definingCalls;
    DenseSet<Operation *> moving;
    for (Value future : apOp.getFutures()) {
      auto call = future.getDefiningOp<CallOp>();
      if (!call)
        return apOp.emitError("@await_parallel future is not produced by a "
                              "call — cannot outline this site");
      if (!future.hasOneUse())
        return apOp.emitError("@await_parallel future has other uses — "
                              "cannot outline this site");
      definingCalls.push_back(call);
      moving.insert(call);
    }
    moving.insert(apOp);

    // Anything the moved ops read but do not define becomes a parameter.
    // The callees' arguments are the whole of it: futures never escape the
    // site, and its result, if any, becomes the wrapper's result.
    SetVector<Value> params;
    for (CallOp call : definingCalls)
      for (Value operand : call.getArgOperands())
        if (!moving.contains(operand.getDefiningOp()))
          params.insert(operand);

    SmallVector<Type> paramTypes;
    for (Value p : params)
      paramTypes.push_back(p.getType());

    std::string wrapperName =
        "__fiber_await_parallel_" + std::to_string(siteIdx);
    Value siteResult = apOp.getResult();
    Type resultType = siteResult ? siteResult.getType() : Type();
    auto futureTy = csl::FutureType::get(ctx, /*module_identifier=*/std::nullopt,
                                         FlatSymbolRefAttr::get(ctx,
                                                                wrapperName),
                                         resultType);
    auto wrapperTy = FunctionType::get(ctx, paramTypes, {futureTy});

    // An async fn: the site suspends in caller_join, so the wrapper has to
    // be able to suspend too. It returns the site's result, if any.
    rewriter.setInsertionPointToEnd(unit.getBody());
    auto wrapper = FuncOp::create(
        rewriter, loc, rewriter.getStringAttr(wrapperName),
        TypeAttr::get(wrapperTy),
        /*is_task=*/UnitAttr(),
        /*is_inline=*/UnitAttr(),
        /*is_noinline=*/UnitAttr(),
        /*is_export=*/UnitAttr(),
        /*is_extern=*/UnitAttr(),
        /*is_async=*/rewriter.getUnitAttr(),
        /*stack_size=*/IntegerAttr(),
        /*linkname=*/StringAttr(),
        /*arg_attrs=*/ArrayAttr(),
        /*res_attrs=*/ArrayAttr(),
        /*async_result_type=*/
        resultType ? TypeAttr::get(resultType) : TypeAttr(),
        /*max_recursion_depth=*/IntegerAttr());

    // A TypedCSL entry block takes each parameter as a const lvalue, not as
    // the bare value (FuncOp::verify enforces it), so the body has to load
    // one before the moved ops can use it in place of the original operand.
    SmallVector<Type> blockArgTypes;
    for (Type paramTy : paramTypes)
      blockArgTypes.push_back(csl::LValueType::get(paramTy, /*isConst=*/true));
    SmallVector<Location> argLocs(paramTypes.size(), loc);
    Block *body = rewriter.createBlock(&wrapper.getBody(), {}, blockArgTypes,
                                       argLocs);

    rewriter.setInsertionPointToStart(body);
    IRMapping mapping;
    for (auto [outer, arg] : llvm::zip(params, body->getArguments()))
      mapping.map(outer, LoadOp::create(rewriter, loc, outer.getType(), arg)
                             .getResult());
    for (CallOp call : definingCalls)
      rewriter.clone(*call.getOperation(), mapping);
    Operation *clonedSite = rewriter.clone(*apOp.getOperation(), mapping);
    if (siteResult)
      ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1,
                       clonedSite->getResult(0));
    else
      ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);

    // Replace the site with `await wrapper(args)`.
    rewriter.setInsertionPoint(apOp);
    auto callWrapper = CallOp::create(
        rewriter, loc, TypeRange{futureTy},
        FlatSymbolRefAttr::get(ctx, wrapperName), params.getArrayRef(),
        /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());
    auto awaitWrapper =
        AwaitOp::create(rewriter, loc, resultType, callWrapper->getResult(0),
                        /*offload=*/UnitAttr());
    if (siteResult)
      rewriter.replaceAllUsesWith(siteResult, awaitWrapper.getResult());

    rewriter.eraseOp(apOp);
    for (CallOp call : definingCalls)
      rewriter.eraseOp(call);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Results of @await_parallel
  //===--------------------------------------------------------------------===//

  /// A worker runs its callee with completion_handler as return address, so
  /// the callee's return value would be lost. Each worker callee with a
  /// result therefore runs through a wrapper that stores the result into a
  /// variable of the caller, which the caller reads after the join. The first
  /// callee runs synchronously and returns its result directly.
  LogicalResult prepareAwaitParallelResults(csl::CompilationUnitOp unit,
                                            IRRewriter &rewriter) {
    resultSlots.clear();
    SmallVector<AwaitParallelOp> apOps;
    unit.walk([&](AwaitParallelOp op) {
      if (op.getResult())
        apOps.push_back(op);
    });

    unsigned slotCounter = 0;
    for (AwaitParallelOp apOp : apOps) {
      SmallVector<Value> slots(apOp.getFutures().size());
      for (unsigned i = 1; i < apOp.getFutures().size(); ++i) {
        Value future = apOp.getFutures()[i];
        auto futureTy = mlir::cast<csl::FutureType>(future.getType());
        Type resultType = futureTy.getResultType();
        if (!resultType)
          continue;
        auto call = future.getDefiningOp<CallOp>();
        FuncOp callee = unit.lookupSymbol<FuncOp>(futureTy.getSymbolName());
        if (!call || !callee || !future.hasOneUse())
          return apOp.emitError("internal error: @await_parallel callee ")
                 << i + 1 << " is not a direct call";

        auto resultPtrTy = csl::PointerType::get(resultType, /*isConst=*/false,
                                                 /*isArray=*/false);
        SmallVector<Type> wrapperInputs(call.getArgOperands().getTypes());
        wrapperInputs.push_back(resultPtrTy);
        if (!ArgRegisterAllocator::fits(wrapperInputs))
          return apOp.emitError("the arguments of @await_parallel callee '")
                 << callee.getSymName()
                 << "' and the address its result is stored to do not fit "
                    "into its four 16-bit argument registers (a 32-bit "
                    "argument takes two, larger arguments are not "
                    "supported); pass fewer arguments, or a pointer to a "
                    "struct holding them";

        FuncOp wrapper = getOrCreateResultWrapper(unit, rewriter, callee);
        Location loc = call.getLoc();
        rewriter.setInsertionPoint(call);
        Value slot = VariableOp::create(
            rewriter, loc,
            csl::LValueType::get(resultType, /*isConst=*/false),
            "__await_parallel_result_" + std::to_string(slotCounter++),
            Value());
        Value slotAddress =
            AddressOfOp::create(rewriter, loc, resultPtrTy, slot).getResult();
        SmallVector<Value> args(call.getArgOperands());
        args.push_back(slotAddress);
        auto wrapperCall = CallOp::create(
            rewriter, loc, wrapper.getFunctionType().getResults(),
            FlatSymbolRefAttr::get(wrapper.getSymNameAttr()), args,
            /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());
        apOp->setOperand(i, wrapperCall->getResult(0));
        rewriter.eraseOp(call);
        slots[i] = slot;
      }
      resultSlots[apOp] = std::move(slots);
    }
    return success();
  }

  /// `async fn wrapper(params..., result: *T) void { result.* =
  /// @await(callee(params...)); }`, shared by all sites that run `callee` on
  /// a worker.
  FuncOp getOrCreateResultWrapper(csl::CompilationUnitOp unit,
                                  IRRewriter &rewriter, FuncOp callee) {
    std::string name =
        ("__await_parallel_result_" + callee.getSymName()).str();
    if (FuncOp existing = unit.lookupSymbol<FuncOp>(name))
      return existing;

    MLIRContext *ctx = unit.getContext();
    Location loc = callee.getLoc();
    FunctionType calleeType = callee.getFunctionType();
    auto calleeFutureTy = mlir::cast<csl::FutureType>(calleeType.getResult(0));
    Type resultType = calleeFutureTy.getResultType();
    auto resultPtrTy = csl::PointerType::get(resultType, /*isConst=*/false,
                                             /*isArray=*/false);
    SmallVector<Type> inputs(calleeType.getInputs());
    inputs.push_back(resultPtrTy);
    auto futureTy = csl::FutureType::get(
        ctx, /*module_identifier=*/std::nullopt,
        FlatSymbolRefAttr::get(ctx, name), /*result_type=*/Type());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(callee);
    auto wrapper = FuncOp::create(
        rewriter, loc, rewriter.getStringAttr(name),
        TypeAttr::get(FunctionType::get(ctx, inputs, {futureTy})),
        /*is_task=*/UnitAttr(),
        /*is_inline=*/UnitAttr(),
        /*is_noinline=*/UnitAttr(),
        /*is_export=*/UnitAttr(),
        /*is_extern=*/UnitAttr(),
        /*is_async=*/rewriter.getUnitAttr(),
        /*stack_size=*/IntegerAttr(),
        /*linkname=*/StringAttr(),
        /*arg_attrs=*/ArrayAttr(),
        /*res_attrs=*/ArrayAttr(),
        /*async_result_type=*/TypeAttr(),
        /*max_recursion_depth=*/IntegerAttr());

    SmallVector<Type> blockArgTypes;
    for (Type input : inputs)
      blockArgTypes.push_back(csl::LValueType::get(input, /*isConst=*/true));
    SmallVector<Location> argLocs(inputs.size(), loc);
    Block *body =
        rewriter.createBlock(&wrapper.getBody(), {}, blockArgTypes, argLocs);
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> params;
    for (BlockArgument arg : body->getArguments())
      params.push_back(LoadOp::create(rewriter, loc, arg).getResult());
    Value resultPtr = params.pop_back_val();

    auto call = CallOp::create(rewriter, loc, TypeRange{calleeFutureTy},
                               FlatSymbolRefAttr::get(callee.getSymNameAttr()),
                               params, /*arg_attrs=*/ArrayAttr(),
                               /*res_attrs=*/ArrayAttr());
    Value result = AwaitOp::create(rewriter, loc, resultType,
                                   call->getResult(0), /*offload=*/UnitAttr())
                       .getResult();
    Value resultLValue = DereferenceOp::create(
        rewriter, loc, csl::LValueType::get(resultType, /*isConst=*/false),
        resultPtr);
    AssignOp::create(rewriter, loc, resultLValue, result);
    ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);
    return wrapper;
  }

  /// Replace the result of `apOp` by the tuple of the callee results, after
  /// the join: the first callee's result `firstResult` and the values the
  /// workers stored into the result slots.
  void replaceAwaitParallelResult(IRRewriter &rewriter, Location loc,
                                  AwaitParallelOp apOp, Value firstResult) {
    Value result = apOp.getResult();
    if (!result)
      return;
    // The worker callees now return through their wrappers, so the slots
    // tell which of them have a result.
    ArrayRef<Value> slots = resultSlots.lookup(apOp);
    SmallVector<Value> fields;
    if (firstResult) {
      // The multi-callee path binds the first result to a variable.
      if (mlir::isa<csl::LValueType>(firstResult.getType()))
        firstResult = LoadOp::create(rewriter, loc, firstResult).getResult();
      fields.push_back(firstResult);
    }
    for (Value slot : slots.drop_front())
      if (slot)
        fields.push_back(LoadOp::create(rewriter, loc, slot).getResult());
    Value tuple =
        ConstructTupleOp::create(rewriter, loc, result.getType(), fields);
    rewriter.replaceAllUsesWith(result, tuple);
  }

  //===--------------------------------------------------------------------===//
  // Capture call graph as IR attributes (before lowering rewrites it away)
  //===--------------------------------------------------------------------===//

  /// Attach per-FuncOp discardable attributes recording the call graph
  /// shape — sync callees, parallel-await sites, and root-task marker.
  /// After stage A finishes, the post-cslc fiber stack resolver reads
  /// these via `extractCapturedCallGraph` to compute max_stack values.
  /// Attribute names live in FiberSizingContext.h.
  void captureCallGraphAsAttributes(
      csl::CompilationUnitOp unit, IRRewriter &rewriter,
      const IndirectCallTargets &indirectTargets) {
    auto *ctx = unit.getContext();

    // Mark root tasks (functions bound via @bind_*_task).
    DenseSet<StringAttr> rootTaskNames;
    for (auto &binding : bindings)
      rootTaskNames.insert(binding.funcOp.getSymNameAttr());

    // Symbol names for runtime calls that the lowering inserts AFTER
    // this capture step but whose stack reach still has to be charged
    // to the enclosing function.  See the SuspendOp / AwaitParallelOp
    // branches below.
    std::string rtSuspendName =
        (Twine(csl::CSLDialect::getFiberRuntimePrefix()) + "suspend").str();
    StringAttr rtSuspendAttr = rewriter.getStringAttr(rtSuspendName);

    SmallVector<FuncOp> funcs;
    unit.walk([&](FuncOp funcOp) { funcs.push_back(funcOp); });

    for (FuncOp funcOp : funcs) {
      if (rootTaskNames.contains(funcOp.getSymNameAttr()))
        funcOp->setAttr(kCapturedRootTaskAttrName, rewriter.getUnitAttr());

      // Collect sync direct calls and parallel-await sites in source order
      // by walking the body once.
      SmallVector<Attribute> syncCallees;
      SmallVector<Attribute> parallelSites;
      funcOp.walk([&](Operation *op) {
        if (auto call = mlir::dyn_cast<CallOp>(op)) {
          // Skip CallOps that produce a future — those are consumed by
          // an AwaitOp (becomes a sync call after `fuseSequentialAwaits`)
          // or by an AwaitParallelOp.  In both cases the callee is
          // recorded via the await op below; recording it here too
          // would double-count.
          for (Type r : call.getResultTypes()) {
            if (mlir::isa<csl::FutureType>(r))
              return;
          }
          syncCallees.push_back(call.getCalleeAttr().getRootReference());
        } else if (auto icall = mlir::dyn_cast<IndirectCallOp>(op)) {
          // An indirect call is a sync call to one of its possible callees;
          // charging every one of them as a sync site makes the resolver
          // take the deepest.
          for (FuncOp callee : indirectTargets.getPossibleCallees(icall))
            syncCallees.push_back(callee.getSymNameAttr());
        } else if (auto aw = mlir::dyn_cast<AwaitOp>(op)) {
          // `await fn()` fuses into a direct sync call.  The callee's
          // contribution stacks on the caller's frame just like any
          // other sync call.
          if (auto ft =
                  mlir::dyn_cast<csl::FutureType>(aw.getFuture().getType()))
            syncCallees.push_back(ft.getSymbolName().getAttr());
        } else if (mlir::isa<SuspendOp>(op)) {
          // `suspend` lowers post-capture to a `call rtSuspend(...)`.
          // The runtime frame sits below the enclosing function's
          // frame at runtime, so charge it to the captured graph the
          // same way as any other sync call.
          syncCallees.push_back(rtSuspendAttr);
        } else if (auto ap = mlir::dyn_cast<AwaitParallelOp>(op)) {
          SmallVector<Attribute> calleesAtSite;
          for (Value future : ap.getFutures()) {
            if (auto ft = mlir::dyn_cast<csl::FutureType>(future.getType()))
              calleesAtSite.push_back(ft.getSymbolName().getAttr());
          }
          parallelSites.push_back(ArrayAttr::get(ctx, calleesAtSite));
        }
      });

      if (!syncCallees.empty())
        funcOp->setAttr(kCapturedSyncCalleesAttrName,
                        ArrayAttr::get(ctx, syncCallees));
      if (!parallelSites.empty())
        funcOp->setAttr(kCapturedParallelSitesAttrName,
                        ArrayAttr::get(ctx, parallelSites));
    }
  }

  //===--------------------------------------------------------------------===//
  // Rewrite async function results (unwrap FutureType)
  //===--------------------------------------------------------------------===//

  void rewriteAsyncResults(csl::CompilationUnitOp unit, IRRewriter &rewriter) {
    auto *ctx = unit.getContext();

    // Collect new function types for async functions (unwrap FutureType).
    DenseMap<StringAttr, FunctionType> newFuncTypes;

    SmallVector<FuncOp> asyncFuncs;
    unit.walk([&](FuncOp funcOp) {
      if (funcOp.getIsAsync())
        asyncFuncs.push_back(funcOp);
    });

    for (FuncOp funcOp : asyncFuncs) {
      auto oldFnTy = funcOp.getFunctionType();

      // Unwrap FutureType from results: future<@fn, T> → T, future<@fn> → void
      SmallVector<Type> newResults;
      bool changed = false;
      for (Type resTy : oldFnTy.getResults()) {
        if (auto futureTy = mlir::dyn_cast<csl::FutureType>(resTy)) {
          changed = true;
          if (futureTy.getResultType())
            newResults.push_back(futureTy.getResultType());
        } else {
          newResults.push_back(resTy);
        }
      }
      if (!changed)
        continue;

      auto newFnTy = FunctionType::get(ctx, oldFnTy.getInputs(), newResults);
      funcOp.setFunctionType(newFnTy);
      newFuncTypes[funcOp.getSymNameAttr()] = newFnTy;
    }

    if (newFuncTypes.empty())
      return;

    // Update all FuncRefAttr references to rewritten functions across the IR.
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](csl::FuncRefAttr funcRef) -> std::optional<Attribute> {
          StringAttr symName = funcRef.getSymbolName().getAttr();
          auto it = newFuncTypes.find(symName);
          if (it == newFuncTypes.end())
            return std::nullopt;
          return csl::FuncRefAttr::get(ctx, funcRef.getModuleIdentifier(),
                                       funcRef.getSymbolName(), it->second);
        });
    replacer.recursivelyReplaceElementsIn(unit, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    // A constant's result type is derived from its value, which the
    // replacement above may have retyped (a reference or an address of one).
    unit.walk([&](ConstantOp constant) {
      Type valueType = constant.getValue().getType();
      if (constant.getResult().getType() != valueType)
        constant.getResult().setType(valueType);
    });
  }

  //===--------------------------------------------------------------------===//
  // Phase 2: Generate fiber infrastructure
  //===--------------------------------------------------------------------===//

  LogicalResult generateInfrastructure(csl::CompilationUnitOp unit,
                                       IRRewriter &rewriter) {
    auto ctx = unit.getContext();
    Location loc = unit.getLoc();

    // --- Look up the per-field pool globals (struct-of-arrays layout) ---
    static constexpr std::array<StringRef, 5> kFieldNames = {
        "sp", "pc", "caller_wid", "join_counter_ptr", "stack_bottom"};
    StringRef rtPrefix = csl::CSLDialect::getFiberRuntimePrefix();
    pool.fields.clear();
    for (StringRef fieldName : kFieldNames) {
      auto globalName = (Twine(rtPrefix) + "pool_" + fieldName).str();
      auto fieldGlobal = unit.lookupSymbol<GlobalOp>(globalName);
      if (!fieldGlobal) {
        unit.emitError("internal error: the async runtime has no pool "
                       "field '")
            << fieldName
            << "'; the runtime is imported automatically, please report this";
        return failure();
      }
      auto arrayTy =
          mlir::cast<csl::ArrayType>(fieldGlobal.getTypeAttr().getValue());
      auto arrayLvalTy = csl::LValueType::get(arrayTy, /*isConst=*/false);
      FiberPoolInfo::Field f;
      f.globalRef = csl::GlobalRefAttr::get(
          ctx, /*module_identifier=*/std::nullopt,
          FlatSymbolRefAttr::get(ctx, fieldGlobal.getSymName()), arrayLvalTy);
      f.arrayTy = arrayTy;
      f.elementTy = arrayTy.getElementType();
      pool.fields[fieldName] = f;
    }
    pool.poolSize = pool.fields.lookup("sp").arrayTy.getShape()[0];

    auto ui32Ty = IntegerType::get(ctx, 32, IntegerType::Unsigned);

    // --- Per-root-task fiber stack globals ---
    // Array dim comes from the FiberSizeSource: pass 1 → placeholder
    // (or the user's `stack_size(N)` if set); pass 2 → measured value
    // from the round-1 ELF.
    for (auto &binding : bindings) {
      uint32_t rootU32Words = source().getRootStackU32Words(binding.funcOp);
      auto stackArrayTy = csl::ArrayType::get({rootU32Words}, ui32Ty);
      std::string stackName =
          ("__fiber_stack_" + binding.funcOp.getSymName()).str();

      rewriter.setInsertionPointToEnd(unit.getBody());
      GlobalOp::create(rewriter, loc,
                       /*type=*/TypeAttr::get(stackArrayTy),
                       /*sym_name=*/rewriter.getStringAttr(stackName),
                       /*initializer=*/TypedAttr(),
                       /*is_const=*/UnitAttr(),
                       /*is_export=*/rewriter.getUnitAttr(),
                       /*is_extern=*/UnitAttr(),
                       /*linkname=*/StringAttr(),
                       /*alignment=*/IntegerAttr());
    }

    // --- Mark every parallel-callee async fn as noinline ---
    // Inlining a parallel callee into its caller would merge its frame
    // into the caller's, breaking the per-callee partition arithmetic
    // (the partition expects each callee's max_stack to be a separate
    // frame in `.stack_sizes`).  We also rely on the noinline guarantee
    // to keep the resolver's call-graph reconciliation against DWARF
    // simple — callees that get inlined here become "inlined-only" in
    // DWARF, which is fine for the formula but never desirable.
    llvm::StringSet<> parallelCallees;
    unit.walk([&](AwaitParallelOp apOp) {
      for (Value future : apOp.getFutures()) {
        if (auto futureTy = mlir::dyn_cast<csl::FutureType>(future.getType()))
          parallelCallees.insert(futureTy.getSymbolName().getValue());
      }
    });

    for (StringRef calleeSym : parallelCallees.keys()) {
      FuncOp calleeFunc = unit.lookupSymbol<FuncOp>(calleeSym);
      if (!calleeFunc) {
        unit.emitError("@await_parallel references unknown async fn '")
            << calleeSym << "'";
        return failure();
      }
      calleeFunc.setIsNoinline(true);
    }

    // --- Verify that the runtime module reached us ---
    // InjectCoroRuntimePass imports it into every program that mentions
    // coroutines, so a miss here means either that pass did not run or the
    // program declared its own conflicting __fiber_runtime.
    if (!rtInit) {
      unit.emitError("internal error: the async runtime has no function "
                     "'init'; the runtime is imported automatically, please "
                     "report this");
      return failure();
    }

    // --- Rewrite the user's @bind_*_task(async_fn, task_id) to point at
    //     worker_entry instead of the (now non-task) async function.
    auto voidFnType = FunctionType::get(ctx, {}, {});
    for (auto &binding : bindings) {
      BuiltinCallOp bindOp = binding.bindOp;
      rewriter.setInsertionPoint(bindOp);
      Value workerEntryRef = createFuncRef(
          rewriter, bindOp.getLoc(), voidFnType, rtWorkerEntry.getSymName());
      Value oldFnRef = bindOp.getArgs()[0];
      rewriter.modifyOpInPlace(
          bindOp, [&]() { bindOp.getArgsMutable()[0].assign(workerEntryRef); });
      if (oldFnRef.use_empty())
        if (Operation *def = oldFnRef.getDefiningOp())
          rewriter.eraseOp(def);
    }

    // --- Per-root-task reset functions ---
    for (auto &binding : bindings) {
      generateRootResetFunction(unit, rewriter, binding);
    }

    // --- Replace init extern with actual implementation ---
    generateInitFunction(unit, rewriter, rtInit);

    return success();
  }

  /// Emit the arg area + pool sp/pc setup for a root task binding.
  /// Writes reset function address to r15 arg area slot,
  /// and sets pool[taskId].sp and pool[taskId].pc for a fresh start.
  void emitRootTaskSetup(IRRewriter &rewriter, Location loc,
                         FiberBinding &binding, StringRef resetName) {
    auto ctx = rewriter.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto ui32Ty = IntegerType::get(ctx, 32, IntegerType::Unsigned);

    Value taskIdVal = ConstantOp::create(
        rewriter, loc, IntegerAttr::get(ui16Ty, binding.taskId));

    // Get a [*]u16 pointer to the fiber stack.  The dim comes from the
    // FiberSizeSource — same value in both rounds for any function with
    // a `stack_size(N)` override; resolver-computed in round 2 otherwise.
    uint32_t rootU32Words = source().getRootStackU32Words(binding.funcOp);
    std::string stackName =
        ("__fiber_stack_" + binding.funcOp.getSymName()).str();
    auto stackArrayTy = csl::ArrayType::get({rootU32Words}, ui32Ty);
    Value stackU16Ptr = getGlobalAsU16ManyPtr(rewriter, loc, stackName,
                                              stackArrayTy, targetArch);

    unsigned argAreaWordOffset = rootU32Words * 2 - kArgAreaSlots;

    LLVM_DEBUG(llvm::dbgs()
               << "fiber-lowering: root task setup for '"
               << binding.funcOp.getSymName() << "' (task id " << binding.taskId
               << "): __fiber_stack = [" << rootU32Words << "]u32 ("
               << (rootU32Words * 4)
               << " B); argAreaWordOffset = " << argAreaWordOffset
               << " (u16 words); reset entry = '" << resetName << "'\n");

    // Write reset function address to r15 slot (slot 4 in arg area).
    Value argAreaOffsetVal = ConstantOp::create(
        rewriter, loc, IntegerAttr::get(ui16Ty, argAreaWordOffset));
    auto voidFnType = FunctionType::get(ctx, {}, {});
    auto resetRef = csl::FuncRefAttr::get(
        ctx, /*module_identifier=*/std::nullopt,
        FlatSymbolRefAttr::get(ctx, resetName), voidFnType);
    auto resetAddr = csl::FuncAddrAttr::get(resetRef);
    Value resetAddrVal = ConstantOp::create(rewriter, loc, resetAddr);
    Value resetU16 = emitBitcastToU16(rewriter, loc, resetAddrVal, targetArch);
    resetU16 = emitHardwareWordAddress(rewriter, loc, resetU16, targetArch);
    Value r15OffsetVal = ConstantOp::create(
        rewriter, loc,
        IntegerAttr::get(ui16Ty, argAreaWordOffset + 4)); // slot 4 = r15
    storeToU16Ptr(rewriter, loc, stackU16Ptr, r15OffsetVal, resetU16);

    // pool[taskId].sp = pointer to the arg area base.
    auto u16LvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
    Value argAreaLval = PtrIndexAccessOp::create(rewriter, loc, u16LvalTy,
                                                 stackU16Ptr, argAreaOffsetVal);
    auto u16PtrTy =
        csl::PointerType::get(ui16Ty, /*isConst=*/false, /*isArray=*/false);
    Value argAreaPtr =
        AddressOfOp::create(rewriter, loc, u16PtrTy, argAreaLval);
    Value argAreaByteAddress =
        emitBitcastToU16(rewriter, loc, argAreaPtr, targetArch);
    Value argAreaSp =
        emitHardwareWordAddress(rewriter, loc, argAreaByteAddress, targetArch);
    AssignOp::create(rewriter, loc,
                     getPoolFieldLValue(rewriter, loc, pool, taskIdVal, "sp",
                                        pool.spFieldTy()),
                     argAreaSp);

    // pool[taskId].pc = &func (ptrcast to pc field type).
    auto funcRefAttr = csl::FuncRefAttr::get(
        ctx, /*module_identifier=*/std::nullopt,
        FlatSymbolRefAttr::get(ctx, binding.funcOp.getSymName()),
        binding.funcOp.getFunctionType());
    auto funcAddr = csl::FuncAddrAttr::get(funcRefAttr);
    Value funcAddrVal = ConstantOp::create(rewriter, loc, funcAddr);
    Value funcByteAddress =
        emitBitcastToU16(rewriter, loc, funcAddrVal, targetArch);
    Value funcPc =
        emitHardwareWordAddress(rewriter, loc, funcByteAddress, targetArch);
    AssignOp::create(rewriter, loc,
                     getPoolFieldLValue(rewriter, loc, pool, taskIdVal, "pc",
                                        pool.pcFieldTy()),
                     funcPc);
  }

  /// Generate a per-root-task reset function that re-initializes the pool
  /// entry and arg area for a fresh start, then terminates.  Entered via
  /// jmp r15 when the async function returns.
  void generateRootResetFunction(csl::CompilationUnitOp unit,
                                 IRRewriter &rewriter, FiberBinding &binding) {
    auto ctx = unit.getContext();
    Location loc = unit.getLoc();

    std::string resetName =
        ("__fiber_reset_" + binding.funcOp.getSymName()).str();
    auto voidFnType = FunctionType::get(ctx, {}, {});

    rewriter.setInsertionPointToEnd(unit.getBody());
    auto resetFunc =
        FuncOp::create(rewriter, loc, rewriter.getStringAttr(resetName),
                       TypeAttr::get(voidFnType),
                       /*is_task=*/UnitAttr(),
                       /*is_inline=*/UnitAttr(),
                       /*is_noinline=*/UnitAttr(),
                       /*is_export=*/UnitAttr(),
                       /*is_extern=*/UnitAttr(),
                       /*is_async=*/UnitAttr(),
                       /*stack_size=*/IntegerAttr(),
                       /*linkname=*/StringAttr(),
                       /*arg_attrs=*/ArrayAttr(),
                       /*res_attrs=*/ArrayAttr(),
                       /*async_result_type=*/TypeAttr(),
                       /*max_recursion_depth=*/IntegerAttr());

    Block *body = rewriter.createBlock(&resetFunc.getBody());
    rewriter.setInsertionPointToStart(body);

    emitRootTaskSetup(rewriter, loc, binding, resetName);

    // Terminate the task (entered via jmp r15, no return address).
    auto emptyStrArr = rewriter.getStrArrayAttr({});
    InlineAsmOp::create(rewriter, loc, /*outputs=*/ValueRange{},
                        /*inputs=*/ValueRange{},
                        rewriter.getStringAttr("nop.term"),
                        /*output_names=*/emptyStrArr,
                        /*output_constraints=*/emptyStrArr,
                        /*input_names=*/emptyStrArr,
                        /*input_constraints=*/emptyStrArr,
                        /*clobbers=*/emptyStrArr,
                        /*is_volatile=*/UnitAttr::get(ctx));

    UnreachableOp::create(rewriter, loc, /*numBreakingRegions=*/1);
  }

  /// Replace the runtime's extern init() with the actual implementation.
  /// Emits the root task setup (arg area + pool sp/pc) inline for each
  /// binding, plus the remaining pool fields.
  void generateInitFunction(csl::CompilationUnitOp unit, IRRewriter &rewriter,
                            FuncOp initFn) {
    auto ctx = unit.getContext();
    Location loc = unit.getLoc();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    // Replace the function body (whether extern or a stub) with init code.
    initFn.setIsExtern(false);
    initFn.setLinkname(std::nullopt);

    // Clear any existing body blocks.
    Region &bodyRegion = initFn.getBody();
    while (!bodyRegion.empty())
      rewriter.eraseBlock(&bodyRegion.front());

    Block *body = rewriter.createBlock(&bodyRegion);
    rewriter.setInsertionPointToStart(body);

    for (auto &binding : bindings) {
      std::string resetName =
          ("__fiber_reset_" + binding.funcOp.getSymName()).str();
      emitRootTaskSetup(rewriter, loc, binding, resetName);

      Value taskIdVal = ConstantOp::create(
          rewriter, loc, IntegerAttr::get(ui16Ty, binding.taskId));

      // pool[taskId].caller_wid = taskId (self)
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, taskIdVal,
                                          "caller_wid", ui16Ty),
                       taskIdVal);

      // pool[taskId].join_counter_ptr is left uninitialized — root tasks
      // don't have a join_counter (they're never the target of
      // completion_handler). Every @await_parallel site allocates its own
      // local join_counter and writes the address into its workers' pool
      // slots before activating them.

      // pool[taskId].stack_bottom = base of the fiber stack array (as u16)
      auto ui32Ty = IntegerType::get(ctx, 32, IntegerType::Unsigned);
      uint32_t rootU32Words = source().getRootStackU32Words(binding.funcOp);
      std::string stackName =
          ("__fiber_stack_" + binding.funcOp.getSymName()).str();
      auto stackArrayTy = csl::ArrayType::get({rootU32Words}, ui32Ty);
      Value stackU16Ptr = getGlobalAsU16ManyPtr(rewriter, loc, stackName,
                                                stackArrayTy, targetArch);
      Value stackBaseU16 =
          emitBitcastToU16(rewriter, loc, stackU16Ptr, targetArch);
      stackBaseU16 =
          emitHardwareWordAddress(rewriter, loc, stackBaseU16, targetArch);
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, taskIdVal,
                                          "stack_bottom", ui16Ty),
                       stackBaseU16);
    }

    ReturnOp::create(rewriter, loc, /*numBreakingRegions=*/1);
  }

  //===--------------------------------------------------------------------===//
  // Phase 3: Fuse sequential @await
  //===--------------------------------------------------------------------===//

  LogicalResult fuseSequentialAwaits(csl::CompilationUnitOp unit,
                                     IRRewriter &rewriter) {
    SmallVector<AwaitOp> awaitOps;
    unit.walk([&](AwaitOp op) { awaitOps.push_back(op); });

    for (AwaitOp awaitOp : awaitOps) {
      Value future = awaitOp.getFuture();
      auto futureTy = mlir::cast<csl::FutureType>(future.getType());

      auto callOp = future.getDefiningOp<CallOp>();
      if (!callOp) {
        awaitOp.emitError("@await must be applied directly to an async "
                          "call; await the call where it is made, e.g. "
                          "`@await(f(x))`, instead of passing on its result");
        return failure();
      }

      if (!future.hasOneUse()) {
        awaitOp.emitError("the result of an async call is used more than "
                          "once; await each async call exactly once, "
                          "directly where it is made");
        return failure();
      }

      StringRef calleeName = futureTy.getSymbolName().getValue();

      // Fuse call + await into a direct call (callee runs on caller's task).
      rewriter.setInsertionPoint(callOp);

      Type resultTy;
      if (awaitOp.getResult())
        resultTy = awaitOp.getResult().getType();

      auto *ctx = rewriter.getContext();
      auto newCall = CallOp::create(
          rewriter, awaitOp.getLoc(), resultTy,
          FlatSymbolRefAttr::get(ctx, calleeName), callOp.getArgOperands(),
          /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());

      if (awaitOp.getResult())
        rewriter.replaceAllUsesWith(awaitOp.getResult(), newCall.getResult());

      rewriter.eraseOp(awaitOp);
      rewriter.eraseOp(callOp);
    }

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Phase 4: Lower @await_parallel
  //===--------------------------------------------------------------------===//

  LogicalResult lowerAwaitParallel(csl::CompilationUnitOp unit,
                                   IRRewriter &rewriter) {
    SmallVector<AwaitParallelOp> apOps;
    unit.walk([&](AwaitParallelOp op) { apOps.push_back(op); });

    for (AwaitParallelOp apOp : apOps) {
      if (failed(lowerSingleAwaitParallel(unit, rewriter, apOp)))
        return failure();
    }
    return success();
  }

  LogicalResult lowerSingleAwaitParallel(csl::CompilationUnitOp unit,
                                         IRRewriter &rewriter,
                                         AwaitParallelOp apOp) {
    auto ctx = unit.getContext();
    Location loc = apOp.getLoc();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto localTaskIdTy = csl::LocalTaskIdType::get(ctx);

    unsigned numChildren = apOp.getFutures().size();
    if (numChildren == 0)
      return apOp.emitError("@await_parallel with no callees");

    // Resolve every future to its defining CallOp + callee name + FuncOp
    // up front; each callee is either run sync (i == 0) or as a worker
    // (i > 0).  The FuncOp lookup is needed for the FiberSizeSource queries
    // below and (in the worker setup loop) for pool[wid].pc.
    SmallVector<CallOp> definingCalls;
    SmallVector<StringRef> calleeNames;
    SmallVector<FuncOp> calleeFuncs;
    definingCalls.reserve(numChildren);
    calleeNames.reserve(numChildren);
    calleeFuncs.reserve(numChildren);
    for (Value future : apOp.getFutures()) {
      auto futureTy = mlir::cast<csl::FutureType>(future.getType());
      auto callOp = future.getDefiningOp<CallOp>();
      if (!callOp)
        return apOp.emitError("parallel future not produced by a direct call");
      StringRef name = futureTy.getSymbolName().getValue();
      FuncOp fn = unit.lookupSymbol<FuncOp>(name);
      if (!fn)
        return apOp.emitError("cannot find callee function '") << name << "'";
      definingCalls.push_back(callOp);
      calleeNames.push_back(name);
      calleeFuncs.push_back(fn);
    }

    // The first callee runs synchronously on the caller's task — emit it
    // last (after worker setup + activation) so workers run concurrently
    // with it. We resolve g_1's actual result type from its FutureType:
    // FutureType::getResultType() is the post-await source-level return
    // type (nullopt / void → no result).
    CallOp firstCall = definingCalls.front();
    StringRef firstCalleeName = calleeNames.front();
    auto firstFutureTy =
        mlir::cast<csl::FutureType>(apOp.getFutures().front().getType());
    SmallVector<Value> firstCallArgs(firstCall.getArgOperands().begin(),
                                     firstCall.getArgOperands().end());

    rewriter.setInsertionPoint(apOp);

    // Single-callee shortcut: just emit the sync call. No worker, no join,
    // no suspend.
    if (numChildren == 1) {
      Type resultTy = firstFutureTy.getResultType();
      if (resultTy && mlir::isa<csl::VoidType>(resultTy))
        resultTy = {};
      auto firstCallee = CallOp::create(
          rewriter, loc, resultTy ? TypeRange{resultTy} : TypeRange{},
          FlatSymbolRefAttr::get(ctx, firstCalleeName), firstCallArgs,
          /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());
      replaceAwaitParallelResult(
          rewriter, loc, apOp,
          resultTy ? firstCallee->getResult(0) : Value());
      rewriter.eraseOp(apOp);
      for (CallOp definingCall : definingCalls)
        rewriter.eraseOp(definingCall);
      return success();
    }

    // ====== Multi-callee path (numChildren >= 2) ======

    // Get caller's task ID from hardware.
    Value callerTaskId = emitGetCurrentTaskId(rewriter, loc);

    // Build the worker stack chain.
    //
    //   sp_at_site   = current SP
    //   top[1]       = sp_at_site - max(contrib(g0), contrib(caller_join))
    //   top[i+1]     = top[i] - <per-callee u16 word count>
    //
    // Callee 0 runs synchronously on the caller's task — its frame lives
    // between sp_at_site and top[1].  The caller's post-site frames
    // (caller_join + the suspend it tail-calls) occupy the same region
    // AFTER g0 returns, so g0's slot must be at least as deep as
    // contrib(caller_join) — the two are never live at the same time,
    // hence max() instead of a sum.  contrib(caller_join) is measured
    // from the round-1 ELF like every other callee; when cslc inlines
    // caller_join + suspend into the awaiting function (the common case)
    // their frames land above sp_at_site and contrib measures ~0.
    // Workers 0..numWorkers-1 handle callees 1..numChildren-1.  Worker w
    // (0-indexed) gets stack region [top[w+2], top[w+1]], with arg-area
    // at top[w+1] - kArgAreaSlots.
    //
    // Per-callee sizes come from the FiberSizeSource.  Pass 1 sees
    // placeholders (or user-set `stack_size(N)`); pass 2 sees measured
    // values from the round-1 ELF.  The same FuncOp's value is returned
    // by both sources when a `stack_size(N)` override is present.  Query
    // each size exactly once and stash so the per-site debug dump below
    // doesn't trigger extra getCallee* queries.
    SmallVector<uint16_t> contribWords;
    contribWords.reserve(numChildren);
    for (FuncOp fn : calleeFuncs)
      contribWords.push_back(source().getCalleeStackU16Words(fn));

    // Post-site reserve: the measured stack reach of the runtime
    // `caller_join` (which tail-calls `suspend`).  Queried through the
    // same FiberSizeSource so round 1 sees a placeholder and round 2 the
    // value resolved from the round-1 ELF.
    assert(rtCallerJoin && "caller_join not found in runtime");
    uint16_t postSiteReserveWords =
        source().getCalleeStackU16Words(rtCallerJoin);

    // Per-callee "chain word count" — what the chain math subtracts from
    // the running SP for this callee.  The sync callee g0 just contributes
    // its own contrib (it runs at sp_at_site and its frame fills the area
    // below sp).  Each WORKER additionally needs kArgAreaSlots above its
    // frame: worker.sp = workerTop - kArgAreaSlots, and the worker's
    // frame chain goes below that down to (workerTop - kArgAreaSlots -
    // contrib).  Without the +kArgAreaSlots here, each worker's frame
    // overflows into the next worker's region by 12 B and silently
    // corrupts neighbors' stacks (manifests as NaN output, kernel stall,
    // or worse).
    SmallVector<uint16_t> chainWords;
    chainWords.reserve(numChildren);
    chainWords.push_back(
        std::max(contribWords[0], postSiteReserveWords)); // sync callee
    for (unsigned i = 1; i < numChildren; ++i)
      chainWords.push_back(contribWords[i] + kArgAreaSlots);

    LLVM_DEBUG({
      llvm::dbgs() << "fiber-lowering: @await_parallel at " << loc << "\n"
                   << "  " << numChildren << " callees:\n";
      llvm::dbgs() << "    [0] SYNC  (runs on caller's task) : "
                   << calleeNames[0] << " — max(contrib " << contribWords[0]
                   << ", caller_join " << postSiteReserveWords
                   << ") = " << chainWords[0] << " u16-words ("
                   << (chainWords[0] * 2) << " B)\n";
      for (unsigned i = 1; i < numChildren; ++i)
        llvm::dbgs() << "    [" << i << "] WORKER " << (i - 1) << "       : "
                     << calleeNames[i] << " — contrib " << contribWords[i]
                     << " + " << kArgAreaSlots
                     << " arg-area = " << chainWords[i] << " u16-words ("
                     << (chainWords[i] * 2) << " B)\n";
      uint32_t totalU16Words = 0;
      for (uint16_t w : chainWords)
        totalU16Words += w;
      llvm::dbgs() << "  total stack reach below SP at site: "
                   << (totalU16Words * 2) << " B\n";
      llvm::dbgs() << "  partition layout (subtract each row from running "
                      "u16 SP):\n";
      llvm::dbgs() << "    sp_at_site\n"
                   << "      - " << chainWords[0] << " (" << calleeNames[0]
                   << " SYNC frame, ≥ caller_join reserve) → top[1]\n";
      for (unsigned i = 1; i < numChildren; ++i)
        llvm::dbgs() << "      - " << chainWords[i] << " (" << calleeNames[i]
                     << " worker " << (i - 1) << " incl. arg-area) → top["
                     << (i + 1) << "]\n";
      llvm::dbgs() << "  worker arg-area: at top[i] - " << kArgAreaSlots
                   << " (kArgAreaSlots)\n";
    });

    auto literalSize = [&](unsigned i) -> Value {
      return ConstantOp::create(rewriter, loc,
                                IntegerAttr::get(ui16Ty, chainWords[i]));
    };

    Value spAtSite = emitReadSP(rewriter, loc, targetArch);

    // Account for the sync callee's stack reach (callees[0]), already
    // max()ed with the post-site reserve in chainWords[0].  After this,
    // runningTop = top[1] = workerTop[1] (top of worker 0's region).
    Value runningTop =
        BinaryOp::create(rewriter, loc, ui16Ty, csl::BinaryOpKind::Sub,
                         spAtSite, literalSize(0));

    // Pre-compute the completion handler address (shared across all workers).
    assert(rtCompletionHandler && "completion_handler not found in runtime");
    assert(rtSuspend && "suspend not found in runtime");
    assert(rtCallerJoin && "caller_join not found in runtime");

    // Allocate a local join_counter for this @await_parallel site. The
    // counter lives on the caller's stack frame so that nested
    // await_parallels on the same task (which happens when our sync first
    // callee itself does an await_parallel) don't share storage with us.
    // Initialised to numChildren; decremented once per worker (in
    // completion_handler) and once by the caller (in caller_join).
    auto joinLvalTy = csl::LValueType::get(ui16Ty, /*isConst=*/false);
    static std::atomic<unsigned> joinUidCounter{0};
    std::string joinName = "__join_counter_" + std::to_string(joinUidCounter++);
    Value joinLval =
        VariableOp::create(rewriter, loc, joinLvalTy, joinName, Value())
            .getResult();
    Value numChildrenVal = ConstantOp::create(
        rewriter, loc, IntegerAttr::get(ui16Ty, numChildren));
    AssignOp::create(rewriter, loc, joinLval, numChildrenVal);
    auto u16PtrTy = csl::PointerType::get(ui16Ty, /*isConst=*/false,
                                          /*isArray=*/false);
    Value joinPtr =
        AddressOfOp::create(rewriter, loc, u16PtrTy, joinLval).getResult();
    auto completionRef = csl::FuncRefAttr::get(
        ctx, /*module_identifier=*/std::nullopt,
        FlatSymbolRefAttr::get(ctx, rtCompletionHandler.getSymName()),
        FunctionType::get(ctx, {}, {}));
    auto completionAddr = csl::FuncAddrAttr::get(completionRef);
    Value completionVal = ConstantOp::create(rewriter, loc, completionAddr);
    Value completionU16 =
        emitBitcastToU16(rewriter, loc, completionVal, targetArch);
    completionU16 =
        emitHardwareWordAddress(rewriter, loc, completionU16, targetArch);

    // Worker entry function reference for runtime @bind_local_task.
    auto voidFnType = FunctionType::get(ctx, {}, {});
    Value workerEntryRef =
        createFuncRef(rewriter, loc, voidFnType, rtWorkerEntry.getSymName());

    // Collect worker IDs for activation after setup.
    SmallVector<Value> workerTaskIds;

    // Workers cover callees i = 1..n-1.
    for (unsigned i = 1; i < numChildren; ++i) {
      CallOp callOp = definingCalls[i];
      StringRef calleeName = calleeNames[i];
      FuncOp calleeFunc = calleeFuncs[i];

      // Dynamically claim a worker task ID.
      Value workerTaskId =
          BuiltinCallOp::create(rewriter, loc, localTaskIdTy,
                                csl::BuiltinKind::ClaimLocalTask, ValueRange{})
              .getResult();
      Value workerIdVal =
          BuiltinCallOp::create(rewriter, loc, ui16Ty, csl::BuiltinKind::GetInt,
                                ValueRange{workerTaskId})
              .getResult();

      // Bind worker_entry to this dynamically claimed task ID.
      BuiltinCallOp::create(rewriter, loc, Type(),
                            csl::BuiltinKind::BindLocalTask,
                            ValueRange{workerEntryRef, workerTaskId});

      workerTaskIds.push_back(workerTaskId);

      // Worker w handles callees[i] (with i = w + 1).
      //
      //   workerTop[w]    = runningTop coming in     (top of region, = SP)
      //   workerBottom[w] = workerTop[w] - sizeOf(callees[i])
      //                     = runningTop after subtracting this callee's size
      //
      // arg-area sits just below workerTop; pool.sp points at it;
      // pool.stack_bottom marks workerBottom for any nested await_parallel
      // inside callees[i].
      Value workerTop = runningTop;
      Value calleeSize = literalSize(i);
      Value workerBottom = BinaryOp::create(
          rewriter, loc, ui16Ty, csl::BinaryOpKind::Sub, workerTop, calleeSize);
      runningTop = workerBottom;

      Value argAreaSlots = ConstantOp::create(
          rewriter, loc, IntegerAttr::get(ui16Ty, kArgAreaSlots));
      Value argAreaBase =
          BinaryOp::create(rewriter, loc, ui16Ty, csl::BinaryOpKind::Sub,
                           workerTop, argAreaSlots);

      // Convert the raw u16 argAreaBase to a [*]u16 pointer for typed writes.
      Value argAreaPtr =
          emitU16ToManyPtr(rewriter, loc, argAreaBase, targetArch);

      // Write args to arg area at register positions using ABI mapping.
      auto emitArgsTooLarge = [&]() {
        return apOp.emitError("the arguments of @await_parallel callee '")
               << callOp.getCalleeAttr().getRootReference().getValue()
               << "' do not fit into its four 16-bit argument registers (a "
                  "32-bit argument takes two, larger arguments are not "
                  "supported); pass fewer arguments, or a pointer to a "
                  "struct holding them";
      };
      ArgRegisterAllocator registers;
      for (Value arg : callOp.getArgOperands()) {
        if (failed(writeArgToArgArea(rewriter, loc, argAreaPtr, arg,
                                     registers)))
          return emitArgsTooLarge();
      }

      // Write completion handler address to r15 slot (slot 4).
      Value r15OffsetVal =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, 4));
      storeToU16Ptr(rewriter, loc, argAreaPtr, r15OffsetVal, completionU16);

      // pool[wid].sp is already in the runtime's hardware word-address
      // representation; argAreaPtr is used only for typed writes above.
      Value argAreaSp = argAreaBase;
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, workerIdVal,
                                          "sp", pool.spFieldTy()),
                       argAreaSp);

      // pool[wid].stack_bottom = workerBottom (for nested await_parallel)
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, workerIdVal,
                                          "stack_bottom", ui16Ty),
                       workerBottom);

      // pool[wid].pc = &callee (ptrcast to pc field type)
      auto funcRefAttr =
          csl::FuncRefAttr::get(ctx, /*module_identifier=*/std::nullopt,
                                FlatSymbolRefAttr::get(ctx, calleeName),
                                calleeFunc.getFunctionType());
      auto funcAddr = csl::FuncAddrAttr::get(funcRefAttr);
      Value funcAddrVal = ConstantOp::create(rewriter, loc, funcAddr);
      Value funcByteAddress =
          emitBitcastToU16(rewriter, loc, funcAddrVal, targetArch);
      Value funcPc =
          emitHardwareWordAddress(rewriter, loc, funcByteAddress, targetArch);
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, workerIdVal,
                                          "pc", pool.pcFieldTy()),
                       funcPc);

      // pool[wid].caller_wid = caller_task_id
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, workerIdVal,
                                          "caller_wid", ui16Ty),
                       callerTaskId);

      // pool[wid].join_counter_ptr = &local_join_counter
      AssignOp::create(rewriter, loc,
                       getPoolFieldLValue(rewriter, loc, pool, workerIdVal,
                                          "join_counter_ptr",
                                          pool.joinCounterPtrFieldTy()),
                       joinPtr);
    }

    // Activate all workers — they may run concurrently with the sync call
    // below at any suspend points the sync call hits.
    for (Value workerTaskId : workerTaskIds) {
      BuiltinCallOp::create(rewriter, loc, Type(), csl::BuiltinKind::Activate,
                            ValueRange{workerTaskId});
    }

    // Now run g_1 synchronously on the caller's task.
    Value firstResult;
    {
      Type resultTy = firstFutureTy.getResultType();
      if (resultTy && mlir::isa<csl::VoidType>(resultTy))
        resultTy = {};
      auto firstCallee = CallOp::create(
          rewriter, loc, resultTy ? TypeRange{resultTy} : TypeRange{},
          FlatSymbolRefAttr::get(ctx, firstCalleeName), firstCallArgs,
          /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());
      // Bind the result to a variable, so the call stays here, before the
      // join: the emitter would otherwise inline it at the use after the
      // join and run it after the workers instead of concurrently.
      if (resultTy && apOp.getResult()) {
        static std::atomic<unsigned> firstResultUid{0};
        Value firstResultVar = VariableOp::create(
            rewriter, loc, csl::LValueType::get(resultTy, /*isConst=*/true),
            "__await_parallel_first_result_" +
                std::to_string(firstResultUid++),
            firstCallee->getResult(0), /*is_const=*/true);
        firstResult = firstResultVar;
      }
    }

    // Caller-side join: decrement join_counter; if any workers are still
    // outstanding, suspend until the last one activates us.
    Value spFieldPtr = getPoolFieldPtr(rewriter, loc, pool, callerTaskId, "sp",
                                       pool.spFieldTy());
    Value pcFieldPtr = getPoolFieldPtr(rewriter, loc, pool, callerTaskId, "pc",
                                       pool.pcFieldTy());
    CallOp::create(rewriter, loc, TypeRange{},
                   FlatSymbolRefAttr::get(ctx, rtCallerJoin.getSymName()),
                   ValueRange{spFieldPtr, pcFieldPtr, joinPtr},
                   /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());

    // All callees have finished: collect their results.
    replaceAwaitParallelResult(rewriter, loc, apOp, firstResult);

    // Erase the await_parallel and all the original async CallOps.
    rewriter.eraseOp(apOp);
    for (CallOp definingCall : definingCalls)
      rewriter.eraseOp(definingCall);

    return success();
  }

  /// Write a single argument value to the slot of the fiber stack arg area
  /// that `registers` assigns to it. argAreaPtr is a [*]u16 pointer to the
  /// arg area. Returns failure if the arg doesn't fit into the registers.
  LogicalResult writeArgToArgArea(IRRewriter &rewriter, Location loc,
                                  Value argAreaPtr, Value arg,
                                  ArgRegisterAllocator &registers) {
    auto ctx = rewriter.getContext();
    auto ui16Ty = IntegerType::get(ctx, 16, IntegerType::Unsigned);

    Type argTy = arg.getType();
    auto sizeOpt = csl::getSizeInBytes(argTy);
    assert(sizeOpt && "arg type must have a known size");
    uint32_t sizeBytes = *sizeOpt;
    std::optional<unsigned> slot = registers.allocate(sizeBytes);
    if (!slot)
      return failure();

    if (sizeBytes <= 2) {
      // 16-bit or smaller arg: single register slot. Cast to u16 since
      // the arg area is [*]u16 and the arg may be i16, bool, or a pointer.
      Value storeVal = arg;
      if (arg.getType() != ui16Ty) {
        if (mlir::isa<IntegerType>(arg.getType()))
          storeVal = CastOp::create(rewriter, loc, ui16Ty, arg).getResult();
        else
          storeVal = emitBitcastToU16(rewriter, loc, arg, targetArch);
      }
      Value regOffsetVal =
          ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, *slot));
      storeToU16Ptr(rewriter, loc, argAreaPtr, regOffsetVal, storeVal);
      return success();
    }

    // 32-bit arg: an even-aligned register pair. Ptrcast the [*]u16 arg area
    // to a many-pointer to the arg's own 4-byte type (e.g. i32, u32, f32) and
    // store at the pair's 4-byte offset.
    auto argManyPtrTy =
        csl::PointerType::get(argTy, /*isConst=*/false, /*isArray=*/true);
    Value argArea32 =
        CastOp::create(rewriter, loc, argManyPtrTy, argAreaPtr).getResult();
    Value pairOffsetVal =
        ConstantOp::create(rewriter, loc, IntegerAttr::get(ui16Ty, *slot / 2));
    auto argLvalTy = csl::LValueType::get(argTy, /*isConst=*/false);
    Value lval = PtrIndexAccessOp::create(rewriter, loc, argLvalTy, argArea32,
                                          pairOffsetVal);
    AssignOp::create(rewriter, loc, lval, arg);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Phase 5: Lower SuspendOp
  //===--------------------------------------------------------------------===//

  LogicalResult lowerSuspendOps(csl::CompilationUnitOp unit,
                                IRRewriter &rewriter) {
    SmallVector<SuspendOp> suspendOps;
    unit.walk([&](SuspendOp op) { suspendOps.push_back(op); });

    auto ctx = unit.getContext();

    for (SuspendOp suspendOp : suspendOps) {
      rewriter.setInsertionPoint(suspendOp);
      Value taskId = emitGetCurrentTaskId(rewriter, suspendOp.getLoc());

      Value spFieldPtr = getPoolFieldPtr(rewriter, suspendOp.getLoc(), pool,
                                         taskId, "sp", pool.spFieldTy());
      Value pcFieldPtr = getPoolFieldPtr(rewriter, suspendOp.getLoc(), pool,
                                         taskId, "pc", pool.pcFieldTy());
      CallOp::create(rewriter, suspendOp.getLoc(), TypeRange{},
                     FlatSymbolRefAttr::get(ctx, rtSuspend.getSymName()),
                     ValueRange{spFieldPtr, pcFieldPtr},
                     /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr());

      rewriter.eraseOp(suspendOp);
    }

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Phase 7: Cleanup
  //===--------------------------------------------------------------------===//

  void cleanupMarkers(csl::CompilationUnitOp unit, IRRewriter &rewriter) {
    unit.walk([&](FuncOp funcOp) {
      if (funcOp.getIsAsync()) {
        funcOp.setIsAsync(false);
        funcOp.setIsTask(false);
        funcOp->removeAttr("coro.wip");
        funcOp->removeAttr("async_result_type");
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::typed_csl::createFiberLoweringPass(
    const FiberLoweringPassOptions &options, FiberSizeSource *sizeSource) {
  return std::make_unique<FiberLoweringPass>(options, sizeSource);
}
