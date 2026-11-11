//===- EmitCSL.cpp - Emit CSL from TypedCSL IR ----------------------------===//
//
// Implementation of CSL source code emission from TypedCSL IR.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Target/CSL/EmitCSL/EmitCSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/NominalTypeRegistry.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "emit-csl"

using namespace mlir;
using namespace mlir::csl;

namespace {

//===----------------------------------------------------------------------===//
// Precedence
//===----------------------------------------------------------------------===//

enum class Precedence : int {
  Lowest = 0,
  LogicalOr = 2,
  LogicalAnd = 3,
  Comparison = 4,
  // CSL (like early Zig) treats all bitwise operators at the same precedence
  // level with left-to-right associativity. Mixing & ^ | without parentheses
  // is parsed left-to-right, so we must use equal precedence here to ensure
  // the emitter adds parentheses wherever needed (e.g. the RHS of `|` when
  // it is an `&` sub-expression).
  BitwiseOr = 5,
  BitwiseXor = 5,
  BitwiseAnd = 5,
  Shift = 8,
  Additive = 9,
  Multiplicative = 10,
  Unary = 11,
  Primary = 13
};

static Precedence getPrecedence(BinaryOpKind op) {
  switch (op) {
  case BinaryOpKind::Add:
  case BinaryOpKind::Sub:
    return Precedence::Additive;
  case BinaryOpKind::Mul:
  case BinaryOpKind::Div:
  case BinaryOpKind::Mod:
    return Precedence::Multiplicative;
  case BinaryOpKind::BitAnd:
    return Precedence::BitwiseAnd;
  case BinaryOpKind::BitOr:
    return Precedence::BitwiseOr;
  case BinaryOpKind::BitXor:
    return Precedence::BitwiseXor;
  case BinaryOpKind::Shl:
  case BinaryOpKind::Shr:
    return Precedence::Shift;
  case BinaryOpKind::Eq:
  case BinaryOpKind::Ne:
  case BinaryOpKind::Lt:
  case BinaryOpKind::Le:
  case BinaryOpKind::Gt:
  case BinaryOpKind::Ge:
    return Precedence::Comparison;
  case BinaryOpKind::LogicalAnd:
    return Precedence::LogicalAnd;
  case BinaryOpKind::LogicalOr:
    return Precedence::LogicalOr;
  }
  llvm_unreachable("unknown binary op");
}

static StringRef getBinaryOpSymbol(BinaryOpKind op) {
  switch (op) {
  case BinaryOpKind::Add:
    return "+";
  case BinaryOpKind::Sub:
    return "-";
  case BinaryOpKind::Mul:
    return "*";
  case BinaryOpKind::Div:
    return "/";
  case BinaryOpKind::Mod:
    return "%";
  case BinaryOpKind::BitAnd:
    return "&";
  case BinaryOpKind::BitOr:
    return "|";
  case BinaryOpKind::BitXor:
    return "^";
  case BinaryOpKind::Shl:
    return "<<";
  case BinaryOpKind::Shr:
    return ">>";
  case BinaryOpKind::Eq:
    return "==";
  case BinaryOpKind::Ne:
    return "!=";
  case BinaryOpKind::Lt:
    return "<";
  case BinaryOpKind::Le:
    return "<=";
  case BinaryOpKind::Gt:
    return ">";
  case BinaryOpKind::Ge:
    return ">=";
  case BinaryOpKind::LogicalAnd:
    return "and";
  case BinaryOpKind::LogicalOr:
    return "or";
  }
  llvm_unreachable("unknown binary op");
}

/// Returns compound operator symbol by reusing getBinaryOpSymbol and appending
/// "=". Returns nullopt for comparison/logical ops which don't have compound
/// forms.
static std::optional<std::string> getCompoundOpSymbol(BinaryOpKind op) {
  switch (op) {
  case BinaryOpKind::Eq:
  case BinaryOpKind::Ne:
  case BinaryOpKind::Lt:
  case BinaryOpKind::Le:
  case BinaryOpKind::Gt:
  case BinaryOpKind::Ge:
  case BinaryOpKind::LogicalAnd:
  case BinaryOpKind::LogicalOr:
    return std::nullopt;
  default:
    return (getBinaryOpSymbol(op).str() + "=");
  }
}

static StringRef getUnaryOpSymbol(UnaryOpKind op) {
  switch (op) {
  case UnaryOpKind::Neg:
    return "-";
  case UnaryOpKind::BitNot:
    return "~";
  case UnaryOpKind::LogicalNot:
    return "!";
  }
  llvm_unreachable("unknown unary op");
}

//===----------------------------------------------------------------------===//
// Effect Analysis
//===----------------------------------------------------------------------===//

/// Collect all memory effects from an operation, recursively if it has
/// the RecursiveMemoryEffects trait.
static void
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    iface.getEffects(effects);
  } else if (mlir::isa<typed_csl::VariableOp>(op)) {
    // A declaration only initializes its own new storage, which no earlier
    // op can access; its initializer is a separate op.
  } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    // An op that declares no effects, e.g. a call, may read and write any
    // memory.
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
  }

  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions()) {
      for (Operation &nested : region.getOps()) {
        collectEffects(&nested, effects);
      }
    }
  }
}

/// The op in `block` that contains `user` if `user` runs exactly once, in
/// place, whenever that op runs, i.e. if it is in `block` or nested in it
/// only through labeled blocks. Null otherwise, e.g. for a use in a loop body,
/// which may run several times.
static Operation *getOnceAncestorInBlock(Operation *user, Block &block) {
  Operation *curr = user;
  while (curr->getBlock() != &block) {
    Operation *parent = curr->getParentOp();
    if (!mlir::isa_and_present<typed_csl::LabeledBlockOp>(parent))
      return nullptr;
    curr = parent;
  }
  return curr;
}

/// Check if an operation has any memory effects.
static bool hasEffects(Operation *op) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  collectEffects(op, effects);
  return !effects.empty();
}

/// The named storage that `lvalue` designates (a part of): a local variable,
/// a parameter, or a global. Null if it is reached through a pointer, which
/// may point to any storage whose address is taken.
static const void *getStorageRoot(Value lvalue) {
  while (true) {
    if (auto arg = mlir::dyn_cast<BlockArgument>(lvalue))
      return arg.getAsOpaquePointer();
    Operation *def = lvalue.getDefiningOp();
    if (mlir::isa<typed_csl::VariableOp>(def))
      return def;
    if (auto constant = mlir::dyn_cast<typed_csl::ConstantOp>(def)) {
      if (auto ref = mlir::dyn_cast<GlobalRefAttr>(constant.getValue()))
        return ref.getSymbolName().getAsOpaquePointer();
      return nullptr;
    }
    if (auto access = mlir::dyn_cast<typed_csl::FieldAccessOp>(def))
      lvalue = access.getStructLvalue();
    else if (auto access = mlir::dyn_cast<typed_csl::TupleIndexAccessOp>(def))
      lvalue = access.getBase();
    else if (auto access = mlir::dyn_cast<typed_csl::ArrayIndexAccessOp>(def))
      lvalue = access.getBase();
    else if (auto cast = mlir::dyn_cast<typed_csl::ConstCastOp>(def))
      lvalue = cast->getOperand(0);
    else
      return nullptr;
  }
}

/// Whether two memory effects may access the same storage. Effects on
/// different named storage cannot; anything else is assumed to.
static bool mayAlias(const MemoryEffects::EffectInstance &a,
                     const MemoryEffects::EffectInstance &b) {
  Value aValue = a.getValue();
  Value bValue = b.getValue();
  if (!aValue || !bValue || !mlir::isa<LValueType>(aValue.getType()) ||
      !mlir::isa<LValueType>(bValue.getType()))
    return true;
  const void *aRoot = getStorageRoot(aValue);
  const void *bRoot = getStorageRoot(bValue);
  return !aRoot || !bRoot || aRoot == bRoot;
}

/// Check if two operations have conflicting memory effects.
/// Conflicts occur when:
/// - Same resource AND
/// - At least one is a write AND
/// - Values may alias (conservatively assume yes if not known)
static bool conflicts(Operation *a, Operation *b) {
  SmallVector<MemoryEffects::EffectInstance> aEffects, bEffects;
  collectEffects(a, aEffects);
  collectEffects(b, bEffects);

  for (const auto &ae : aEffects) {
    for (const auto &be : bEffects) {
      // Different resources don't conflict
      if (ae.getResource() != be.getResource())
        continue;

      // Read-Read doesn't conflict
      bool aIsRead = isa<MemoryEffects::Read>(ae.getEffect());
      bool bIsRead = isa<MemoryEffects::Read>(be.getEffect());
      if (aIsRead && bIsRead)
        continue;

      // At least one write on same resource - conflict, unless the two
      // accesses are to different named storage
      if (mayAlias(ae, be))
        return true;
    }
  }
  return false;
}

} // namespace

namespace {

//===----------------------------------------------------------------------===//
// CSLEmitter
//===----------------------------------------------------------------------===//

class CSLEmitter {
public:
  CSLEmitter(raw_ostream &os, StringRef varPrefix, unsigned indentWidth)
      : os(os), varPrefix(varPrefix), indentWidth(indentWidth) {}

  LogicalResult emit(CompilationUnitOp op);

private:
  //===--------------------------------------------------------------------===//
  // Value Classification
  //===--------------------------------------------------------------------===//

  /// The symbol `op` refers to, if it is a function or global reference
  /// constant. Such a value is emitted as the bare symbol name.
  static std::optional<StringRef> getReferencedSymbol(Operation *op) {
    auto constant = dyn_cast_if_present<typed_csl::ConstantOp>(op);
    if (!constant)
      return std::nullopt;
    if (auto funcRef = dyn_cast<csl::FuncRefAttr>(constant.getValue()))
      return funcRef.getSymbolName().getValue();
    if (auto globalRef = dyn_cast<csl::GlobalRefAttr>(constant.getValue()))
      return globalRef.getSymbolName().getValue();
    return std::nullopt;
  }

  /// A "bound" value has an inherent name we emit directly.
  /// Block args, globals, funcs, variables, and materialized temps.
  bool isBound(Value value) const {
    // Block arguments are bound to parameter names
    if (isa<BlockArgument>(value))
      return true;

    Operation *def = value.getDefiningOp();
    if (!def)
      return true;

    // Symbol references are bound to their symbol names
    if (getReferencedSymbol(def))
      return true;

    // User variables are bound to their declared names
    if (isa<typed_csl::VariableOp>(def))
      return true;

    // Materialized values are bound to generated temp names
    if (materialized.contains(value))
      return true;

    return false;
  }

  /// Check if a constant can be emitted as a simple literal.
  bool isSimpleLiteral(typed_csl::ConstantOp op) const {
    TypedAttr val = op.getValue();
    // Simple numeric types and bools are literals
    if (isa<IntegerAttr, FloatAttr, ComptimeFloatAttr, ComptimeIntAttr,
            BoolAttr>(val))
      return true;
    // Type constants are simple
    if (isa<csl::TypeAttr>(val))
      return true;
    // Function and global references and addresses are simple
    if (isa<csl::FuncRefAttr, csl::FuncAddrAttr, csl::GlobalRefAttr,
            csl::GlobalAddrAttr>(val))
      return true;
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Materialization Analysis
  //===--------------------------------------------------------------------===//

  /// Compute which values need to be materialized as temporaries.
  void computeMaterialization(Block &block);

  /// Get the operands of an operation in emission order.
  /// This determines evaluation order when inlined.
  SmallVector<Value> getOperandsInEmissionOrder(Operation *op);

  /// Collect effectful operations reachable from a value in emission order.
  /// Stops at block boundaries — parent-block ops are already sequenced —
  /// and at values that are not `transparent`, which are emitted on their
  /// own instead of inside this expression.
  void collectEffectfulInEmissionOrder(Value value,
                                       SmallVectorImpl<Operation *> &result,
                                       Block *contextBlock,
                                       const DenseSet<Value> &transparent);

  //===--------------------------------------------------------------------===//
  // Name Management
  //===--------------------------------------------------------------------===//

  /// Get the bound name for a value.
  StringRef getName(Value value);

  /// Generate a fresh temporary name.
  std::string generateTempName() {
    return varPrefix + std::to_string(tempCounter++);
  }

  //===--------------------------------------------------------------------===//
  // Emission
  //===--------------------------------------------------------------------===//

  /// Emit a value as an expression (may be name, literal, or compound expr).
  LogicalResult emitExpr(Value value, Precedence ctx = Precedence::Lowest);

  /// Emit a region's contents. When isInline, emits statements without
  /// indentation/newlines, using "; " as separator between statements.
  LogicalResult emitRegion(Region &region, bool isInline = false);

  /// Emit a single operation.
  LogicalResult emitOp(Operation *op);

  /// Emit the expression for a defining operation.
  LogicalResult emitDefExpr(Operation *def, Precedence ctx);

  /// Emits a region that ends with a yield as an expression, returning the
  /// yield value.
  void emitRegionAsValue(Region &region);

  // Operation-specific expression emitters
  void emitConstantExpr(typed_csl::ConstantOp op);
  void emitBinaryExpr(typed_csl::BinaryOp op, Precedence ctx);
  void emitUnaryExpr(typed_csl::UnaryOp op);
  void emitCastExpr(typed_csl::CastOp op);
  void emitCallExpr(typed_csl::CallOp op);
  void emitIndirectCallExpr(typed_csl::IndirectCallOp op);
  void emitBuiltinCallExpr(typed_csl::BuiltinCallOp op);
  void emitIfExpr(typed_csl::IfOp op);
  void emitSwitchExpr(typed_csl::SwitchOp op);
  void emitLabeledBlockExpr(typed_csl::LabeledBlockOp op);
  void emitShortCircuitExpr(typed_csl::ShortCircuitOp op, Precedence ctx);
  void emitConstructStructExpr(typed_csl::ConstructStructOp op);
  void emitConstructTupleExpr(typed_csl::ConstructTupleOp op);
  void emitConstructArrayExpr(typed_csl::ConstructArrayOp op);
  void emitFieldAccessExpr(typed_csl::FieldAccessOp op);
  void emitArrayIndexAccessExpr(typed_csl::ArrayIndexAccessOp op);
  void emitTupleIndexAccessExpr(typed_csl::TupleIndexAccessOp op);
  void emitPtrIndexAccessExpr(typed_csl::PtrIndexAccessOp op);
  void emitAddressOfExpr(typed_csl::AddressOfOp op);
  void emitDereferenceExpr(typed_csl::DereferenceOp op);
  void emitLoadExpr(typed_csl::LoadOp op, Precedence ctx);

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  raw_ostream &os;
  std::string varPrefix;
  unsigned indentWidth;
  unsigned indent = 0;
  unsigned tempCounter = 0;

  /// When true, statement ops emit inline (no indentation/newlines, semicolons
  /// as compact terminators).
  bool inlineMode = false;

  /// Registry for nominal types (enums and named structs).
  NominalTypeRegistry typeRegistry;

  /// Values that must be materialized as temporaries.
  DenseSet<Value> materialized;

  /// Cached names for values.
  DenseMap<Value, std::string> valueNames;

  void emitIndent() {
    for (unsigned i = 0; i < indent * indentWidth; ++i)
      os << ' ';
  }

  /// Begin a statement: in normal mode, emit indentation; in inline mode,
  /// nothing.
  void beginStmt() {
    if (!inlineMode)
      emitIndent();
  }

  /// End a statement: in normal mode, emit ";\n"; in inline mode, "; ".
  void endStmt() {
    if (inlineMode)
      os << "; ";
    else
      os << ";\n";
  }

  /// Helper to print type using the nominal type registry.
  void printType(Type type) {
    assert(!csl::hasPerPELayout(type) &&
           "types with per-PE layout must be lowered before emission");
    csl::printType(os, type, &typeRegistry);
  }

  /// Helper to print value using the nominal type registry.
  /// Wraps runtime-typed integer (non-bool) and float literals in @as(type,
  /// ...) so that emitted CSL code preserves concrete types instead of
  /// producing bare comptime_int/comptime_float literals.
  void printValue(TypedAttr attr) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
      if (auto intType = mlir::dyn_cast<IntegerType>(intAttr.getType())) {
        if (intType.getWidth() != 1) {
          os << "@as(";
          printType(intType);
          os << ", ";
          if (intType.isSigned() || intType.isSignless())
            intAttr.getValue().print(os, /*isSigned=*/true);
          else
            intAttr.getValue().print(os, /*isSigned=*/false);
          os << ")";
          return;
        }
      }
    } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(attr)) {
      const llvm::APFloat &val = floatAttr.getValue();
      if (!val.isFinite()) {
        // Inf and NaN have no CSL literal syntax; emit as @bitcast from the
        // raw bit pattern, e.g. @bitcast(f32, @as(u32, 0x7f800000)).
        llvm::APInt bits = val.bitcastToAPInt();
        os << "@bitcast(";
        printType(floatAttr.getType());
        os << ", @as(u" << bits.getBitWidth() << ", 0x";
        llvm::SmallString<16> hex;
        bits.toStringUnsigned(hex, /*Radix=*/16);
        os << hex;
        os << "))";
        return;
      }
      os << "@as(";
      printType(floatAttr.getType());
      os << ", ";
      llvm::SmallString<32> str;
      val.toString(str);
      if (str.find('.') == llvm::StringRef::npos)
        str.append(".0");
      os << str;
      os << ")";
      return;
    }
    csl::printValue(os, attr, &typeRegistry);
  }
};

//===----------------------------------------------------------------------===//
// Materialization Analysis
//===----------------------------------------------------------------------===//

SmallVector<Value> CSLEmitter::getOperandsInEmissionOrder(Operation *op) {
  // Most operations emit operands left-to-right
  SmallVector<Value> result;
  for (Value operand : op->getOperands())
    result.push_back(operand);
  return result;
}

void CSLEmitter::collectEffectfulInEmissionOrder(
    Value value, SmallVectorImpl<Operation *> &result, Block *contextBlock,
    const DenseSet<Value> &transparent) {
  // Bound values don't contribute inlined effectful ops
  if (isBound(value))
    return;

  Operation *def = value.getDefiningOp();
  if (!def)
    return;

  // Values defined in a parent region are already sequenced before this
  // block executes — they don't participate in intra-block ordering.
  if (def->getBlock() != contextBlock)
    return;

  // A value that is not inlined is emitted at its own position.
  if (!transparent.contains(value))
    return;

  // Simple literals don't have effects
  if (auto constant = dyn_cast<typed_csl::ConstantOp>(def)) {
    if (isSimpleLiteral(constant))
      return;
  }

  // Recursively collect from operands in emission order
  for (Value operand : getOperandsInEmissionOrder(def)) {
    collectEffectfulInEmissionOrder(operand, result, contextBlock,
                                    transparent);
  }

  // Add this op if it has effects
  if (hasEffects(def))
    result.push_back(def);
}

void CSLEmitter::computeMaterialization(Block &block) {
  // Phase 1: Find structural candidates (single-use expressions)
  DenseSet<Value> transparent;

  for (Operation &op : block) {
    // Skip ops that don't produce values or are inherently bound
    if (op.getNumResults() == 0)
      continue;

    Value result = op.getResult(0);

    // Bound values are never transparent
    if (isa<typed_csl::VariableOp>(&op) || getReferencedSymbol(&op))
      continue;

    // Simple constants are always transparent (can repeat literals)
    if (auto constant = dyn_cast<typed_csl::ConstantOp>(&op)) {
      if (isSimpleLiteral(constant)) {
        transparent.insert(result);
        continue;
      }
    }

    // LValue-typed results must always be transparent because CSL has no
    // way to create a named alias for an lvalue expression
    if (isa<LValueType>(result.getType())) {
      transparent.insert(result);
      continue;
    }

    // AddressOfOp is also transparent as it improves readability of coroutine
    // generated code (single '&' vs materialized temp) and doesn't have any
    // effects.
    if (isa<typed_csl::AddressOfOp>(&op)) {
      transparent.insert(result);
      continue;
    }

    // Load of const lvalues are also transparent as they just emit the lvalue
    // expression directly
    if (auto load = dyn_cast<typed_csl::LoadOp>(&op)) {
      LValueType lvalueType = load.getLvalue().getType();
      if (lvalueType.getIsConst()) {
        transparent.insert(result);
        continue;
      }
    }

    // A load may be emitted again at each of its uses in this block: that
    // only reads the storage again, and the ordering rules below materialize
    // it if something may write the storage before a use.
    if (isa<typed_csl::LoadOp>(&op) && !result.use_empty() &&
        llvm::all_of(result.getUsers(), [&](Operation *user) {
          return getOnceAncestorInBlock(user, block) != nullptr;
        })) {
      transparent.insert(result);
      continue;
    }

    // Multiple uses require materialization
    if (!result.hasOneUse())
      continue;

    // If the single use is in a nested region that may run several times,
    // e.g. a loop body, the value must be materialized — inlining would
    // re-evaluate the expression there instead of once in this block. A
    // labeled block runs once, in place.
    if (!getOnceAncestorInBlock(*result.getUsers().begin(), block))
      continue;

    // Single-use expression in the same block - candidate for transparency
    transparent.insert(result);
  }

  // Workaround for cslc bugs: complex expressions as operands of ptr[index]
  // can cause "invalid runtime operand to '[]' operator" errors. Force both
  // the base and index operands of PtrIndexAccessOp to materialize as
  // temporaries unless they are already simple (bound name, literal, or
  // load of a block arg).
  auto shouldMaterializeForPtrAccess = [&](Value v) -> bool {
    if (!transparent.contains(v))
      return false;
    if (isBound(v))
      return false;
    if (auto constant = v.getDefiningOp<typed_csl::ConstantOp>())
      if (isSimpleLiteral(constant))
        return false;
    // A LoadOp from a block argument (e.g. function parameter) just emits
    // the parameter name — no complex expression.
    if (auto load = v.getDefiningOp<typed_csl::LoadOp>())
      if (isa<BlockArgument>(load.getLvalue()))
        return false;
    return true;
  };
  for (Operation &op : block) {
    auto ptrAccess = dyn_cast<typed_csl::PtrIndexAccessOp>(&op);
    if (!ptrAccess)
      continue;
    if (shouldMaterializeForPtrAccess(ptrAccess.getBase()))
      transparent.erase(ptrAccess.getBase());
    if (shouldMaterializeForPtrAccess(ptrAccess.getIndex()))
      transparent.erase(ptrAccess.getIndex());
  }

  // Phase 2: Remove candidates that violate ordering constraints
  bool changed = true;
  while (changed) {
    changed = false;

    for (Operation &op : block) {
      if (op.getNumResults() == 0)
        continue;

      Value result = op.getResult(0);
      if (!transparent.contains(result))
        continue;

      // Rule 1: Check for intervening non-transparent conflicts. Only loads
      // can be transparent with several uses; each use is checked.
      for (Operation *user : result.getUsers()) {
        if (!transparent.contains(result))
          break;

        // Walk from def to user checking for conflicts. A user nested in
        // labeled blocks runs at the position of the outermost one. Values
        // that are always inlined, e.g. constants, may also be used in other
        // nested regions; there the walk goes to the end of the block.
        Operation *anchor = getOnceAncestorInBlock(user, block);
        for (Operation *curr = op.getNextNode();
             curr != anchor && curr != nullptr; curr = curr->getNextNode()) {
          // If curr is emitted in place (a statement, or a materialized
          // value) and conflicts with op, we must materialize op
          if (curr->getNumResults() == 0 ||
              !transparent.contains(curr->getResult(0))) {
            if (conflicts(&op, curr)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Materializing due to intervening conflict: " << op
                         << "\n");
              transparent.erase(result);
              changed = true;
              break;
            }
          }
        }

        // Inside the labeled blocks, everything before the user also runs
        // between def and user.
        for (Operation *inner = user;
             anchor && inner != anchor && transparent.contains(result);
             inner = inner->getParentOp()) {
          for (Operation &curr : *inner->getBlock()) {
            if (&curr == inner)
              break;
            if (conflicts(&op, &curr)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Materializing due to conflict in labeled block: "
                         << op << "\n");
              transparent.erase(result);
              changed = true;
              break;
            }
          }
        }
      }

      if (!transparent.contains(result))
        continue;

      // Rule 2: Check expression tree order for each use of this op
      for (Operation *user : result.getUsers()) {

        // Collect effectful ops in emission order for the user
        SmallVector<Operation *> effectfulOps;
        for (Value operand : getOperandsInEmissionOrder(user)) {
          collectEffectfulInEmissionOrder(operand, effectfulOps, &block,
                                          transparent);
        }

        // Check that effectful ops are in IR order
        for (size_t i = 1; i < effectfulOps.size(); ++i) {
          Operation *prev = effectfulOps[i - 1];
          Operation *curr = effectfulOps[i];

          // If curr is before prev in IR but after in emission, and they
          // conflict
          if (curr->isBeforeInBlock(prev) && conflicts(prev, curr)) {
            // prev is out of place - defined later but emitted earlier
            if (transparent.contains(prev->getResult(0))) {
              LLVM_DEBUG(llvm::dbgs() << "Materializing due to emission order: "
                                      << *prev << "\n");
              transparent.erase(prev->getResult(0));
              changed = true;
            }
          }
        }
      }
    }
  }

  // Everything not transparent must be materialized
  for (Operation &op : block) {
    if (op.getNumResults() == 0)
      continue;
    Value result = op.getResult(0);

    // Skip inherently bound values
    if (isa<typed_csl::VariableOp>(&op) || getReferencedSymbol(&op))
      continue;

    if (!transparent.contains(result)) {
      materialized.insert(result);
    }
  }
}

//===----------------------------------------------------------------------===//
// Name Management
//===----------------------------------------------------------------------===//

StringRef CSLEmitter::getName(Value value) {
  auto it = valueNames.find(value);
  if (it != valueNames.end())
    return it->second;

  // Block arguments
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();

    // For loop variables
    if (auto forOp = dyn_cast<typed_csl::ForOp>(parentOp)) {
      if (blockArg.getArgNumber() == 0) {
        valueNames[value] = forOp.getElemName().str();
        return valueNames[value];
      }
      if (forOp.getIndexName()) {
        valueNames[value] = forOp.getIndexName()->str();
        return valueNames[value];
      }
    }

    // Function parameters and other block args: use generated names
    std::string name = "param" + std::to_string(blockArg.getArgNumber());
    valueNames[value] = std::move(name);
    return valueNames[value];
  }

  // Symbol references
  if (std::optional<StringRef> symbol =
          getReferencedSymbol(value.getDefiningOp())) {
    valueNames[value] = symbol->str();
    return valueNames[value];
  }

  // User variables
  if (auto var = value.getDefiningOp<typed_csl::VariableOp>()) {
    valueNames[value] = var.getName().str();
    return valueNames[value];
  }

  // Materialized temporaries
  if (materialized.contains(value)) {
    std::string name = generateTempName();
    LLVM_DEBUG(llvm::dbgs()
               << "getName: " << name << " for " << value << " defined by "
               << (value.getDefiningOp()
                       ? value.getDefiningOp()->getName().getStringRef()
                       : "block arg")
               << "\n");
    valueNames[value] = std::move(name);
    return valueNames[value];
  }

  llvm_unreachable("getName called on non-bound value");
}

//===----------------------------------------------------------------------===//
// Expression Emission
//===----------------------------------------------------------------------===//

LogicalResult CSLEmitter::emitExpr(Value value, Precedence ctx) {
  // Bound values: emit their name
  if (isBound(value)) {
    os << getName(value);
    return success();
  }

  // Not bound - emit the defining expression
  Operation *def = value.getDefiningOp();
  if (!def) {
    // Should not happen - block args are bound
    return failure();
  }

  return emitDefExpr(def, ctx);
}

LogicalResult CSLEmitter::emitDefExpr(Operation *def, Precedence ctx) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(def)
      .Case<typed_csl::ConstantOp>([&](auto op) {
        // GlobalAddrAttr and FuncAddrAttr emit as `&symbol` (Unary
        // precedence). Add parens when used in a higher-precedence context,
        // e.g. as the operand of `.*` or `[i]`.
        bool needParens =
            isa<csl::GlobalAddrAttr, csl::FuncAddrAttr>(op.getValue()) &&
            Precedence::Unary < ctx;
        if (needParens)
          os << "(";
        emitConstantExpr(op);
        if (needParens)
          os << ")";
        return success();
      })
      .Case<typed_csl::BinaryOp>([&](auto op) {
        emitBinaryExpr(op, ctx);
        return success();
      })
      .Case<typed_csl::UnaryOp>([&](auto op) {
        bool needParens = Precedence::Unary < ctx;
        if (needParens)
          os << "(";
        emitUnaryExpr(op);
        if (needParens)
          os << ")";
        return success();
      })
      .Case<typed_csl::CastOp>([&](auto op) {
        emitCastExpr(op);
        return success();
      })
      .Case<typed_csl::ConstCastOp>([&](auto op) {
        // const_cast is a no-op in CSL — just emit the source expression.
        return emitExpr(op.getSource(), ctx);
      })
      .Case<typed_csl::LoadOp>([&](auto op) {
        emitLoadExpr(op, ctx);
        return success();
      })
      .Case<typed_csl::CallOp>([&](auto op) {
        emitCallExpr(op);
        return success();
      })
      .Case<typed_csl::IndirectCallOp>([&](auto op) {
        emitIndirectCallExpr(op);
        return success();
      })
      .Case<typed_csl::BuiltinCallOp>([&](auto op) {
        emitBuiltinCallExpr(op);
        return success();
      })
      .Case<typed_csl::IfOp>([&](auto op) {
        bool needParens = Precedence::Lowest < ctx;
        if (needParens)
          os << "(";
        emitIfExpr(op);
        if (needParens)
          os << ")";
        return success();
      })
      .Case<typed_csl::SwitchOp>([&](auto op) {
        bool needParens = Precedence::Lowest < ctx;
        if (needParens)
          os << "(";
        emitSwitchExpr(op);
        if (needParens)
          os << ")";
        return success();
      })
      .Case<typed_csl::LabeledBlockOp>([&](auto op) {
        bool needParens = Precedence::Lowest < ctx;
        if (needParens)
          os << "(";
        emitLabeledBlockExpr(op);
        if (needParens)
          os << ")";
        return success();
      })
      .Case<typed_csl::ShortCircuitOp>([&](auto op) {
        emitShortCircuitExpr(op, ctx);
        return success();
      })
      .Case<typed_csl::ConstructStructOp>([&](auto op) {
        emitConstructStructExpr(op);
        return success();
      })
      .Case<typed_csl::ConstructTupleOp>([&](auto op) {
        emitConstructTupleExpr(op);
        return success();
      })
      .Case<typed_csl::ConstructArrayOp>([&](auto op) {
        emitConstructArrayExpr(op);
        return success();
      })
      .Case<typed_csl::FieldAccessOp>([&](auto op) {
        emitFieldAccessExpr(op);
        return success();
      })
      .Case<typed_csl::ArrayIndexAccessOp>([&](auto op) {
        emitArrayIndexAccessExpr(op);
        return success();
      })
      .Case<typed_csl::TupleIndexAccessOp>([&](auto op) {
        emitTupleIndexAccessExpr(op);
        return success();
      })
      .Case<typed_csl::PtrIndexAccessOp>([&](auto op) {
        emitPtrIndexAccessExpr(op);
        return success();
      })
      .Case<typed_csl::AddressOfOp>([&](auto op) {
        bool needParens = Precedence::Unary < ctx;
        if (needParens)
          os << "(";
        emitAddressOfExpr(op);
        if (needParens)
          os << ")";
        return success();
      })
      .Case<typed_csl::DereferenceOp>([&](auto op) {
        emitDereferenceExpr(op);
        return success();
      })
      .Case<typed_csl::AwaitOp>([&](auto op) {
        os << "await ";
        (void)emitExpr(op.getFuture());
        return success();
      })
      .Case<typed_csl::AwaitBrOp>([&](auto op) {
        return op.emitError("await_br should be lowered before emission");
      })
      .Default([&](Operation *op) {
        return op->emitError("cannot emit expression for: ") << op->getName();
      });
}

void CSLEmitter::emitConstantExpr(typed_csl::ConstantOp op) {
  printValue(op.getValue());
}

void CSLEmitter::emitBinaryExpr(typed_csl::BinaryOp op, Precedence ctx) {
  Precedence opPrec = getPrecedence(op.getOpcode());
  bool needParens = opPrec < ctx;

  if (needParens)
    os << "(";
  (void)emitExpr(op.getLhs(), opPrec);
  os << " " << getBinaryOpSymbol(op.getOpcode()) << " ";
  // Right side needs higher precedence to handle associativity
  (void)emitExpr(op.getRhs(),
                 static_cast<Precedence>(static_cast<int>(opPrec) + 1));
  if (needParens)
    os << ")";
}

void CSLEmitter::emitUnaryExpr(typed_csl::UnaryOp op) {
  os << getUnaryOpSymbol(op.getOpcode());
  (void)emitExpr(op.getOperand(), Precedence::Unary);
}

void CSLEmitter::emitCastExpr(typed_csl::CastOp op) {
  // Pointer-to-pointer casts use @ptrcast; everything else uses @as.
  bool isPtrCast = mlir::isa<csl::PointerType>(op.getSource().getType()) &&
                   mlir::isa<csl::PointerType>(op.getDest().getType());
  os << (isPtrCast ? "@ptrcast(" : "@as(");
  printType(op.getDest().getType());
  os << ", ";
  (void)emitExpr(op.getSource());
  os << ")";
}

void CSLEmitter::emitLoadExpr(typed_csl::LoadOp op, Precedence ctx) {
  (void)emitExpr(op.getLvalue(), ctx);
}

void CSLEmitter::emitCallExpr(typed_csl::CallOp op) {
  os << op.getCallee().getLeafReference().getValue() << "(";
  llvm::interleaveComma(op.getOperands(), os,
                        [&](Value v) { (void)emitExpr(v); });
  os << ")";
}

void CSLEmitter::emitIndirectCallExpr(typed_csl::IndirectCallOp op) {
  // Emit the callee expression (function pointer)
  (void)emitExpr(op.getCallee());
  os << "(";
  llvm::interleaveComma(op.getArgs(), os, [&](Value v) { (void)emitExpr(v); });
  os << ")";
}

void CSLEmitter::emitBuiltinCallExpr(typed_csl::BuiltinCallOp op) {
  os << "@" << stringifyBuiltinKind(op.getBuiltin()) << "(";
  llvm::interleaveComma(op.getArgs(), os, [&](Value v) { (void)emitExpr(v); });
  os << ")";
}

void CSLEmitter::emitIfExpr(typed_csl::IfOp op) {
  os << "if (";
  (void)emitExpr(op.getCondition());
  os << ") ";
  emitRegionAsValue(op.getThenRegion());
  os << " else ";
  emitRegionAsValue(op.getElseRegion());
}

void CSLEmitter::emitSwitchExpr(typed_csl::SwitchOp op) {
  os << "switch (";
  (void)emitExpr(op.getInput());
  os << ") {\n";
  ++indent;

  auto caseValues = op.getCaseValues();
  auto caseRegions = op.getCaseRegions();
  bool hasElse = op.hasElseBranch();

  for (size_t i = 0; i < caseRegions.size(); ++i) {
    emitIndent();
    auto thisCaseValues = mlir::cast<ArrayAttr>(caseValues[i]);
    llvm::interleaveComma(thisCaseValues, os, [&](Attribute val) {
      printValue(mlir::cast<TypedAttr>(val));
    });
    os << " => ";
    emitRegionAsValue(caseRegions[i]);
    // Emit comma unless this is the last case and there's no else
    if (i + 1 < caseRegions.size() || hasElse)
      os << ",";
    os << "\n";
  }

  if (hasElse) {
    emitIndent();
    os << "else => ";
    emitRegionAsValue(op.getElseRegion());
    os << "\n";
  }

  --indent;
  emitIndent();
  os << "}";
}

void CSLEmitter::emitLabeledBlockExpr(typed_csl::LabeledBlockOp op) {
  os << op.getLabel() << ": {\n";
  ++indent;
  (void)emitRegion(op.getBody());
  --indent;
  emitIndent();
  os << "}";
}

void CSLEmitter::emitShortCircuitExpr(typed_csl::ShortCircuitOp op,
                                      Precedence ctx) {
  Precedence opPrec = op.getKind() == ShortCircuitKind::And
                          ? Precedence::LogicalAnd
                          : Precedence::LogicalOr;
  bool needParens = opPrec < ctx;

  if (needParens)
    os << "(";
  (void)emitExpr(op.getLhs(), opPrec);
  os << (op.getKind() == ShortCircuitKind::And ? " and " : " or ");
  emitRegionAsValue(op.getRhsRegion());
  if (needParens)
    os << ")";
}

void CSLEmitter::emitConstructStructExpr(typed_csl::ConstructStructOp op) {
  // For named structs, print TypeName { ... } instead of .{ ... }
  auto structType = mlir::dyn_cast<StructType>(op.getResult().getType());
  if (structType && structType.getDefLocation()) {
    if (auto name = typeRegistry.getName(structType)) {
      os << *name << " ";
    }
  } else {
    os << ".";
  }
  os << "{ ";
  for (size_t i = 0; i < op.getFieldNames().size(); ++i) {
    if (i > 0)
      os << ", ";
    os << "." << cast<StringAttr>(op.getFieldNames()[i]).getValue() << " = ";
    (void)emitExpr(op.getFieldValues()[i]);
  }
  os << " }";
}

void CSLEmitter::emitConstructTupleExpr(typed_csl::ConstructTupleOp op) {
  if (op.getFieldValues().empty()) {
    os << ".{}";
    return;
  }
  os << ".{ ";
  llvm::interleaveComma(op.getFieldValues(), os,
                        [&](Value v) { (void)emitExpr(v); });
  os << " }";
}

void CSLEmitter::emitConstructArrayExpr(typed_csl::ConstructArrayOp op) {
  auto arrayType = mlir::cast<ArrayType>(op.getResult().getType());
  printType(arrayType);
  os << "{ ";
  llvm::interleaveComma(op.getElements(), os,
                        [&](Value v) { (void)emitExpr(v); });
  os << " }";
}

void CSLEmitter::emitFieldAccessExpr(typed_csl::FieldAccessOp op) {
  (void)emitExpr(op.getStructLvalue(), Precedence::Primary);
  os << "." << op.getFieldName();
}

void CSLEmitter::emitArrayIndexAccessExpr(typed_csl::ArrayIndexAccessOp op) {
  // Emit base in parentheses to ensure correct precedence. Likely a bug in
  // Cerebras' compiler.
  os << "(";
  (void)emitExpr(op.getBase(), Precedence::Primary);
  os << ")";
  os << "[";
  llvm::interleaveComma(op.getIndices(), os,
                        [&](Value idx) { (void)emitExpr(idx); });
  os << "]";
}

void CSLEmitter::emitTupleIndexAccessExpr(typed_csl::TupleIndexAccessOp op) {
  (void)emitExpr(op.getBase(), Precedence::Primary);
  os << "[" << op.getIndex() << "]";
}

void CSLEmitter::emitPtrIndexAccessExpr(typed_csl::PtrIndexAccessOp op) {
  (void)emitExpr(op.getBase(), Precedence::Primary);
  os << "[";
  (void)emitExpr(op.getIndex());
  os << "]";
}

void CSLEmitter::emitAddressOfExpr(typed_csl::AddressOfOp op) {
  os << "&";
  (void)emitExpr(op.getOperand(), Precedence::Unary);
}

void CSLEmitter::emitDereferenceExpr(typed_csl::DereferenceOp op) {
  (void)emitExpr(op.getOperand(), Precedence::Primary);
  os << ".*";
}

void CSLEmitter::emitRegionAsValue(Region &region) {
  Block &block = region.front();

  // Check for CondYieldOp (WhileOp condition region) or YieldOp.
  Value yieldValue;
  typed_csl::YieldOp yieldOp;
  if (!block.empty() && block.mightHaveTerminator()) {
    if (auto condYield =
            mlir::dyn_cast<typed_csl::CondYieldOp>(block.getTerminator())) {
      yieldValue = condYield.getCondition();
    } else if ((yieldOp = mlir::dyn_cast<typed_csl::YieldOp>(
                    block.getTerminator()))) {
      yieldValue = yieldOp.getValue();
    }
  }

  if (!yieldValue) {
    // No yield — emit as unlabeled block
    os << "{\n";
    ++indent;
    (void)emitRegion(region);
    --indent;
    emitIndent();
    os << "}";
    return;
  }

  // Check whether any op in the block would emit a statement.
  // If not, we can emit just the yield value as an inline expression
  // without wrapping in a labeled block.
  computeMaterialization(block);
  bool needsBlock = false;
  for (Operation &op : block) {
    if (isa<typed_csl::YieldOp, typed_csl::CondYieldOp>(&op))
      continue;
    // Ops with results that are not materialized (and not VariableOp)
    // are transparent — they emit nothing and are inlined at use sites.
    if (op.getNumResults() > 0 && !isa<typed_csl::VariableOp>(&op) &&
        !materialized.contains(op.getResult(0)))
      continue;
    needsBlock = true;
    break;
  }

  if (!needsBlock) {
    // Simple case: all ops are transparent, just emit the yield value
    (void)emitExpr(yieldValue);
    return;
  }

  // Complex case: emit a labeled block with break
  static std::atomic<unsigned> branchCounter{0};
  std::string branchName = "valueBlock" + std::to_string(branchCounter++);
  os << branchName << ": {\n";
  ++indent;
  (void)emitRegion(region);
  emitIndent();
  os << "break :" << branchName << " ";
  (void)emitExpr(yieldValue);
  os << ";\n";
  --indent;
  emitIndent();
  os << "}";
}

//===----------------------------------------------------------------------===//
// Statement/Operation Emission
//===----------------------------------------------------------------------===//

LogicalResult CSLEmitter::emitRegion(Region &region, bool isInline) {
  // Save and restore inline mode state.
  bool savedInlineMode = inlineMode;
  inlineMode = isInline;
  llvm::scope_exit restore([&] { inlineMode = savedInlineMode; });

  for (Block &block : region) {
    // Compute materialization for this block
    computeMaterialization(block);

    for (Operation &op : block) {
      if (failed(emitOp(&op)))
        return failure();
    }
  }
  return success();
}

LogicalResult CSLEmitter::emitOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      // Globals
      .Case<typed_csl::GlobalOp>([&](auto global) {
        emitIndent();
        if (global.getIsExport())
          os << "export ";
        if (global.getIsExtern())
          os << "extern ";
        os << (global.getIsConst() ? "const " : "var ");
        os << global.getSymName() << ": ";
        printType(global.getType());
        if (auto linkname = global.getLinkname())
          os << " linkname(\"" << *linkname << "\")";
        if (auto alignment = global.getAlignment())
          os << " align(" << *alignment << ")";
        if (auto init = global.getInitializer()) {
          os << " = ";
          printValue(*init);
        }
        os << ";\n";
        return success();
      })

      // Functions
      .Case<typed_csl::FuncOp>([&](auto func) {
        emitIndent();
        if (func.getIsExport())
          os << "export ";
        if (func.getIsExtern())
          os << "extern ";
        if (func.getIsInline())
          os << "inline ";
        if (func.getIsNoinline())
          os << "noinline ";
        if (func.getIsAsync())
          os << "async ";
        if (func.getIsTask())
          os << "task ";
        else
          os << "fn ";
        os << func.getSymName() << "(";

        auto funcType = func.getFunctionType();
        bool hasBody = !func.getBody().empty();

        for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
          if (i > 0)
            os << ", ";
          // Use block argument name if we have a body, else generate a name
          if (hasBody) {
            Block &entry = func.getBody().front();
            os << getName(entry.getArgument(i)) << ": ";
          } else {
            os << "param" << i << ": ";
          }
          printType(funcType.getInput(i));
        }
        os << ") ";

        if (auto linkname = func.getLinkname())
          os << "linkname(\"" << *linkname << "\") ";

        if (func.getIsAsync()) {
          // For async functions, print the source-level return type from
          // async_result_type, not the CoroFrameType in function_type.
          auto asyncResultType = func.getAsyncResultType();
          if (!asyncResultType || isa<VoidType>(*asyncResultType))
            os << "void";
          else
            printType(*asyncResultType);
        } else if (funcType.getNumResults() == 0 ||
                   isa<VoidType>(funcType.getResult(0))) {
          os << "void";
        } else {
          printType(funcType.getResult(0));
        }

        if (hasBody) {
          os << " {\n";
          ++indent;
          if (failed(emitRegion(func.getBody())))
            return failure();
          --indent;
          emitIndent();
          os << "}\n";
        } else {
          // Function declaration (no body)
          os << ";\n";
        }
        return success();
      })

      // Variables
      .Case<typed_csl::VariableOp>([&](auto var) {
        beginStmt();
        os << (var.getIsConst() ? "const " : "var ");
        os << var.getName() << ": ";
        auto lvType = cast<LValueType>(var.getResult().getType());
        printType(lvType.getValueType());
        if (auto init = var.getInitializer()) {
          os << " = ";
          (void)emitExpr(init);
        }
        endStmt();
        valueNames[var.getResult()] = var.getName().str();
        return success();
      })

      // Assignments
      .Case<typed_csl::AssignOp>([&](auto assign) {
        // cslc cannot store array values through pointer dereferences
        // (e.g. `ptr.* = [N]T{...}`). Decompose into element-by-element
        // stores when the target is a dereferenced pointer to an array.
        auto lvalueType = mlir::cast<LValueType>(assign.getLvalue().getType());
        if (auto arrayTy =
                mlir::dyn_cast<ArrayType>(lvalueType.getValueType())) {
          if (assign.getLvalue()
                  .template getDefiningOp<typed_csl::DereferenceOp>()) {
            auto shape = arrayTy.getShape();
            assert(shape.size() == 1 &&
                   "multi-dim array store decomposition not implemented");
            for (int64_t i = 0; i < shape[0]; i++) {
              beginStmt();
              os << "(";
              (void)emitExpr(assign.getLvalue());
              os << ")[@as(u16, " << i << ")]";
              os << " = ";
              os << "(";
              (void)emitExpr(assign.getValue());
              os << ")[@as(u16, " << i << ")]";
              endStmt();
            }
            return success();
          }
        }

        beginStmt();
        (void)emitExpr(assign.getLvalue());

        // Detect compound assignment pattern: assign(lv, binary(load(lv), rhs))
        if (auto binaryOp =
                assign.getValue()
                    .template getDefiningOp<typed_csl::BinaryOp>()) {
          if (auto loadOp = binaryOp.getLhs()
                                .template getDefiningOp<typed_csl::LoadOp>()) {
            if (loadOp.getLvalue() == assign.getLvalue()) {
              if (auto symbol = getCompoundOpSymbol(binaryOp.getOpcode())) {
                os << " " << *symbol << " ";
                (void)emitExpr(binaryOp.getRhs());
                endStmt();
                return success();
              }
            }
          }
        }

        os << " = ";
        (void)emitExpr(assign.getValue());
        endStmt();
        return success();
      })

      // Return
      .Case<typed_csl::ReturnOp>([&](auto ret) {
        emitIndent();
        os << "return";
        if (auto val = ret.getOperand()) {
          os << " ";
          (void)emitExpr(val);
        }
        os << ";\n";
        return success();
      })

      // Unreachable - compiler-internal terminator; all paths already return
      .Case<typed_csl::UnreachableOp>([&](auto) {
        emitIndent();
        os << "// unreachable\n";
        return success();
      })

      // Coroutine ops
      .Case<typed_csl::SuspendOp, typed_csl::SuspendBrOp>([&](auto) {
        emitIndent();
        os << "suspend;\n";
        return success();
      })

      .Case<typed_csl::AwaitOp>([&](auto awaitOp) {
        emitIndent();
        os << "await ";
        (void)emitExpr(awaitOp.getFuture());
        os << ";\n";
        return success();
      })

      .Case<typed_csl::AwaitBrOp>([&](auto awaitOp) {
        return awaitOp.emitError("await_br should be lowered before emission");
      })

      // Control flow
      .Case<typed_csl::IfOp>([&](auto ifOp) -> LogicalResult {
        if (ifOp.isExpression()) {
          // Expression if - handle like other expression-producing ops
          if (materialized.contains(ifOp.getResult())) {
            emitIndent();
            os << "const " << getName(ifOp.getResult()) << " = ";
            emitIfExpr(ifOp);
            os << ";\n";
          }
          // If not materialized, it's transparent - emitted inline when used
          return success();
        }

        // Statement if
        emitIndent();
        os << "if (";
        (void)emitExpr(ifOp.getCondition());
        os << ") {\n";
        ++indent;
        if (failed(emitRegion(ifOp.getThenRegion())))
          return failure();
        --indent;
        emitIndent();
        os << "}";
        if (!ifOp.getElseRegion().empty()) {
          os << " else {\n";
          ++indent;
          if (failed(emitRegion(ifOp.getElseRegion())))
            return failure();
          --indent;
          emitIndent();
          os << "}";
        }
        os << "\n";
        return success();
      })

      .Case<typed_csl::WhileOp>([&](auto whileOp) -> LogicalResult {
        emitIndent();
        if (auto label = whileOp.getLabel())
          os << *label << ": ";
        os << "while (";
        emitRegionAsValue(whileOp.getCondRegion());
        os << ")";
        if (!whileOp.getContinuation().empty()) {
          os << " : ({";
          if (failed(emitRegion(whileOp.getContinuation(),
                                /*isInline=*/true)))
            return failure();
          os << "})";
        }
        os << " {\n";
        ++indent;
        if (failed(emitRegion(whileOp.getBody())))
          return failure();
        --indent;
        emitIndent();
        os << "}\n";
        return success();
      })

      .Case<typed_csl::ForOp>([&](auto forOp) {
        emitIndent();
        if (auto label = forOp.getLabel())
          os << *label << ": ";
        if (forOp.getIsInline())
          os << "inline ";
        os << "for (";
        (void)emitExpr(forOp.getIterable());
        os << ") |" << forOp.getElemName();
        if (forOp.hasIndex())
          os << ", " << *forOp.getIndexName();
        os << "| {\n";
        ++indent;
        if (failed(emitRegion(forOp.getBody())))
          return failure();
        --indent;
        emitIndent();
        os << "}\n";
        return success();
      })

      .Case<typed_csl::SwitchOp>([&](auto switchOp) {
        // Switch can be used as statement (no result) or expression (has
        // result)
        if (switchOp.getResult()) {
          // Expression switch - handle like other expression-producing ops
          if (materialized.contains(switchOp.getResult())) {
            emitIndent();
            os << "const " << getName(switchOp.getResult()) << " = ";
            emitSwitchExpr(switchOp);
            os << ";\n";
          }
          // If not materialized, it's transparent - emitted inline when used
          return success();
        }

        // Statement switch - emit with braced bodies
        emitIndent();
        os << "switch (";
        (void)emitExpr(switchOp.getInput());
        os << ") {\n";
        ++indent;

        auto caseValues = switchOp.getCaseValues();
        auto caseRegions = switchOp.getCaseRegions();
        bool hasElse = switchOp.hasElseBranch();

        for (size_t i = 0; i < caseRegions.size(); ++i) {
          emitIndent();
          auto thisCaseValues = mlir::cast<ArrayAttr>(caseValues[i]);
          llvm::interleaveComma(thisCaseValues, os, [&](Attribute val) {
            printValue(mlir::cast<TypedAttr>(val));
          });
          os << " => {\n";
          ++indent;
          if (failed(emitRegion(caseRegions[i])))
            return failure();
          --indent;
          emitIndent();
          os << "}";
          // Emit comma unless this is the last case and there's no else
          if (i + 1 < caseRegions.size() || hasElse)
            os << ",";
          os << "\n";
        }

        if (hasElse) {
          emitIndent();
          os << "else => {\n";
          ++indent;
          if (failed(emitRegion(switchOp.getElseRegion())))
            return failure();
          --indent;
          emitIndent();
          os << "}\n";
        }

        --indent;
        emitIndent();
        os << "}\n";
        return success();
      })

      .Case<typed_csl::LabeledBlockOp>([&](auto labeledBlock) {
        bool hasResult = labeledBlock.getNumResults() > 0;

        if (hasResult) {
          // Expression labeled block
          if (materialized.contains(labeledBlock.getResult())) {
            emitIndent();
            os << "const " << getName(labeledBlock.getResult()) << " = ";
            emitLabeledBlockExpr(labeledBlock);
            os << ";\n";
          }
          // If not materialized, it's transparent - emitted inline when used
          return success();
        }

        // Statement labeled block (void result)
        emitIndent();
        os << labeledBlock.getLabel() << ": {\n";

        ++indent;
        if (failed(emitRegion(labeledBlock.getBody())))
          return failure();
        --indent;
        emitIndent();
        os << "}\n";
        return success();
      })

      .Case<typed_csl::BreakOp>([&](auto brk) {
        emitIndent();
        os << "break";
        if (auto label = brk.getTargetLabel())
          os << " :" << *label;
        if (brk.getValue()) {
          os << " ";
          (void)emitExpr(brk.getValue());
        }
        os << ";\n";
        return success();
      })

      .Case<typed_csl::ContinueOp>([&](auto cont) {
        emitIndent();
        os << "continue";
        if (auto label = cont.getTargetLabel())
          os << " :" << *label;
        os << ";\n";
        return success();
      })

      .Case<typed_csl::ComptimeOp>([&](auto comptime) {
        emitIndent();
        os << "comptime {\n";
        ++indent;
        if (failed(emitRegion(comptime.getBody())))
          return failure();
        --indent;
        emitIndent();
        os << "}\n";
        return success();
      })

      .Case<typed_csl::LayoutOp>([&](auto layout) {
        emitIndent();
        os << "layout {\n";
        ++indent;
        if (failed(emitRegion(layout.getBody())))
          return failure();
        --indent;
        emitIndent();
        os << "}\n";
        return success();
      })

      // Yield/CondYield are handled by parent ops
      .Case<typed_csl::YieldOp, typed_csl::CondYieldOp>(
          [&](auto) { return success(); })

      // ConstCastOp - no statement needed (emitted inline as its source)
      .Case<typed_csl::ConstCastOp>([&](auto) { return success(); })

      // Expression-producing ops: emit temp var if materialized
      .Case<typed_csl::ConstantOp, typed_csl::BinaryOp, typed_csl::UnaryOp,
            typed_csl::CastOp, typed_csl::LoadOp, typed_csl::ShortCircuitOp,
            typed_csl::ConstructStructOp, typed_csl::ConstructTupleOp,
            typed_csl::ConstructArrayOp, typed_csl::FieldAccessOp,
            typed_csl::ArrayIndexAccessOp, typed_csl::TupleIndexAccessOp,
            typed_csl::PtrIndexAccessOp, typed_csl::AddressOfOp,
            typed_csl::DereferenceOp>([&](Operation *exprOp) {
        Value result = exprOp->getResult(0);
        if (materialized.contains(result)) {
          beginStmt();
          os << "const " << getName(result) << " = ";
          (void)emitDefExpr(exprOp, Precedence::Lowest);
          endStmt();
        }
        return success();
      })

      // Calls: emit as statement or temp var
      .Case<typed_csl::CallOp>([&](auto call) {
        if (call.getResult()) {
          if (materialized.contains(call.getResult())) {
            beginStmt();
            os << "const " << getName(call.getResult()) << " = ";
            emitCallExpr(call);
            endStmt();
          }
          // If not materialized, it's transparent - no statement needed
        } else {
          // Void call - always emit as statement
          beginStmt();
          emitCallExpr(call);
          endStmt();
        }
        return success();
      })

      .Case<typed_csl::IndirectCallOp>([&](auto call) {
        if (call.getResult()) {
          if (materialized.contains(call.getResult())) {
            beginStmt();
            os << "const " << getName(call.getResult()) << " = ";
            emitIndirectCallExpr(call);
            endStmt();
          }
          // If not materialized, it's transparent - no statement needed
        } else {
          // Void call - always emit as statement
          beginStmt();
          emitIndirectCallExpr(call);
          endStmt();
        }
        return success();
      })

      .Case<typed_csl::BuiltinCallOp>([&](auto call) {
        if (call.getResult()) {
          if (materialized.contains(call.getResult())) {
            beginStmt();
            os << "const " << getName(call.getResult()) << " = ";
            emitBuiltinCallExpr(call);
            endStmt();
          }
          // If not materialized, it's transparent - no statement needed
        } else {
          // Void call - always emit as statement
          beginStmt();
          emitBuiltinCallExpr(call);
          endStmt();
        }
        return success();
      })

      // Inline assembly
      .Case<typed_csl::InlineAsmOp>([&](auto asmOp) {
        beginStmt();
        os << "asm";
        if (asmOp.getIsVolatile())
          os << " volatile";
        os << " (\"";
        csl::printEscapedString(asmOp.getAsmString(), os);
        os << "\"";

        bool hasOutputs = !asmOp.getOutputs().empty();
        bool hasInputs = !asmOp.getInputs().empty();
        bool hasClobbers = !asmOp.getClobbers().empty();

        // Output section
        if (hasOutputs || hasInputs || hasClobbers) {
          os << "\n";
          ++indent;
          emitIndent();
          os << ": ";
          for (auto [idx, output] : llvm::enumerate(asmOp.getOutputs())) {
            if (idx > 0)
              os << ", ";
            auto name = mlir::cast<StringAttr>(asmOp.getOutputNames()[idx]);
            auto constraint =
                mlir::cast<StringAttr>(asmOp.getOutputConstraints()[idx]);
            os << "[" << name.getValue() << "] \"" << constraint.getValue()
               << "\" (";
            (void)emitExpr(output);
            os << ")";
          }

          // Input section
          if (hasInputs || hasClobbers) {
            os << "\n";
            emitIndent();
            os << ": ";
            for (auto [idx, input] : llvm::enumerate(asmOp.getInputs())) {
              if (idx > 0)
                os << ", ";
              auto name = mlir::cast<StringAttr>(asmOp.getInputNames()[idx]);
              auto constraint =
                  mlir::cast<StringAttr>(asmOp.getInputConstraints()[idx]);
              os << "[" << name.getValue() << "] \"" << constraint.getValue()
                 << "\" (";
              (void)emitExpr(input);
              os << ")";
            }

            // Clobber section
            if (hasClobbers) {
              os << "\n";
              emitIndent();
              os << ": ";
              for (auto [idx, clobber] : llvm::enumerate(asmOp.getClobbers())) {
                if (idx > 0)
                  os << ", ";
                os << "\"" << mlir::cast<StringAttr>(clobber).getValue()
                   << "\"";
              }
            }
          }
          --indent;
          os << "\n";
          emitIndent();
        }
        os << ")";
        endStmt();
        return success();
      })

      .Default([&](Operation *op) {
        return op->emitError("unsupported operation: ") << op->getName();
      });
}

LogicalResult CSLEmitter::emit(CompilationUnitOp op) {
  // Phase 1: Collect all nominal types
  typeRegistry.collectTypes(op);

  // Phase 2: Emit type definitions at the start
  typeRegistry.emitDefinitions(os);

  // Phase 3: Emit operations
  for (Operation &child : op.getBody()->getOperations()) {
    if (failed(emitOp(&child)))
      return failure();
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_DEF_EMITCSLPASS
#include "csl-mlir/Conversion/Passes.h.inc"

class EmitCSLPass : public impl::EmitCSLPassBase<EmitCSLPass> {
public:
  using impl::EmitCSLPassBase<EmitCSLPass>::EmitCSLPassBase;

  void runOnOperation() override {
    if (failed(
            csl::emitCSL(getOperation(), llvm::outs(), varPrefix, indentWidth)))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::emitCSL(CompilationUnitOp op, raw_ostream &os,
                                 const std::string &varPrefix,
                                 unsigned indentWidth) {
  CSLEmitter emitter(os, varPrefix, indentWidth);
  return emitter.emit(op);
}

std::unique_ptr<Pass> mlir::csl::createEmitCSLPass() {
  return std::make_unique<EmitCSLPass>();
}

std::unique_ptr<Pass>
mlir::csl::createEmitCSLPass(const EmitCSLPassOptions &options) {
  return std::make_unique<EmitCSLPass>(options);
}
