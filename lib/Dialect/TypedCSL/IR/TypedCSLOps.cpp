//===- TypedCSLOps.cpp - TypedCSL Operations ----------------------------===//
//
// Implementation of TypedCSL operations, folders, and folder helpers.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSL.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::typed_csl;

//===----------------------------------------------------------------------===//
// TableGen Interface Definitions
//===----------------------------------------------------------------------===//
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen Operation Definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "csl-mlir/Dialect/TypedCSL/IR/TypedCSLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Region exits
//===----------------------------------------------------------------------===//

LogicalResult typed_csl::detail::verifyBreakingTerminator(Operation *op) {
  auto terminator = cast<BreakingTerminatorOpInterface>(op);
  int32_t numBreakingRegions = terminator.getNumBreakingRegions();
  if (numBreakingRegions < 1)
    return op->emitOpError("must exit at least one region, but exits ")
           << numBreakingRegions;

  Operation *target = terminator.getTargetOp();
  if (!target)
    return op->emitOpError("exits ")
           << numBreakingRegions
           << " regions, exceeding the number of enclosing operations";

  // Exiting only the immediately enclosing region is a normal region exit.
  if (numBreakingRegions == 1)
    return success();

  for (Operation *parent = op->getParentOp(); parent != target;
       parent = parent->getParentOp()) {
    if (!parent->mightHaveTrait<OpTrait::PropagateControlFlowBreak>())
      return op->emitOpError("breaks through an operation that does not have "
                             "the PropagateControlFlowBreak trait: ")
             << OpWithFlags(parent, OpPrintingFlags().skipRegions());
  }

  if (!target->isRegistered())
    return success();
  auto receiver = dyn_cast<HasBreakingControlFlowOpInterface>(target);
  if (!receiver)
    return op->emitOpError("targets an operation that does not implement "
                           "HasBreakingControlFlowOpInterface: ")
           << OpWithFlags(target, OpPrintingFlags().skipRegions());
  if (!receiver.acceptsTerminator(op))
    return op->emitOpError("is not accepted as a breaking terminator by ")
           << OpWithFlags(target, OpPrintingFlags().skipRegions());
  return success();
}

Operation *typed_csl::getRegionExitTarget(Operation *terminator) {
  auto regionExit = dyn_cast<RegionExitTerminatorOpInterface>(terminator);
  if (!regionExit)
    return nullptr;
  SmallVector<Operation *> targets = regionExit.getPotentialTargets();
  return targets.size() == 1 ? targets.front() : nullptr;
}

static void
visitNestedRegionExitsImpl(Operation *root, Operation *op, int level,
                           function_ref<void(Operation *, int)> callback) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block)
        if (nested.getNumRegions() != 0)
          visitNestedRegionExitsImpl(root, &nested, level + 1, callback);
      if (block.empty())
        continue;
      Operation *terminator = &block.back();
      Operation *target = getRegionExitTarget(terminator);
      if (target && (target == root || target->isProperAncestor(root)))
        callback(terminator, level);
    }
  }
}

void typed_csl::visitNestedRegionExits(
    Operation *op, function_ref<void(Operation *, int)> callback) {
  visitNestedRegionExitsImpl(op, op, /*level=*/1, callback);
}

void typed_csl::collectNestedRegionExits(Operation *op,
                                         SmallVectorImpl<Operation *> &exits) {
  visitNestedRegionExits(op, [&](Operation *terminator, int) {
    if (getRegionExitTarget(terminator) == op)
      exits.push_back(terminator);
  });
}

//===----------------------------------------------------------------------===//
// Symbol reference helpers
//===----------------------------------------------------------------------===//

Value typed_csl::createFuncRef(OpBuilder &builder, Location loc,
                               FunctionType type, FlatSymbolRefAttr name) {
  return ConstantOp::create(builder, loc,
                            csl::FuncRefAttr::get(builder.getContext(),
                                                  /*module_identifier=*/
                                                  std::nullopt, name, type));
}

Value typed_csl::createFuncRef(OpBuilder &builder, Location loc,
                               FunctionType type, StringRef name) {
  return createFuncRef(builder, loc, type,
                       FlatSymbolRefAttr::get(builder.getContext(), name));
}

Value typed_csl::createGlobalRef(OpBuilder &builder, Location loc,
                                 csl::LValueType type, FlatSymbolRefAttr name) {
  return ConstantOp::create(builder, loc,
                            csl::GlobalRefAttr::get(builder.getContext(),
                                                    /*module_identifier=*/
                                                    std::nullopt, name, type));
}

Value typed_csl::createGlobalRef(OpBuilder &builder, Location loc,
                                 csl::LValueType type, StringRef name) {
  return createGlobalRef(builder, loc, type,
                         FlatSymbolRefAttr::get(builder.getContext(), name));
}

csl::FuncRefAttr typed_csl::getFuncRef(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  return mlir::dyn_cast<csl::FuncRefAttr>(constant.getValue());
}

csl::GlobalRefAttr typed_csl::getGlobalRef(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  return mlir::dyn_cast<csl::GlobalRefAttr>(constant.getValue());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

SmallVector<Operation *> YieldOp::getPotentialTargets() {
  return {getOperation()->getParentOp()};
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  // Verify that the value attribute type matches the result type
  Type attrType = getValue().getType();
  Type resultType = getResult().getType();

  if (attrType != resultType) {
    return emitOpError("value type ")
           << attrType << " does not match result type " << resultType;
  }

  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  // Constants always fold to their value attribute
  return getValue();
}

/// Check that every function and global reference nested in `value` names an
/// existing function or global whose type matches the one the reference
/// carries. Nothing checks symbol references inside attributes generically,
/// so every op that holds such a value has to call this.
static LogicalResult verifyNestedRefs(Operation *op, Attribute value,
                                      SymbolTableCollection &symbolTable) {
  LogicalResult result = success();
  value.walk([&](Attribute nested) -> WalkResult {
    if (auto funcRef = mlir::dyn_cast<csl::FuncRefAttr>(nested)) {
      auto func =
          symbolTable.lookupNearestSymbolFrom<FuncOp>(op, funcRef.getSymbolName());
      if (!func) {
        result = op->emitOpError("'")
                 << funcRef.getSymbolName().getValue()
                 << "' does not reference a valid function";
        return WalkResult::interrupt();
      }
      if (funcRef.getType() != func.getFunctionType()) {
        result = op->emitOpError("reference type ")
                 << funcRef.getType() << " does not match function type "
                 << func.getFunctionType() << " of "
                 << funcRef.getSymbolName();
        return WalkResult::interrupt();
      }
    } else if (auto globalRef = mlir::dyn_cast<csl::GlobalRefAttr>(nested)) {
      auto global = symbolTable.lookupNearestSymbolFrom<GlobalOp>(
          op, globalRef.getSymbolName());
      if (!global) {
        result = op->emitOpError("'")
                 << globalRef.getSymbolName().getValue()
                 << "' does not reference a valid global";
        return WalkResult::interrupt();
      }
      if (globalRef.getType().getValueType() != global.getType()) {
        result = op->emitOpError("reference type ")
                 << globalRef.getType() << " does not match variable type "
                 << global.getType() << " of the global "
                 << globalRef.getSymbolName();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

LogicalResult
ConstantOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyNestedRefs(*this, getValue(), symbolTable);
}

//===----------------------------------------------------------------------===//
// BinaryOp - Interface Methods
//===----------------------------------------------------------------------===//

LogicalResult BinaryOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Extract opcode from properties
  if (!properties) {
    if (location)
      mlir::emitError(*location) << "missing properties";
    return failure();
  }
  auto *props = properties.as<Properties *>();

  BinaryOpKindAttr opcodeAttr = props->getOpcode();
  if (!opcodeAttr) {
    if (location)
      mlir::emitError(*location) << "missing or invalid opcode attribute";
    return failure();
  }
  BinaryOpKind opcode = opcodeAttr.getValue();

  if (operands.size() != 2) {
    if (location)
      mlir::emitError(*location)
          << "expected 2 operands, got " << operands.size();
    return failure();
  }

  // Build TypeOrAttr for each operand: include comptime-known values when
  // available, since inferBinaryOp needs values for comptime types.
  auto buildOperand = [](Value v) -> csl::TypeOrAttr {
    if (auto constOp = v.getDefiningOp<ConstantOp>())
      return csl::TypeOrAttr(constOp.getValue());
    return csl::TypeOrAttr(v.getType());
  };

  csl::TypeOrAttr lhs = buildOperand(operands[0]);
  csl::TypeOrAttr rhs = buildOperand(operands[1]);

  auto result = csl::inferBinaryOp(opcode, lhs, rhs);
  if (failed(result)) {
    if (location)
      mlir::emitError(*location)
          << "cannot infer result type for binary operation '"
          << stringifyEnum(opcode) << "'";
    return failure();
  }

  inferredReturnTypes.push_back(csl::getType(result->result));
  return success();
}

//===----------------------------------------------------------------------===//
// BinaryOp - Folder and Verifier
//===----------------------------------------------------------------------===//

LogicalResult BinaryOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  bool isShift =
      getOpcode() == BinaryOpKind::Shl || getOpcode() == BinaryOpKind::Shr;

  // Non-shift: operand types must match. Shift: types may differ.
  if (!isShift && lhsType != rhsType) {
    return emitOpError("operand types must match, got ")
           << lhsType << " and " << rhsType;
  }

  // Infer expected result type
  auto expectedTypeOr = csl::inferBinaryOp(getOpcode(), lhsType, rhsType);
  if (failed(expectedTypeOr)) {
    return emitOpError("invalid operand types for operation ")
           << stringifyEnum(getOpcode());
  }

  Type expectedType = csl::getType(expectedTypeOr->result);
  if (resultType != expectedType) {
    return emitOpError("result type ")
           << resultType << " does not match expected type " << expectedType;
  }

  return success();
}

OpFoldResult BinaryOp::fold(FoldAdaptor adaptor) {
  auto lhsTypedAttr = mlir::dyn_cast_or_null<TypedAttr>(adaptor.getLhs());
  auto rhsTypedAttr = mlir::dyn_cast_or_null<TypedAttr>(adaptor.getRhs());
  if (!lhsTypedAttr || !rhsTypedAttr)
    return nullptr;

  auto result = csl::inferBinaryOp(getOpcode(), csl::TypeOrAttr(lhsTypedAttr),
                                   csl::TypeOrAttr(rhsTypedAttr));
  if (failed(result) || !csl::isComptimeKnown(result->result))
    return nullptr;
  return std::get<TypedAttr>(result->result);
}

// BinaryOp::getCanonicalizationPatterns is defined in TypedCSLCanonicalize.cpp

//===----------------------------------------------------------------------===//
// UnaryOp - Folder and Verifier
//===----------------------------------------------------------------------===//

LogicalResult UnaryOp::verify() {
  Type operandType = getOperand().getType();

  // Verify the operation is valid for the operand type
  auto expectedTypeOr = csl::inferUnaryResultType(getOpcode(), operandType);
  if (failed(expectedTypeOr)) {
    return emitOpError("invalid operand type for operation ")
           << stringifyEnum(getOpcode());
  }

  return success();
}

OpFoldResult UnaryOp::fold(FoldAdaptor adaptor) {
  auto operandTypedAttr =
      mlir::dyn_cast_or_null<TypedAttr>(adaptor.getOperand());
  if (!operandTypedAttr)
    return nullptr;

  auto result = csl::evaluateUnaryStrict(getOpcode(), operandTypedAttr);
  if (failed(result))
    return nullptr;
  return *result;
}

//===----------------------------------------------------------------------===//
// CastOp - Interface Methods
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  // Delegate to CSL helper
  return csl::areCastCompatible(inputs[0], outputs[0]);
}

//===----------------------------------------------------------------------===//
// CastOp - Folder
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  // No-op cast
  if (getSource().getType() == getDest().getType())
    return getSource();

  // Can only fold if source is constant
  auto sourceAttr = mlir::dyn_cast_or_null<TypedAttr>(adaptor.getSource());
  if (!sourceAttr)
    return nullptr;

  auto result = csl::evaluateCast(sourceAttr, getDest().getType());
  if (failed(result))
    return nullptr;
  return *result;
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

/// Whether the address of the storage behind `lvalue` is taken, directly or
/// through a field, element or const view of it.
static bool isAddressTaken(Value lvalue) {
  for (Operation *user : lvalue.getUsers()) {
    if (mlir::isa<AddressOfOp>(user))
      return true;
    if (mlir::isa<FieldAccessOp, TupleIndexAccessOp, ArrayIndexAccessOp,
                  ConstCastOp>(user) &&
        isAddressTaken(user->getResult(0)))
      return true;
  }
  return false;
}

OpFoldResult VariableOp::fold(FoldAdaptor adaptor) {
  // Only const variables can be folded
  if (!getIsConst()) {
    return nullptr;
  }

  // Taking the address needs the variable's storage, not its value.
  if (isAddressTaken(getResult()))
    return nullptr;

  // Must have an initializer to fold
  auto initAttr =
      llvm::dyn_cast_if_present<TypedAttr>(adaptor.getInitializer());
  if (!initAttr) {
    return nullptr;
  }

  // Const variable with constant initializer folds to the initializer value
  return LValueAttr::get(initAttr);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

OpFoldResult LoadOp::fold(FoldAdaptor adaptor) {
  // Make sure the lvalue type is const
  auto lvalueType = llvm::cast<typed_csl::LValueType>(getLvalue().getType());

  if (!lvalueType.getIsConst()) {
    return nullptr;
  }

  auto lvalueAttr = llvm::dyn_cast_if_present<LValueAttr>(adaptor.getLvalue());
  if (lvalueAttr) {
    return lvalueAttr.getRvalue();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

LogicalResult AddressOfOp::verify() {
  // We verify that pointee type and constness match.
  csl::LValueType lvalueType = getOperand().getType();
  csl::PointerType ptrType = getResult().getType();

  if (ptrType.getPointeeType() != lvalueType.getValueType()) {
    return emitOpError("pointer pointee type ")
           << csl::prettyPrintType(ptrType.getPointeeType())
           << " does not match lvalue value type "
           << csl::prettyPrintType(lvalueType.getValueType());
  }

  if (ptrType.getIsConst() != lvalueType.getIsConst()) {
    return emitOpError("pointer constness does not match lvalue constness");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DereferenceOp
//===----------------------------------------------------------------------===//

LogicalResult DereferenceOp::verify() {
  // We verify that pointee type and constness match.
  csl::PointerType ptrType = getOperand().getType();
  csl::LValueType lvalueType = getResult().getType();

  if (lvalueType.getValueType() != ptrType.getPointeeType()) {
    return emitOpError("lvalue value type ")
           << csl::prettyPrintType(lvalueType.getValueType())
           << " does not match pointer pointee type "
           << csl::prettyPrintType(ptrType.getPointeeType());
  }

  if (lvalueType.getIsConst() != ptrType.getIsConst()) {
    return emitOpError("lvalue constness does not match pointer constness");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::print(OpAsmPrinter &p) {
  // Print: typed_csl.func [async] @name(%arg0: type0, %arg1: type1) ->
  // returnType {
  if (getIsAsync())
    p << " async";
  p << ' ';
  p.printSymbolName(getSymName());

  FunctionType fnType = getFunctionType();
  Region &body = getBody();

  p << '(';
  for (unsigned i = 0, e = fnType.getNumInputs(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    if (!body.empty()) {
      p.printOperand(body.front().getArgument(i));
      p << ": ";
    }
    p.printType(fnType.getInput(i));
  }
  p << ')';

  // Print return type if not void
  if (fnType.getNumResults() > 0) {
    p << " -> ";
    if (fnType.getNumResults() == 1) {
      p.printType(fnType.getResult(0));
    } else {
      p << '(';
      llvm::interleaveComma(fnType.getResults(), p);
      p << ')';
    }
  }

  // Print linkname if present
  if (auto linkname = getLinkname())
    p << " linkname(\"" << *linkname << "\")";

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"sym_name", "function_type",
                                           "linkname", "is_async", "arg_attrs",
                                           "res_attrs"});
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: typed_csl.func [async] @name(%arg0: type0, %arg1: type1) ->
  // returnType {

  // Parse optional 'async' keyword
  if (succeeded(parser.parseOptionalKeyword("async")))
    result.addAttribute(getIsAsyncAttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;

  // Parse argument list
  if (parser.parseArgumentList(args, AsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  for (auto &arg : args)
    argTypes.push_back(arg.type);

  // Parse optional return type
  if (succeeded(parser.parseOptionalArrow())) {
    Type returnType;
    if (parser.parseType(returnType))
      return failure();
    resultTypes.push_back(returnType);
  }

  // Build function type (uses plain types for calling convention)
  auto fnType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  // Block args are const lvalue wrappers of the function_type inputs
  for (auto &arg : args)
    arg.type = csl::LValueType::get(arg.type, /*isConst=*/true);

  // Parse optional linkname
  if (succeeded(parser.parseOptionalKeyword("linkname"))) {
    if (parser.parseLParen())
      return failure();
    std::string linknameStr;
    if (parser.parseString(&linknameStr))
      return failure();
    if (parser.parseRParen())
      return failure();
    result.addAttribute(getLinknameAttrName(result.name),
                        StringAttr::get(parser.getContext(), linknameStr));
  }

  // Parse body region
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Attribute FuncOp::getArgAttr(unsigned index, StringRef name) {
  ArrayAttr attrs = getArgAttrsAttr();
  if (!attrs || index >= attrs.size())
    return {};
  return mlir::cast<DictionaryAttr>(attrs[index]).get(name);
}

void FuncOp::setArgAttr(unsigned index, StringRef name, Attribute value) {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> dicts;
  if (ArrayAttr attrs = getArgAttrsAttr())
    dicts.assign(attrs.begin(), attrs.end());
  dicts.resize(getFunctionType().getNumInputs(), DictionaryAttr::get(ctx));
  NamedAttrList list(mlir::cast<DictionaryAttr>(dicts[index]));
  list.set(name, value);
  dicts[index] = list.getDictionary(ctx);
  setArgAttrsAttr(ArrayAttr::get(ctx, dicts));
}

LogicalResult FuncOp::verify() {
  // Verify that the function body has the correct number of arguments
  FunctionType fnType = getFunctionType();
  Region &body = getBody();

  // The dialect verifies its argument attributes, as MLIR does for the
  // arguments of a FunctionOpInterface op.
  if (ArrayAttr argAttrs = getArgAttrsAttr()) {
    auto *dialect = getContext()->getLoadedDialect<TypedCSLDialect>();
    for (auto [index, attrs] : llvm::enumerate(argAttrs)) {
      for (NamedAttribute attr : mlir::cast<DictionaryAttr>(attrs)) {
        if (!attr.getName().getValue().starts_with("typed_csl."))
          continue;
        if (failed(dialect->verifyRegionArgAttribute(*this, /*regionIndex=*/0,
                                                     index, attr)))
          return failure();
      }
    }
  }

  if (body.empty())
    return success();

  Block &entryBlock = body.front();
  if (entryBlock.getNumArguments() != fnType.getNumInputs()) {
    return emitOpError("entry block has ")
           << entryBlock.getNumArguments()
           << " arguments but function type has " << fnType.getNumInputs()
           << " inputs";
  }

  for (unsigned i = 0, e = fnType.getNumInputs(); i < e; ++i) {
    auto expectedType =
        csl::LValueType::get(fnType.getInput(i), /*isConst=*/true);
    if (entryBlock.getArgument(i).getType() != expectedType) {
      return emitOpError("entry block argument ")
             << i << " has type " << entryBlock.getArgument(i).getType()
             << " but expected " << expectedType;
    }
  }

  if (getIsNoinline() && getIsInline())
    return emitOpError("cannot be both 'inline' and 'noinline'");
  if (getIsNoinline() && getIsExport())
    return emitOpError("exported functions cannot be 'noinline'");
  if (getIsNoinline() && getIsExtern())
    return emitOpError("extern functions cannot be 'noinline'");
  if (getIsNoinline() && getIsTask())
    return emitOpError("tasks cannot be 'noinline'");

  return success();
}

//===----------------------------------------------------------------------===//
// SuspendBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands SuspendBrOp::getSuccessorOperands(unsigned index) {
  return SuccessorOperands(MutableOperandRange(getOperation(), 0, 0));
}

/// If the successor is a pass-through block (contains only a cf.br),
/// redirect to the final target.
static bool collapsePassThroughSuccessor(Operation *op, unsigned succIdx) {
  Block *succ = op->getSuccessor(succIdx);
  // Must have exactly one op (the terminator).
  if (!llvm::hasSingleElement(*succ))
    return false;
  // Must be an unconditional branch.
  auto br = mlir::dyn_cast<cf::BranchOp>(succ->getTerminator());
  if (!br || br.getDest() == succ)
    return false;
  // Must have no block args (we don't remap args).
  if (!succ->args_empty())
    return false;
  op->setSuccessor(br.getDest(), succIdx);
  return true;
}

void SuspendBrOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add(+[](SuspendBrOp op, PatternRewriter &rewriter) {
    return success(collapsePassThroughSuccessor(op, 0));
  });
}

//===----------------------------------------------------------------------===//
// AwaitBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands AwaitBrOp::getSuccessorOperands(unsigned index) {
  return SuccessorOperands(MutableOperandRange(getOperation(), 0, 0));
}

void AwaitBrOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add(+[](AwaitBrOp op, PatternRewriter &rewriter) {
    return success(collapsePassThroughSuccessor(op, 0));
  });
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // Get parent function
  auto funcOp = (*this)->getParentOfType<FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested in a typed_csl.func operation");

  // For async functions, the function_type result is CoroFrameType;
  // the actual return type is stored in async_result_type.
  Type expectedType;
  unsigned numRealResults;
  if (funcOp.getIsAsync()) {
    auto asyncResultType = funcOp.getAsyncResultType();
    if (asyncResultType) {
      expectedType = *asyncResultType;
      numRealResults = 1;
    } else {
      numRealResults = 0;
    }
  } else {
    FunctionType fnType = funcOp.getFunctionType();
    numRealResults = fnType.getNumResults();
    if (numRealResults > 0)
      expectedType = fnType.getResult(0);
  }

  // Check return value matches function return type
  Value operand = getOperand();
  if (operand) {
    if (numRealResults == 0) {
      return emitOpError("returns a value but function has void return type");
    }

    if (numRealResults > 1) {
      return emitOpError("returns a single value but function has ")
             << numRealResults << " return types";
    }

    Type returnType = operand.getType();
    if (returnType != expectedType) {
      return emitOpError("return type ")
             << returnType << " does not match function return type "
             << expectedType;
    }
  } else {
    if (numRealResults != 0) {
      return emitOpError("returns no value but function expects return type ")
             << expectedType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RegionBranchTerminatorOpInterface implementations
//===----------------------------------------------------------------------===//

MutableOperandRange
CondYieldOp::getMutableSuccessorOperands(RegionSuccessor successor) {
  // The condition is consumed by WhileOp's branching logic, not forwarded.
  return MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0);
}

void CondYieldOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  auto whileOp = (*this)->getParentOfType<WhileOp>();

  auto boolAttr = dyn_cast_or_null<BoolAttr>(operands[0]);
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&whileOp.getBody());
  if (!boolAttr || !boolAttr.getValue())
    regions.push_back(RegionSuccessor(whileOp.getOperation()));
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface implementations
//===----------------------------------------------------------------------===//

/// If `point` is a region exit that breaks through `op` to an enclosing op,
/// append the successors that the addressed op reports for it and return
/// true. An addressed op that does not model region branches (typed_csl.func)
/// has no region successors: control leaves the function.
static bool
appendPropagatedSuccessors(Operation *op, RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) {
  Operation *terminator = point.getTerminatorPredecessorOrNull();
  if (!terminator)
    return false;
  Operation *target = getRegionExitTarget(terminator);
  if (!target || !target->isProperAncestor(op))
    return false;
  if (auto targetBranch = dyn_cast<RegionBranchOpInterface>(target))
    targetBranch.getSuccessorRegions(point, regions);
  return true;
}

/// Return true if `point` is a typed_csl.break addressed to `op`.
static bool isBreakTo(Operation *op, RegionBranchPoint point) {
  Operation *terminator = point.getTerminatorPredecessorOrNull();
  return isa_and_nonnull<BreakOp>(terminator) &&
         getRegionExitTarget(terminator) == op;
}

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    if (appendPropagatedSuccessors(getOperation(), point, regions))
      return;
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  regions.push_back(RegionSuccessor(&getThenRegion()));
  if (!getElseRegion().empty())
    regions.push_back(RegionSuccessor(&getElseRegion()));
  else
    regions.push_back(RegionSuccessor(getOperation()));
}

void ForOp::getSuccessorRegions(RegionBranchPoint point,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    // May enter body or skip entirely (empty range).
    regions.push_back(RegionSuccessor(&getBody()));
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  if (appendPropagatedSuccessors(getOperation(), point, regions))
    return;
  if (isBreakTo(getOperation(), point)) {
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  // From body: may loop back or exit.
  regions.push_back(RegionSuccessor(&getBody()));
  regions.push_back(RegionSuccessor(getOperation()));
}

void WhileOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getCondRegion()));
    return;
  }
  if (appendPropagatedSuccessors(getOperation(), point, regions))
    return;

  // Breaks and continues addressed to this loop may be nested in other ops.
  Operation *terminator = point.getTerminatorPredecessorOrNull();
  if (getRegionExitTarget(terminator) == getOperation()) {
    if (isa<BreakOp>(terminator)) {
      regions.push_back(RegionSuccessor(getOperation()));
      return;
    }
    if (isa<ContinueOp>(terminator)) {
      if (!getContinuation().empty())
        regions.push_back(RegionSuccessor(&getContinuation()));
      else
        regions.push_back(RegionSuccessor(&getCondRegion()));
      return;
    }
  }

  Region *predecessorRegion = terminator->getParentRegion();

  if (predecessorRegion == &getCondRegion()) {
    // Condition yields true → body, false → exit.
    regions.push_back(RegionSuccessor(&getBody()));
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }

  if (predecessorRegion == &getBody()) {
    // Body → continuation (if present) or back to condition.
    if (!getContinuation().empty())
      regions.push_back(RegionSuccessor(&getContinuation()));
    else
      regions.push_back(RegionSuccessor(&getCondRegion()));
    return;
  }

  // Continuation → back to condition.
  assert(predecessorRegion == &getContinuation());
  regions.push_back(RegionSuccessor(&getCondRegion()));
}

void LabeledBlockOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  if (appendPropagatedSuccessors(getOperation(), point, regions))
    return;
  regions.push_back(RegionSuccessor(getOperation()));
}

void SwitchOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    if (appendPropagatedSuccessors(getOperation(), point, regions))
      return;
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  // From parent: any case region or else region.
  for (Region &caseRegion : getCaseRegions())
    regions.push_back(RegionSuccessor(&caseRegion));
  if (hasElseBranch())
    regions.push_back(RegionSuccessor(&getElseRegion()));
}

void ShortCircuitOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    // LHS may short-circuit (→ parent) or evaluate RHS (→ rhs_region).
    regions.push_back(RegionSuccessor(&getRhsRegion()));
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  regions.push_back(RegionSuccessor(getOperation()));
}

void ComptimeOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  if (appendPropagatedSuccessors(getOperation(), point, regions))
    return;
  regions.push_back(RegionSuccessor(getOperation()));
}

void LayoutOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  if (appendPropagatedSuccessors(getOperation(), point, regions))
    return;
  regions.push_back(RegionSuccessor(getOperation()));
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface::getSuccessorInputs implementations
//===----------------------------------------------------------------------===//

ValueRange IfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                 : ValueRange();
}

ValueRange ShortCircuitOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                 : ValueRange();
}

OperandRange
ShortCircuitOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  // On the short-circuit path (parent → parent), the LHS is forwarded as the
  // result. On the non-short-circuit path (parent → rhs_region), nothing is
  // forwarded (the RHS region has no block arguments).
  if (successor.isOperation()) {
    // LHS is operand #1 (after the kind attribute which is a property).
    return OperandRange(getOperation()->operand_begin(),
                        getOperation()->operand_begin() + 1);
  }
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

ValueRange SwitchOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                 : ValueRange();
}

ValueRange LabeledBlockOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getOperation()->getResults())
                                 : ValueRange();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  // Type verification is handled by the type constraints in the operation
  // definition The functional-type format ensures operands and results match
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (TypedAttr initializer = getInitializerAttr())
    return verifyNestedRefs(*this, initializer, symbolTable);
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  assert(fn->getNumResults() <= 1 && "multiple return values not supported");
  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult().getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  // Async functions can only be called from async functions or tasks.
  if (fn.getIsAsync()) {
    auto parentFunc = (*this)->getParentOfType<FuncOp>();
    if (!parentFunc || !parentFunc.getIsAsync())
      return emitOpError("calls to async functions are only allowed in "
                         "async functions or tasks");
  }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// IndirectCallOp
//===----------------------------------------------------------------------===//

LogicalResult IndirectCallOp::verify() {
  // Verify that the callee is a function pointer type
  Type calleeType = getCallee().getType();
  auto ptrType = mlir::dyn_cast<csl::PointerType>(calleeType);
  if (!ptrType)
    return emitOpError("callee must be a function pointer type, got ")
           << calleeType;

  Type pointeeType = ptrType.getPointeeType();
  auto funcType = mlir::dyn_cast<FunctionType>(pointeeType);
  if (!funcType)
    return emitOpError("callee pointer must point to a function type, got ")
           << pointeeType;

  // Verify argument count matches
  if (funcType.getNumInputs() != getArgs().size())
    return emitOpError("incorrect number of arguments for callee: expected ")
           << funcType.getNumInputs() << ", got " << getArgs().size();

  // Verify argument types match
  for (unsigned i = 0, e = funcType.getNumInputs(); i != e; ++i)
    if (getArgs()[i].getType() != funcType.getInput(i))
      return emitOpError("argument type mismatch at index ")
             << i << ": expected " << funcType.getInput(i) << ", got "
             << getArgs()[i].getType();

  // Verify result type matches (if any)
  if (funcType.getNumResults() == 0) {
    if (getResult())
      return emitOpError("callee returns void but call has result");
  } else {
    if (!getResult())
      return emitOpError("callee returns value but call has no result");
    if (getResult().getType() != funcType.getResult(0))
      return emitOpError("result type mismatch: expected ")
             << funcType.getResult(0) << ", got " << getResult().getType();
  }

  return success();
}

FunctionType IndirectCallOp::getCalleeType() {
  Type calleeType = getCallee().getType();
  auto ptrType = mlir::cast<csl::PointerType>(calleeType);
  return mlir::cast<FunctionType>(ptrType.getPointeeType());
}

//===----------------------------------------------------------------------===//
// BuiltinCallOp
//===----------------------------------------------------------------------===//

void BuiltinCallOp::getEffects(
    ::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
        &effects) {
  // Check if this builtin is pure using the descriptor
  const csl::BuiltinDescriptor &desc =
      csl::BuiltinDescriptor::get(getBuiltin(), getContext());

  bool inComptimeContext =
      getOperation()->getParentOfType<ComptimeOp>() != nullptr;
  if (desc.mustKeepInOutput(inComptimeContext)) {
    using namespace mlir::MemoryEffects;
    // Not pure - has unknown side effects (conservative)
    effects.emplace_back(Write::get());
  }
  // Pure builtins have no effects, so nothing to add
}

LogicalResult BuiltinCallOp::verify() {
  // Verify argument types match the builtin signature using the signature
  // matcher
  SmallVector<TypedAttr> argValues;
  for (Value arg : getArgs()) {
    if (auto constOp =
            llvm::dyn_cast_or_null<ConstantOp>(arg.getDefiningOp())) {
      argValues.push_back(constOp.getValue());
    } else {
      argValues.push_back(csl::NoValueAttr::get(arg.getType()));
    }
  }

  csl::SignatureMatchResult matchResult = csl::BuiltinSignatureMatcher::match(
      getBuiltin(), argValues, getContext());

  if (!matchResult.isValid) {
    return emitOpError() << matchResult.errorMessage;
  }

  // Verify result type matches inferred type
  Type expectedResultType = matchResult.inferredReturnType;
  Type actualResultType =
      getResult() ? getResult().getType() : csl::VoidType::get(getContext());

  if (expectedResultType != actualResultType) {
    return emitOpError("result type ")
           << actualResultType << " does not match expected type "
           << expectedResultType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  // Expression form requires both regions
  if (isExpression() && getElseRegion().empty())
    return emitOpError("if-expression requires an else region");

  // Helper to get YieldOp if the region ends with one (may not if region
  // contains return/break/continue which transfer control elsewhere)
  auto getRegionYield = [](Region &region) -> YieldOp {
    if (region.empty() || region.front().empty())
      return nullptr;
    return mlir::dyn_cast<YieldOp>(&region.front().back());
  };

  // If this is an expression, verify yield types match result type
  if (isExpression()) {
    Type resultType = getResult().getType();

    if (auto yieldOp = getRegionYield(getThenRegion())) {
      if (yieldOp.getValue().getType() != resultType)
        return emitOpError("then branch yield type ")
               << yieldOp.getValue().getType() << " must match result type "
               << resultType;
    }

    if (auto yieldOp = getRegionYield(getElseRegion())) {
      if (yieldOp.getValue().getType() != resultType)
        return emitOpError("else branch yield type ")
               << yieldOp.getValue().getType() << " must match result type "
               << resultType;
    }
  }

  return success();
}

LogicalResult IfOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  // Can't fold region-based if - use canonicalizer for constant conditions
  return failure();
}

/// Custom assembly format:
///   Statement form: typed_csl.if %cond : i1 { ... } (else { ... })?
///   Expression form: %r = typed_csl.if %cond : i1 { ... } else { ... } : type
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand cond;
  Type condType;
  if (parser.parseOperand(cond) || parser.parseColonType(condType) ||
      parser.resolveOperand(cond, condType, result.operands))
    return failure();

  // Parse then region
  Region *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion))
    return failure();

  // Parse optional else region
  Region *elseRegion = result.addRegion();
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion))
      return failure();
  }

  // Parse optional result type (expression form)
  if (!parser.parseOptionalColon()) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

void IfOp::print(OpAsmPrinter &p) {
  p << ' ' << getCondition() << " : " << getCondition().getType() << ' ';
  p.printRegion(getThenRegion());
  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion());
  }
  if (isExpression())
    p << " : " << getResult().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse input operand and type: %input : si32
  OpAsmParser::UnresolvedOperand inputOperand;
  Type inputType;
  if (parser.parseOperand(inputOperand) || parser.parseColonType(inputType))
    return failure();

  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return failure();

  // Parse "cases" keyword and case values: cases [[1, 2], [3]]
  if (parser.parseKeyword("cases"))
    return failure();

  SmallVector<Attribute> caseValuesAttrs;
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOptionalRSquare().failed()) {
    do {
      SmallVector<Attribute> thisCaseValues;
      if (parser.parseLSquare())
        return failure();

      if (parser.parseOptionalRSquare().failed()) {
        do {
          Attribute valAttr;
          if (parser.parseAttribute(valAttr))
            return failure();
          thisCaseValues.push_back(valAttr);
        } while (succeeded(parser.parseOptionalComma()));

        if (parser.parseRSquare())
          return failure();
      }
      caseValuesAttrs.push_back(
          ArrayAttr::get(parser.getContext(), thisCaseValues));
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare())
      return failure();
  }

  result.addAttribute(getCaseValuesAttrName(result.name),
                      ArrayAttr::get(parser.getContext(), caseValuesAttrs));

  // Add else region first (MLIR requires variadic regions last)
  Region *elseRegion = result.addRegion();

  // Parse case regions
  if (parser.parseLSquare())
    return failure();

  do {
    Region *caseRegion = result.addRegion();
    if (parser.parseRegion(*caseRegion))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare())
    return failure();

  // Parse optional else region content
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion))
      return failure();
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse optional result type: : si32
  if (succeeded(parser.parseOptionalColon())) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
  }

  return success();
}

void SwitchOp::print(OpAsmPrinter &p) {
  p << " " << getInput() << " : " << getInput().getType();

  // Print case values: cases [[1, 2], [3]]
  p << " cases [";
  auto caseValues = getCaseValues();
  for (size_t i = 0; i < caseValues.size(); ++i) {
    if (i > 0)
      p << ", ";
    p << "[";
    auto thisCaseValues = mlir::cast<ArrayAttr>(caseValues[i]);
    llvm::interleaveComma(thisCaseValues, p,
                          [&](Attribute val) { p.printAttribute(val); });
    p << "]";
  }
  p << "]";

  // Print case regions
  p << " [";
  llvm::interleaveComma(getCaseRegions(), p,
                        [&](Region &region) { p.printRegion(region); });
  p << "]";

  // Print else region if present
  if (hasElseBranch()) {
    p << " else ";
    p.printRegion(getElseRegion());
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getCaseValuesAttrName()});

  // Print result type if present
  if (getResult()) {
    p << " : " << getResult().getType();
  }
}

LogicalResult SwitchOp::verify() {
  // Verify case values count matches case regions
  auto caseValues = getCaseValues();
  if (caseValues.size() != getCaseRegions().size()) {
    return emitOpError("number of case value groups (")
           << caseValues.size() << ") must match number of case regions ("
           << getCaseRegions().size() << ")";
  }

  // Helper to get YieldOp if the region ends with one (may not if region
  // contains return/break/continue which transfer control elsewhere)
  auto getRegionYield = [](Region &region) -> YieldOp {
    if (region.empty() || region.front().empty())
      return nullptr;
    return mlir::dyn_cast<YieldOp>(&region.front().back());
  };

  // Verify case regions are not empty
  for (auto [idx, caseRegion] : llvm::enumerate(getCaseRegions())) {
    if (caseRegion.empty()) {
      return emitOpError("case region ") << idx << " must not be empty";
    }
  }

  // If this is an expression (has result), verify all yields have matching type
  // Regions that don't end with yield (e.g., end with return) don't contribute
  // to the switch result type
  if (getResult()) {
    Type resultType = getResult().getType();

    for (auto [idx, caseRegion] : llvm::enumerate(getCaseRegions())) {
      if (auto yieldOp = getRegionYield(caseRegion)) {
        if (yieldOp.getValue().getType() != resultType) {
          return emitOpError("case region ")
                 << idx << " yield type " << yieldOp.getValue().getType()
                 << " must match result type " << resultType;
        }
      }
    }

    if (hasElseBranch()) {
      if (auto yieldOp = getRegionYield(getElseRegion())) {
        if (yieldOp.getValue().getType() != resultType) {
          return emitOpError("else region yield type ")
                 << yieldOp.getValue().getType() << " must match result type "
                 << resultType;
        }
      }
    }
  }

  return success();
}

LogicalResult SwitchOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  // Can't fold switch with regions - use canonicalizer for constant input
  return failure();
}

// getCanonicalizationPatterns is defined in TypedCSLCanonicalize.cpp

//===----------------------------------------------------------------------===//
// ShortCircuitOp
//===----------------------------------------------------------------------===//

LogicalResult ShortCircuitOp::verify() {
  // Verify rhs region is terminated by YieldOp
  if (getRhsRegion().empty())
    return emitOpError("rhs region must not be empty");
  auto *terminator = getRhsRegion().front().getTerminator();
  if (!mlir::isa<YieldOp>(terminator))
    return emitOpError("rhs region must be terminated by 'typed_csl.yield'");

  // Verify yielded type is bool (i1)
  auto yieldOp = mlir::cast<YieldOp>(terminator);
  if (!yieldOp.getValue().getType().isInteger(1))
    return emitOpError("rhs region must yield a boolean (i1) value");

  return success();
}

OpFoldResult ShortCircuitOp::fold(FoldAdaptor adaptor) {
  // Can't fold region-based short-circuit - use canonicalizer for constant LHS
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConstructTupleOp
//===----------------------------------------------------------------------===//

LogicalResult ConstructTupleOp::verify() {
  auto structType = mlir::cast<csl::StructType>(getResult().getType());

  // Verify it's a tuple (no field names)
  if (!structType.isTuple()) {
    return emitOpError(
        "result type must be a tuple (struct with no field names)");
  }

  // Verify field count matches
  if (getFieldValues().size() != structType.getNumFields()) {
    return emitOpError("expected ")
           << structType.getNumFields() << " field values, got "
           << getFieldValues().size();
  }

  // Verify field types match
  for (size_t i = 0; i < getFieldValues().size(); ++i) {
    Type valueType = getFieldValues()[i].getType();
    Type fieldType = structType.getFieldType(i);
    if (valueType != fieldType) {
      return emitOpError("field ")
             << i << " has type " << valueType << " but struct field has type "
             << fieldType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructStructOp
//===----------------------------------------------------------------------===//

LogicalResult ConstructStructOp::verify() {
  auto structType = mlir::cast<csl::StructType>(getResult().getType());

  // Verify it's not a tuple (tuples use construct_tuple)
  if (structType.isTuple()) {
    return emitOpError("result type must not be a tuple "
                       "(use construct_tuple for tuples)");
  }

  // Verify field count matches
  ArrayRef<Attribute> fieldNames = getFieldNames().getValue();
  if (fieldNames.size() != structType.getNumFields()) {
    return emitOpError("field_names array has ")
           << fieldNames.size() << " elements but struct has "
           << structType.getNumFields() << " fields";
  }

  if (getFieldValues().size() != structType.getNumFields()) {
    return emitOpError("expected ")
           << structType.getNumFields() << " field values, got "
           << getFieldValues().size();
  }

  // Verify field names match
  for (size_t i = 0; i < fieldNames.size(); ++i) {
    auto nameAttr = mlir::dyn_cast<StringAttr>(fieldNames[i]);
    if (!nameAttr) {
      return emitOpError("field_names[") << i << "] must be a string attribute";
    }
    FailureOr<StringRef> expectedName = structType.getFieldName(i);
    assert(succeeded(expectedName));
    if (nameAttr.getValue() != *expectedName) {
      return emitOpError("field_names[")
             << i << "] is '" << nameAttr.getValue()
             << "' but struct field name is '" << *expectedName << "'";
    }
  }

  // Verify field types match
  for (size_t i = 0; i < getFieldValues().size(); ++i) {
    Type valueType = getFieldValues()[i].getType();
    Type fieldType = structType.getFieldType(i);
    if (valueType != fieldType) {
      return emitOpError("field '")
             << *structType.getFieldName(i) << "' has type " << valueType
             << " but struct field has type " << fieldType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructArrayOp
//===----------------------------------------------------------------------===//

LogicalResult ConstructArrayOp::verify() {
  auto arrayType = mlir::cast<csl::ArrayType>(getResult().getType());

  // Verify element count matches array total size
  int64_t expectedSize = arrayType.getNumElements();
  if (static_cast<int64_t>(getElements().size()) != expectedSize) {
    return emitOpError("expected ")
           << expectedSize << " elements for array type, got "
           << getElements().size();
  }

  // Verify element types match
  Type expectedElemType = arrayType.getElementType();
  for (auto [i, elem] : llvm::enumerate(getElements())) {
    Type elemType = elem.getType();
    if (elemType != expectedElemType) {
      return emitOpError("element ")
             << i << " has type " << elemType << " but array element type is "
             << expectedElemType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FieldAccessOp
//===----------------------------------------------------------------------===//

LogicalResult FieldAccessOp::verify() {
  Type baseType = getStructLvalue().getType();
  bool isLValue = false;
  bool isConst = false;

  if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(baseType)) {
    isLValue = true;
    isConst = lvalueType.getIsConst();
    baseType = lvalueType.getValueType();
  }

  auto structType = mlir::dyn_cast<csl::StructType>(baseType);

  if (!structType) {
    return emitOpError("operand must be a struct type");
  }

  // Verify it's not a tuple (tuples don't have named fields)
  if (structType.isTuple()) {
    return emitOpError("cannot use field access on tuple (use index access)");
  }

  // Verify field exists
  StringRef fieldName = getFieldName();
  auto fieldIndexOpt = structType.getFieldIndex(fieldName);
  if (failed(fieldIndexOpt)) {
    return emitOpError("struct type does not have field '") << fieldName << "'";
  }

  // Verify result type
  Type resultType = getResult().getType();
  Type expectedFieldType = structType.getFieldType(*fieldIndexOpt);

  if (isLValue) {
    // If input is lvalue, result must also be lvalue
    auto resultLValueType = mlir::dyn_cast<csl::LValueType>(resultType);
    if (!resultLValueType) {
      return emitOpError("result must be an lvalue when input is an lvalue");
    }

    if (resultLValueType.getValueType() != expectedFieldType) {
      return emitOpError("result lvalue has value type ")
             << resultLValueType.getValueType() << " but struct field has type "
             << expectedFieldType;
    }

    // Verify constness matches
    if (resultLValueType.getIsConst() != isConst) {
      return emitOpError(
          "result lvalue constness must match struct lvalue constness");
    }
  } else {
    // If input is plain struct, result must be the field type directly
    if (resultType != expectedFieldType) {
      return emitOpError("result type ")
             << resultType << " does not match struct field type "
             << expectedFieldType;
    }
  }

  return success();
}

OpFoldResult FieldAccessOp::fold(FoldAdaptor adaptor) {
  Attribute baseAttr = adaptor.getStructLvalue();
  if (!baseAttr)
    return nullptr;

  // Handle LValueAttr (from const variable fold)
  csl::StructValueAttr structValue;
  bool wasLValue = false;

  if (auto lvalueAttr = mlir::dyn_cast<csl::LValueAttr>(baseAttr)) {
    structValue = mlir::dyn_cast<csl::StructValueAttr>(lvalueAttr.getRvalue());
    wasLValue = true;
  } else {
    structValue = mlir::dyn_cast<csl::StructValueAttr>(baseAttr);
  }

  if (!structValue)
    return nullptr;

  // Extract the field value
  TypedAttr fieldValue = structValue.getField(getFieldName());
  if (!fieldValue)
    return nullptr;

  // If base was an lvalue, wrap result in LValueAttr
  if (wasLValue)
    return csl::LValueAttr::get(fieldValue);

  return fieldValue;
}

// FieldAccessOp::getCanonicalizationPatterns is defined in
// TypedCSLCanonicalize.cpp

//===----------------------------------------------------------------------===//
// TupleIndexAccessOp
//===----------------------------------------------------------------------===//

LogicalResult AwaitParallelOp::verify() {
  SmallVector<Type> resultTypes;
  for (Value future : getFutures())
    if (Type resultType =
            mlir::cast<csl::FutureType>(future.getType()).getResultType())
      resultTypes.push_back(resultType);
  Value result = getResult();
  if (!result)
    return success();
  auto tupleType = mlir::cast<csl::StructType>(result.getType());
  if (!tupleType.isTuple() || tupleType.getFieldTypes() != ArrayRef(resultTypes))
    return emitOpError("result type ")
           << tupleType << " must be the tuple of the callee result types";
  return success();
}

LogicalResult TupleIndexAccessOp::verify() {
  Type baseType = getBase().getType();

  // Extract the struct type from lvalue or direct rvalue
  csl::StructType structType;
  bool isLvalue = false;
  bool isConst = false;

  if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(baseType)) {
    structType = mlir::dyn_cast<csl::StructType>(lvalueType.getValueType());
    isLvalue = true;
    isConst = lvalueType.getIsConst();
  } else {
    structType = mlir::dyn_cast<csl::StructType>(baseType);
  }

  if (!structType) {
    return emitOpError("operand must be a struct/tuple type or lvalue thereof");
  }

  int64_t index = getIndex();
  if (index < 0 || static_cast<size_t>(index) >= structType.getNumFields()) {
    return emitOpError("index ")
           << index << " is out of bounds for struct with "
           << structType.getNumFields() << " fields";
  }

  Type expectedFieldType = structType.getFieldType(static_cast<size_t>(index));
  Type resultType = getResult().getType();

  if (isLvalue) {
    // Result should be lvalue with matching constness
    auto resultLValueType = mlir::dyn_cast<csl::LValueType>(resultType);
    if (!resultLValueType) {
      return emitOpError("result must be an lvalue when base is an lvalue");
    }
    if (resultLValueType.getValueType() != expectedFieldType) {
      return emitOpError("result lvalue has value type ")
             << resultLValueType.getValueType() << " but expected "
             << expectedFieldType;
    }
    if (resultLValueType.getIsConst() != isConst) {
      return emitOpError(
          "result lvalue constness must match base lvalue constness");
    }
  } else {
    // Result should be the field type directly (rvalue)
    if (resultType != expectedFieldType) {
      return emitOpError("result type ")
             << resultType << " but expected " << expectedFieldType;
    }
  }

  return success();
}

OpFoldResult TupleIndexAccessOp::fold(FoldAdaptor adaptor) {
  // Could potentially fold if base is a constant struct value
  // For now, no folding
  return nullptr;
}

// TupleIndexAccessOp::getCanonicalizationPatterns is defined in
// TypedCSLCanonicalize.cpp

//===----------------------------------------------------------------------===//
// ArrayIndexAccessOp
//===----------------------------------------------------------------------===//

LogicalResult ArrayIndexAccessOp::verify() {
  Type baseType = getBase().getType();

  // Extract the array type from lvalue or direct rvalue
  csl::ArrayType arrayType;
  bool isLvalue = false;
  bool isConst = false;

  if (auto lvalueType = mlir::dyn_cast<csl::LValueType>(baseType)) {
    arrayType = mlir::dyn_cast<csl::ArrayType>(lvalueType.getValueType());
    isLvalue = true;
    isConst = lvalueType.getIsConst();
  } else {
    arrayType = mlir::dyn_cast<csl::ArrayType>(baseType);
  }

  if (!arrayType) {
    return emitOpError("operand must be an array type or lvalue thereof");
  }

  ArrayRef<int64_t> shape = arrayType.getShape();
  size_t numIndices = getIndices().size();

  if (numIndices == 0) {
    return emitOpError("must have at least one index");
  }

  if (numIndices > shape.size()) {
    return emitOpError("too many indices (")
           << numIndices << ") for array with rank " << shape.size();
  }

  // Determine expected element type based on number of indices
  Type expectedElemType;
  if (numIndices == shape.size()) {
    // Full indexing: result is element type
    expectedElemType = arrayType.getElementType();
  } else {
    // Partial indexing: result is sub-array
    SmallVector<int64_t> remainingShape(shape.begin() + numIndices,
                                        shape.end());
    expectedElemType = csl::ArrayType::get(getContext(), remainingShape,
                                           arrayType.getElementType());
  }

  Type resultType = getResult().getType();

  if (isLvalue) {
    // Result should be lvalue with matching constness
    auto resultLValueType = mlir::dyn_cast<csl::LValueType>(resultType);
    if (!resultLValueType) {
      return emitOpError("result must be an lvalue when base is an lvalue");
    }
    if (resultLValueType.getValueType() != expectedElemType) {
      return emitOpError("result lvalue has value type ")
             << resultLValueType.getValueType() << " but expected "
             << expectedElemType;
    }
    if (resultLValueType.getIsConst() != isConst) {
      return emitOpError(
          "result lvalue constness must match base lvalue constness");
    }
  } else {
    // Result should be the element type directly (rvalue)
    if (resultType != expectedElemType) {
      return emitOpError("result type ")
             << resultType << " but expected " << expectedElemType;
    }
  }

  return success();
}

OpFoldResult ArrayIndexAccessOp::fold(FoldAdaptor adaptor) {
  // Could potentially fold if base and indices are constants
  // For now, no folding
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PtrIndexAccessOp
//===----------------------------------------------------------------------===//

LogicalResult PtrIndexAccessOp::verify() {
  csl::PointerType ptrType = getBase().getType();

  if (!ptrType.getIsManyPtr()) {
    return emitOpError("cannot index into single-item pointer; use .* to "
                       "dereference");
  }

  csl::LValueType resultLValueType = getResult().getType();
  if (resultLValueType.getValueType() != ptrType.getPointeeType()) {
    return emitOpError("result lvalue has value type ")
           << resultLValueType.getValueType() << " but expected "
           << ptrType.getPointeeType();
  }

  if (resultLValueType.getIsConst() != ptrType.getIsConst()) {
    return emitOpError("result lvalue constness must match pointer constness");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

LogicalResult ForOp::verify() {
  // Verify the body region has exactly one block
  if (getBody().empty())
    return emitOpError("body region must have at least one block");

  Block &bodyBlock = getBody().front();

  // Verify the body block has at least one argument (the element)
  if (bodyBlock.getNumArguments() < 1)
    return emitOpError("body block must have at least one argument (element)");

  // If index_name is set, verify the body block has two arguments
  if (hasIndex() && bodyBlock.getNumArguments() < 2)
    return emitOpError("body block must have two arguments when index_name is "
                       "set (element and index)");

  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::verify() {
  // Verify the condition region has exactly one block
  if (getCondRegion().empty())
    return emitOpError("condition region must have at least one block");

  // Condition block should not have any arguments
  Block &condBlock = getCondRegion().front();
  if (condBlock.getNumArguments() != 0)
    return emitOpError("condition block must not have any arguments");

  // Condition region must be terminated by CondYieldOp
  auto condYieldOp = dyn_cast<CondYieldOp>(condBlock.getTerminator());
  if (!condYieldOp)
    return emitOpError("condition region must be terminated by cond_yield");

  // Verify the body region has exactly one block
  if (getBody().empty())
    return emitOpError("body region must have at least one block");

  // Body block should not have any arguments
  Block &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != 0)
    return emitOpError("body block must not have any arguments");

  // Continuation region is optional, but if present must have at most one block
  if (!getContinuation().empty()) {
    if (getContinuation().getBlocks().size() > 1)
      return emitOpError("continuation region must have at most one block");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LabeledBlockOp
//===----------------------------------------------------------------------===//

ParseResult LabeledBlockOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr labelAttr;
  if (parser.parseAttribute(labelAttr, getLabelAttrName(result.name),
                            result.attributes))
    return failure();

  // Parse optional result type: -> si32
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
  }

  // Parse body region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void LabeledBlockOp::print(OpAsmPrinter &p) {
  p << " " << getLabelAttr();
  if (getNumResults() > 0)
    p << " -> " << getResult().getType();
  p << " ";
  p.printRegion(getBody());
  p.printOptionalAttrDict((*this)->getAttrs(), {getLabelAttrName()});
}

LogicalResult LabeledBlockOp::verify() {
  if (getLabel().empty())
    return emitOpError("label cannot be empty");

  bool hasResult = getNumResults() > 0;
  StringRef myLabel = getLabel();

  // Collect all breaks targeting this block
  SmallVector<typed_csl::BreakOp> targetingBreaks;
  getBody().walk([&](typed_csl::BreakOp breakOp) {
    if (breakOp.getTargetLabel() && *breakOp.getTargetLabel() == myLabel)
      targetingBreaks.push_back(breakOp);
  });

  // Verify break values match result presence
  for (auto breakOp : targetingBreaks) {
    if (hasResult) {
      if (!breakOp.getValue()) {
        return breakOp.emitOpError("break targeting valued labeled block must "
                                   "provide a value");
      }
      if (breakOp.getValue().getType() != getResult().getType()) {
        return breakOp.emitOpError("break value type ")
               << breakOp.getValue().getType()
               << " does not match labeled block result type "
               << getResult().getType();
      }
    } else {
      if (breakOp.getValue()) {
        return breakOp.emitOpError(
            "break targeting void labeled block must not provide a value");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

LogicalResult BreakOp::verify() {
  // Value can only be provided with a target label
  if (getValue() && !getTargetLabel())
    return emitOpError("break with value requires a target label");

  return success();
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

LogicalResult ContinueOp::verify() {
  // Find the target - continue cannot target labeled blocks
  Operation *parent = getOperation()->getParentOp();
  while (parent) {
    if (auto labeledBlock = dyn_cast<typed_csl::LabeledBlockOp>(parent)) {
      if (getTargetLabel() && *getTargetLabel() == labeledBlock.getLabel()) {
        return emitOpError("continue cannot target labeled block :")
               << *getTargetLabel() << "; only loops can be continued";
      }
    }

    if (isa<typed_csl::ForOp, typed_csl::WhileOp>(parent)) {
      if (!getTargetLabel())
        return success(); // Unlabeled continue targets innermost loop

      StringAttr loopLabel;
      if (auto forOp = dyn_cast<typed_csl::ForOp>(parent))
        loopLabel = forOp.getLabelAttr();
      else if (auto whileOp = dyn_cast<typed_csl::WhileOp>(parent))
        loopLabel = whileOp.getLabelAttr();

      if (loopLabel && loopLabel.getValue() == *getTargetLabel())
        return success();
    }

    parent = parent->getParentOp();
  }

  if (getTargetLabel())
    return emitOpError("continue label :") << *getTargetLabel() << " not found";
  return emitOpError("continue statement not within a loop");
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

LogicalResult InlineAsmOp::verify() {
  if (getOutputNames().size() != getOutputs().size())
    return emitOpError("output_names count (")
           << getOutputNames().size() << ") must match outputs count ("
           << getOutputs().size() << ")";
  if (getOutputConstraints().size() != getOutputs().size())
    return emitOpError("output_constraints count (")
           << getOutputConstraints().size() << ") must match outputs count ("
           << getOutputs().size() << ")";
  if (getInputNames().size() != getInputs().size())
    return emitOpError("input_names count (")
           << getInputNames().size() << ") must match inputs count ("
           << getInputs().size() << ")";
  if (getInputConstraints().size() != getInputs().size())
    return emitOpError("input_constraints count (")
           << getInputConstraints().size() << ") must match inputs count ("
           << getInputs().size() << ")";

  // All output operands must be lvalue types
  for (auto [idx, output] : llvm::enumerate(getOutputs())) {
    if (!mlir::isa<csl::LValueType>(output.getType()))
      return emitOpError("output operand #")
             << idx << " must be an lvalue type, but got " << output.getType();
  }

  return success();
}
