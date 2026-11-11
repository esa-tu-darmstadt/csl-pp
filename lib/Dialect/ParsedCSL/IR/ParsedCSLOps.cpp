//===- ParsedCSLOps.cpp - ParsedCSL Operation Implementations --*- C++ -*-===//
//
// Implementation of ParsedCSL dialect operations.
//
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSLFolders.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <variant>

using namespace mlir;
using namespace mlir::parsed_csl;
using namespace mlir::csl;

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd interface definitions
//===----------------------------------------------------------------------===//

#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSLInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// BuiltinCallOp
//===----------------------------------------------------------------------===//

void BuiltinCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto builtinKind = resolveBuiltin();
  if (!builtinKind) {
    // Unknown builtin — conservatively assume side effects
    using namespace mlir::MemoryEffects;
    effects.emplace_back(Write::get(), 0, false,
                         SideEffects::DefaultResource::get());
    return;
  }

  // Check if this builtin is pure using the descriptor
  const csl::BuiltinDescriptor &desc =
      csl::BuiltinDescriptor::get(*builtinKind, getContext());

  bool inComptimeContext =
      getOperation()->getParentWithTrait<OpTrait::ComptimeScope>() != nullptr;
  if (desc.mustKeepInOutput(inComptimeContext)) {
    using namespace mlir::MemoryEffects;
    effects.emplace_back(Write::get(), 0, false,
                         SideEffects::DefaultResource::get());
  }
  // Pure builtins have no effects, so nothing to add
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace mlir::parsed_csl {

/// Get the inferred type for a value using the InferredTypeInterface if
/// present. Returns failure if no inferred type is set or the operation does
/// not implement the interface.
FailureOr<Type> getInferredType(Value value) {
  if (!value)
    return failure();

  // Handle result values - check if defining op has InferredTypeInterface
  if (Operation *defOp = value.getDefiningOp()) {
    if (auto inferredTypeOp = dyn_cast<InferredTypeInterface>(defOp)) {
      std::optional<Type> type = inferredTypeOp.getInferredType();
      if (!type)
        return failure();
      return *type;
    }
    return failure();
  }

  // Handle block arguments - check if parent op is FuncOp with function_type
  BlockArgument blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return failure();

  Block *parentBlock = blockArg.getOwner();
  if (!parentBlock)
    return failure();

  Operation *parentOp = parentBlock->getParentOp();
  if (!parentOp)
    return failure();

  if (auto funcOp = dyn_cast<FuncOp>(parentOp)) {
    FunctionType funcType = funcOp.getFunctionType();
    unsigned argIndex = blockArg.getArgNumber();
    if (argIndex >= funcType.getNumInputs())
      return failure();

    return (Type)csl::LValueType::get(funcType.getInput(argIndex),
                                      /*isConst=*/true);
  }

  // Handle ForOp block arguments (element and optional index)
  if (auto forOp = dyn_cast<ForOp>(parentOp)) {
    unsigned argIndex = blockArg.getArgNumber();
    if (argIndex == 0) {
      // Element argument
      if (auto elemType = forOp.getInferredElemType())
        return *elemType;
    } else if (argIndex == 1) {
      // Index argument
      if (auto indexType = forOp.getInferredIndexType())
        return *indexType;
    }
    return failure();
  }

  return failure();
}

} // namespace mlir::parsed_csl

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
//   // Parse: value attr-dict
//   Attribute valueAttr;

//   if (parser.parseAttribute(valueAttr) ||
//       parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   result.addAttribute("value", valueAttr);
//   result.addTypes(UnresolvedType::get(parser.getContext()));
//   return success();
// }

// void ConstantOp::print(OpAsmPrinter &p) {
//   // Print: value attr-dict
//   p << " ";
//   Type t = getValue().getType();
//   if (csl::isAnyRuntimeIntType(t) || csl::isAnyRuntimeFloatType(t))
//     p.printAttribute(getValue());
//   else
//     p.printAttributeWithoutType(getValue());
//   p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
// }

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  // Constants always fold to their value attribute
  return getValue();
}

//===----------------------------------------------------------------------===//
// BuiltinTypeOp
//===----------------------------------------------------------------------===//

OpFoldResult BuiltinTypeOp::fold(FoldAdaptor adaptor) {
  // Builtin type ops are constant-like and always fold to their type attribute
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// BinaryOp - Folder and Helpers
//===----------------------------------------------------------------------===//

OpFoldResult BinaryOp::fold(FoldAdaptor adaptor) {
  // Both operands must be constant
  auto lhsAttr = llvm::dyn_cast_if_present<TypedAttr>(adaptor.getLhs());
  auto rhsAttr = llvm::dyn_cast_if_present<TypedAttr>(adaptor.getRhs());

  if (!lhsAttr || !rhsAttr)
    return nullptr;

  // Try to evaluate
  auto result = csl::inferBinaryOp(getKind(), csl::TypeOrAttr(lhsAttr),
                                   csl::TypeOrAttr(rhsAttr));
  if (failed(result))
    return nullptr;

  return std::get<TypedAttr>(result->result);
}

//===----------------------------------------------------------------------===//
// UnaryOp - Folder and Helpers
//===----------------------------------------------------------------------===//

OpFoldResult UnaryOp::fold(FoldAdaptor adaptor) {
  // Operand must be constant
  auto operandAttr = llvm::dyn_cast_if_present<TypedAttr>(adaptor.getOperand());

  if (!operandAttr)
    return nullptr;

  // Try to evaluate
  auto result = csl::evaluateUnary(getKind(), operandAttr);
  if (failed(result))
    return nullptr;

  return *result;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  // Source must be constant, target type must be constant
  auto sourceAttr = llvm::dyn_cast_if_present<TypedAttr>(adaptor.getSource());
  auto targetTypeAttr =
      llvm::dyn_cast_if_present<csl::TypeAttr>(adaptor.getTargetType());

  if (!sourceAttr || !targetTypeAttr)
    return nullptr;

  Type targetType = targetTypeAttr.getValue();

  // Try to evaluate
  auto result = csl::evaluateCast(sourceAttr, targetType);
  if (failed(result))
    return nullptr;

  return *result;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // Can't fold region-based select - use canonicalizer for constant conditions
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value input,
                     ValueRange caseOperands,
                     ArrayRef<int32_t> caseOperandSegments, unsigned numCases) {
  result.addOperands(input);
  result.addOperands(caseOperands);
  result.addAttribute(getCaseOperandSegmentsAttrName(result.name),
                      builder.getDenseI32ArrayAttr(caseOperandSegments));
  result.addTypes(UnresolvedType::get(builder.getContext()));

  // Add else region first (MLIR requires variadic regions last)
  result.addRegion();
  // Add empty regions for each case
  for (unsigned i = 0; i < numCases; ++i) {
    result.addRegion();
  }
}

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
  Type unresolvedType = UnresolvedType::get(parser.getContext());

  // Parse input operand
  OpAsmParser::UnresolvedOperand inputOperand;
  if (parser.parseLParen() || parser.parseOperand(inputOperand) ||
      parser.parseRParen())
    return failure();

  if (parser.resolveOperand(inputOperand, unresolvedType, result.operands))
    return failure();

  // Parse "cases" keyword and case operands: cases([[%a, %b], [%c]])
  if (parser.parseKeyword("cases"))
    return failure();

  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> caseOperandsList;
  if (parser.parseLParen())
    return failure();

  // Parse outer list
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOptionalRSquare().failed()) {
    // Non-empty outer list
    do {
      SmallVector<OpAsmParser::UnresolvedOperand> thisCaseOperands;
      if (parser.parseLSquare())
        return failure();

      if (parser.parseOptionalRSquare().failed()) {
        // Non-empty inner list
        do {
          OpAsmParser::UnresolvedOperand operand;
          if (parser.parseOperand(operand))
            return failure();
          thisCaseOperands.push_back(operand);
        } while (succeeded(parser.parseOptionalComma()));

        if (parser.parseRSquare())
          return failure();
      }
      caseOperandsList.push_back(std::move(thisCaseOperands));
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare())
      return failure();
  }

  if (parser.parseRParen())
    return failure();

  // Build segments and resolve operands
  SmallVector<int32_t> segments;
  for (auto &caseOps : caseOperandsList) {
    segments.push_back(caseOps.size());
    for (auto &op : caseOps) {
      if (parser.resolveOperand(op, unresolvedType, result.operands))
        return failure();
    }
  }

  result.addAttribute(getCaseOperandSegmentsAttrName(result.name),
                      DenseI32ArrayAttr::get(parser.getContext(), segments));

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

  // Add result type
  result.addTypes(unresolvedType);

  return success();
}

void SwitchOp::print(OpAsmPrinter &p) {
  p << "(" << getInput() << ")";

  // Print case operands: cases([[%a, %b], [%c]])
  p << " cases([";
  auto segments = getCaseOperandSegments();
  unsigned offset = 0;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (i > 0)
      p << ", ";
    p << "[";
    for (int32_t j = 0; j < segments[i]; ++j) {
      if (j > 0)
        p << ", ";
      p << getCaseOperands()[offset + j];
    }
    p << "]";
    offset += segments[i];
  }
  p << "])";

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
                          /*elidedAttrs=*/{getCaseOperandSegmentsAttrName(),
                                           getCaseValuesAttrName()});
}

LogicalResult SwitchOp::verify() {
  // Verify segments match number of case regions
  auto segments = getCaseOperandSegments();
  if (segments.size() != getCaseRegions().size()) {
    return emitOpError("number of case operand segments (")
           << segments.size() << ") must match number of case regions ("
           << getCaseRegions().size() << ")";
  }

  // Verify total operands match
  unsigned totalOperands = 0;
  for (int32_t seg : segments)
    totalOperands += seg;

  if (totalOperands != getCaseOperands().size()) {
    return emitOpError("total case operands (")
           << getCaseOperands().size() << ") does not match sum of segments ("
           << totalOperands << ")";
  }

  return success();
}

OpFoldResult SwitchOp::fold(FoldAdaptor adaptor) {
  // Can't fold switch with regions - use canonicalizer for constant input
  return nullptr;
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

// ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &result) {
//   // Parse: "varName" (dynamicType(%type))? (init(%init))? (const)?
//   (comptime)?
//   // attr-dict

//   // Parse variable name
//   StringAttr nameAttr;
//   if (parser.parseAttribute(nameAttr))
//     return failure();
//   result.addAttribute("sym_name", nameAttr);

//   // Parse optional dynamicType
//   OpAsmParser::UnresolvedOperand dynamicTypeOperand;
//   bool hasDynamicType = false;
//   if (succeeded(parser.parseOptionalKeyword("dynamicType"))) {
//     if (parser.parseLParen() || parser.parseOperand(dynamicTypeOperand) ||
//         parser.parseRParen())
//       return failure();
//     hasDynamicType = true;
//   }

//   // Parse optional initializer
//   OpAsmParser::UnresolvedOperand initializerOperand;
//   bool hasInitializer = false;
//   if (succeeded(parser.parseOptionalKeyword("init"))) {
//     if (parser.parseLParen() || parser.parseOperand(initializerOperand) ||
//         parser.parseRParen())
//       return failure();
//     hasInitializer = true;
//   }

//   // Parse optional const keyword
//   bool isConst = succeeded(parser.parseOptionalKeyword("const"));
//   if (isConst)
//     result.addAttribute("is_const", parser.getBuilder().getUnitAttr());

//   // Parse optional comptime keyword
//   bool isComptime = succeeded(parser.parseOptionalKeyword("comptime"));
//   if (isComptime)
//     result.addAttribute("is_comptime", parser.getBuilder().getUnitAttr());

//   // Parse optional attributes
//   if (parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   // Resolve operands
//   Type unresolvedType = parsed_csl::UnresolvedType::get(parser.getContext());
//   if (hasDynamicType) {
//     if (parser.resolveOperand(dynamicTypeOperand, unresolvedType,
//                               result.operands))
//       return failure();
//   }

//   if (hasInitializer) {
//     if (parser.resolveOperand(initializerOperand, unresolvedType,
//                               result.operands))
//       return failure();
//   }

//   // Add operand segment sizes for AttrSizedOperandSegments
//   result.addAttribute("operandSegmentSizes",
//                       parser.getBuilder().getDenseI32ArrayAttr(
//                           {hasDynamicType ? 1 : 0, hasInitializer ? 1 : 0}));

//   // Infer result type from isConst
//   Type resultType = csl::LValueType::get(unresolvedType, isConst);
//   result.addTypes(resultType);

//   return success();
// }

// void VariableOp::print(OpAsmPrinter &p) {
//   // Print: "varName" (dynamicType(%type))? (init(%init))? (const)?
//   (comptime)?
//   // attr-dict

//   p << " ";
//   p.printAttribute(getNameAttr());

//   // Print optional dynamicType
//   if (getDynamicType()) {
//     p << " dynamicType(" << getDynamicType() << ")";
//   }

//   // Print optional initializer
//   if (getInitializer()) {
//     p << " init(" << getInitializer() << ")";
//   }

//   // Print optional const
//   if (getIsConst()) {
//     p << " const";
//   }

//   // Print optional comptime
//   if (getIsComptime()) {
//     p << " comptime";
//   }

//   // Print attributes (excluding the ones we already printed)
//   p.printOptionalAttrDict((*this)->getAttrs(),
//                           /*elidedAttrs=*/{"sym_name", "is_const",
//                                            "is_comptime",
//                                            "operandSegmentSizes"});
// }
/// Whether the address of the storage behind `lvalue` is taken, directly or
/// through a field or index of it.
static bool isAddressTaken(Value lvalue) {
  for (Operation *user : lvalue.getUsers()) {
    if (mlir::isa<AddressOfOp>(user))
      return true;
    if (mlir::isa<FieldAccessOp, IndexAccessOp, LocOp>(user) &&
        isAddressTaken(user->getResult(0)))
      return true;
  }
  return false;
}

OpFoldResult VariableOp::fold(FoldAdaptor adaptor) {
  // Only const variables can be folded
  if (!getIsConst())
    return nullptr;

  // Taking the address needs the variable's storage, not its value.
  if (isAddressTaken(getResult()))
    return nullptr;

  // Must have an initializer to fold
  auto initAttr =
      llvm::dyn_cast_if_present<TypedAttr>(adaptor.getInitializer());
  if (!initAttr)
    return nullptr;

  // Get the expected type from inferred_type (which is an LValueType)
  auto inferredType = getInferredTypeAttr();
  if (!inferredType)
    return nullptr;

  auto lvalueType = mlir::dyn_cast<csl::LValueType>(inferredType.getValue());
  if (!lvalueType)
    return nullptr;

  Type expectedType = lvalueType.getValueType();

  // Try to cast the initializer to the expected type (implicit cast)
  if (initAttr.getType() != expectedType) {
    auto castResult = csl::evaluateImplicitCast(initAttr, expectedType);
    if (failed(castResult))
      return nullptr; // Cast failed, don't fold
    initAttr = *castResult;
  }

  // Return as const LValueAttr
  return csl::LValueAttr::get(initAttr);
}

LogicalResult VariableOp::verify() {
  if (getIsComptime()) {
    // Comptime variables must have an initializer
    if (!getInitializer()) {
      return emitOpError("comptime variable must have an initializer");
    }

    // Comptime variables must be initialized either with a comptime expression
    // (after parsing) or a constant op (after comptime evaluation)
    if (!mlir::isa<ComptimeExprOp>(getInitializer().getDefiningOp()) &&
        !mlir::isa<ConstantOp>(getInitializer().getDefiningOp())) {
      // FIXME: During pattern application, this could also be a cast of a
      // constant... umpf
      // return emitOpError("comptime variable must be initialized with a "
      //                    "comptime expression");
    }

    // Ensure comptime variables are not nested in comptime scope
    if ((*this)->getParentOfType<ComptimeExprOp>()) {
      return emitOpError("'comptime var' is redundant in comptime scope");
    }
  }

  if (!getInitializer() && !getDynamicType()) {
    return emitOpError("variable must have either a dynamic type or an "
                       "initializer");
  }

  return success();
}

void VariableOp::getEffects(
    llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  using namespace mlir::MemoryEffects;
  // Comptime variables have no runtime effects — they are fully evaluated
  // during comptime and should be eligible for DCE.
  if (getIsComptime())
    return;
  effects.emplace_back(Allocate::get(), getOperation()->getOpResult(0), 0,
                       false,
                       SideEffects::AutomaticAllocationScopeResource::get());
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verify() {
  bool hasType = (getDynamicType() != nullptr);
  bool hasInit = hasInitializer();

  // Export and extern are mutually exclusive
  if (getIsExport() && getIsExtern()) {
    return emitOpError("'export' and 'extern' are mutually exclusive");
  }

  // Extern globals must have a type
  if (getIsExtern() && !hasType) {
    return emitOpError("extern global variable must have a type");
  }

  // Extern globals cannot have an initializer
  if (getIsExtern() && hasInit) {
    return emitOpError("extern global variable cannot have an initializer");
  }

  // Must have either dynamicType or initializer
  if (!hasType && !hasInit) {
    return emitOpError("global variable must have either a type or an "
                       "initializer");
  }

  // Const globals must have an initializer (unless extern or unbound param)
  if (getIsConst() && !hasInit && !getIsExtern() && !getIsUnboundParam()) {
    return emitOpError("const global variable must have an initializer");
  }

  // If initRegion is non-empty, it must have exactly one block
  if (!getInitRegion().empty() && !getInitRegion().hasOneBlock()) {
    return emitOpError("init region must have exactly one block");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GenericFuncOp
//===----------------------------------------------------------------------===//

LogicalResult GenericFuncOp::verify() {
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
// FuncOp
//===----------------------------------------------------------------------===//

LogicalResult FuncOp::verify() {
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
// GetSymbolOp
//===----------------------------------------------------------------------===//

LogicalResult
GetSymbolOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol = nullptr;
  if (auto moduleId = getLexicalModuleIdentifier()) {
    auto unit = (*this)->getParentOfType<csl::CompilationUnitOp>();
    if (auto module = findModuleById(unit, static_cast<uint64_t>(*moduleId)))
      symbol = symbolTable.lookupSymbolIn(module, getNameAttr());
  } else {
    symbol = symbolTable.lookupNearestSymbolFrom(*this, getNameAttr());
  }
  auto globalOp = dyn_cast_if_present<GlobalOp>(symbol);
  auto genericFunc = dyn_cast_if_present<GenericFuncOp>(symbol);
  auto func = dyn_cast_if_present<FuncOp>(symbol);
  if (!symbol)
    return emitError() << "use of undeclared identifier '" << getName() << "'";
  if (!globalOp && !genericFunc && !func)
    return emitError() << "'" << getName()
                       << "' is not a variable, constant or function";
  return success();
}

//===----------------------------------------------------------------------===//
// ReadOp
//===----------------------------------------------------------------------===//

OpFoldResult ReadOp::fold(FoldAdaptor adaptor) {
  // Once comptime evaluation has inferred the types, a read of a value that
  // is not a location is that value.
  if (FailureOr<Type> sourceType = parsed_csl::getInferredType(getSource());
      succeeded(sourceType) && !mlir::isa<csl::LValueType>(*sourceType))
    return getSource();

  Attribute source = adaptor.getSource();
  if (!source)
    return nullptr;
  // A const location always holds the same value. A mutable one may change,
  // so its read stays where it is.
  if (auto lvalueAttr = llvm::dyn_cast<LValueAttr>(source)) {
    if (!lvalueAttr.getType().getIsConst())
      return nullptr;
    return lvalueAttr.getRvalue();
  }
  // A global reference denotes a location whose content is not known here.
  if (auto typed = llvm::dyn_cast<TypedAttr>(source);
      typed && mlir::isa<csl::LValueType>(typed.getType()))
    return nullptr;
  // Anything else is not a location and reads as itself.
  return source;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

OpFoldResult LoadOp::fold(FoldAdaptor adaptor) {
  // A const location always holds the same value.
  auto lvalueAttr = llvm::dyn_cast_if_present<LValueAttr>(adaptor.getLvalue());
  if (!lvalueAttr || !lvalueAttr.getType().getIsConst())
    return nullptr;
  return lvalueAttr.getRvalue();
}

//===----------------------------------------------------------------------===//
// GenericFuncOp
//===----------------------------------------------------------------------===//

Block::BlockArgListType GenericFuncOp::getBodyArgs() {
  assert(!getBody().empty() && "GenericFuncOp body region is empty");
  return getBody().front().getArguments();
}

Block::BlockArgListType GenericFuncOp::getTypeRegionArgs() {
  assert(!getTypeRegion().empty() && "GenericFuncOp type region is empty");
  return getTypeRegion().front().getArguments();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

unsigned FuncOp::getNumParams() { return getFunctionType().getNumInputs(); }

ArrayRef<Type> FuncOp::getParamTypes() { return getFunctionType().getInputs(); }

Type FuncOp::getBodyReturnType() {
  if (getIsAsync()) {
    auto asyncResultType = getAsyncResultType();
    return asyncResultType ? *asyncResultType
                           : csl::VoidType::get(getContext());
  }
  return getFunctionType().getResult(0);
}

Type FuncOp::getCallableReturnType() { return getFunctionType().getResult(0); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // ReturnOp can be in either FuncOp (after instantiation) or GenericFuncOp
  // (during parsing). We can't fully verify return type compatibility during
  // parsing since types aren't resolved yet.

  auto funcOp = (*this)->getParentOfType<FuncOp>();
  auto genericFuncOp = (*this)->getParentOfType<GenericFuncOp>();

  if (!funcOp && !genericFuncOp)
    return emitOpError(
        "must be nested in a parsed_csl.func or parsed_csl.generic_func");

  // For GenericFuncOp, we can't verify return type compatibility yet
  // (types are computed in type_region during instantiation)
  if (genericFuncOp)
    return success();

  // For FuncOp, verify against the body return type (which for async
  // functions is async_result_type, not the CoroFrameType in function_type).
  FunctionType fnType = funcOp.getFunctionType();
  size_t numOperands = getOperands().size();

  if (fnType.getNumResults() != 1) {
    return emitOpError(
        "function must have exactly one return type (can be void)");
  }

  bool isVoidReturn = mlir::isa<VoidType>(funcOp.getBodyReturnType());

  if (!isVoidReturn) {
    // Function has a return type, must return exactly one value
    if (numOperands != 1) {
      return emitOpError("must return exactly one value when function has a "
                         "return type, got ")
             << numOperands << " values";
    }
  } else {
    // Function has no return type (void), must return no values
    if (numOperands != 0) {
      return emitOpError("must return no values when function has no return "
                         "type, got ")
             << numOperands << " values";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

// ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
//   // Parse callee
//   StringAttr callee;
//   if (parser.parseSymbolName(callee))
//     return failure();
//   result.addAttribute("callee", callee);

//   // Parse operand list
//   SmallVector<OpAsmParser::UnresolvedOperand> operands;
//   if (parser.parseLParen() || parser.parseOperandList(operands) ||
//       parser.parseRParen())
//     return failure();

//   // Parse optional attributes
//   if (parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   // Resolve operands to UnresolvedType
//   Type unresolvedType = parsed_csl::UnresolvedType::get(parser.getContext());
//   if (parser.resolveOperands(operands, unresolvedType, result.operands))
//     return failure();

//   // Add optional result (call might be void)
//   // We'll assume if we're parsing a result, the caller expects one
//   // The parser context will tell us if we need a result
//   if (parser.getNumResults() > 0) {
//     result.addTypes(unresolvedType);
//   }

//   return success();
// }

// void CallOp::print(OpAsmPrinter &p) {
//   p << " ";
//   p.printSymbolName(getCallee());
//   p << "(";
//   p.printOperands(getOperands());
//   p << ")";
//   p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"callee"});
// }

Speculation::Speculatability CallOp::getSpeculatability() {
  return Speculation::Speculatability::NotSpeculatable;
}

void CallOp::getEffects(
    ::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
        &effects) {
  using namespace mlir::MemoryEffects;
  effects.emplace_back(Write::get());
  effects.emplace_back(Read::get());
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

// ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
//   // Parse: condition then-region [else else-region]

//   // Parse condition operand
//   OpAsmParser::UnresolvedOperand condition;
//   if (parser.parseOperand(condition))
//     return failure();

//   // Resolve condition to unresolved type
//   Type unresolvedType = parsed_csl::UnresolvedType::get(parser.getContext());
//   if (parser.resolveOperand(condition, unresolvedType, result.operands))
//     return failure();

//   // Parse then region
//   Region *thenRegion = result.addRegion();
//   if (parser.parseRegion(*thenRegion, /*args=*/{}, /*argTypes=*/{}))
//     return failure();

//   // Parse optional else region
//   if (succeeded(parser.parseOptionalKeyword("else"))) {
//     Region *elseRegion = result.addRegion();
//     if (parser.parseRegion(*elseRegion, /*args=*/{}, /*argTypes=*/{}))
//       return failure();
//   }

//   // Parse optional attributes
//   if (parser.parseOptionalAttrDict(result.attributes))
//     return failure();

//   return success();
// }

// void IfOp::print(OpAsmPrinter &p) {
//   p << " " << getCondition();
//   p << " ";
//   p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);

//   // Print else region if present
//   if (!getElseRegion().empty()) {
//     p << " else ";
//     p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
//   }

//   p.printOptionalAttrDict((*this)->getAttrs());
// }

//===----------------------------------------------------------------------===//
// NamedStructTypeOp
//===----------------------------------------------------------------------===//

ParseResult NamedStructTypeOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  // Parse: loc(...) field1 : %type1, field2 : %type2, ...

  // Parse def_location
  LocationAttr defLocation;
  if (parser.parseAttribute(defLocation))
    return failure();
  result.addAttribute("def_location", defLocation);

  // Parse fields: name : operand, name : operand, ...
  SmallVector<StringRef> fieldNames;
  SmallVector<OpAsmParser::UnresolvedOperand> fieldTypeOperands;
  Type unresolvedType = parsed_csl::UnresolvedType::get(parser.getContext());

  // Parse at least one field
  StringRef firstName;
  OpAsmParser::UnresolvedOperand firstTypeOperand;

  if (parser.parseKeyword(&firstName) || parser.parseColon() ||
      parser.parseOperand(firstTypeOperand))
    return failure();

  fieldNames.push_back(firstName);
  fieldTypeOperands.push_back(firstTypeOperand);

  // Parse remaining fields
  while (succeeded(parser.parseOptionalComma())) {
    StringRef name;
    OpAsmParser::UnresolvedOperand typeOperand;

    if (parser.parseKeyword(&name) || parser.parseColon() ||
        parser.parseOperand(typeOperand))
      return failure();

    fieldNames.push_back(name);
    fieldTypeOperands.push_back(typeOperand);
  }

  // Build field_names array attribute
  SmallVector<Attribute> nameAttrs;
  for (auto name : fieldNames)
    nameAttrs.push_back(parser.getBuilder().getStringAttr(name));
  result.addAttribute("field_names",
                      parser.getBuilder().getArrayAttr(nameAttrs));

  // Resolve field type operands
  if (parser.resolveOperands(fieldTypeOperands, unresolvedType,
                             result.operands))
    return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Add result type
  result.addTypes(unresolvedType);

  return success();
}

void NamedStructTypeOp::print(OpAsmPrinter &p) {
  // Print: loc(...) field1 : %type1, field2 : %type2, ...
  p << " ";
  p.printAttribute(getDefLocation());

  auto fieldNames = getFieldNames();
  auto fieldTypes = getFieldTypes();

  for (size_t i = 0; i < fieldNames.size(); ++i) {
    if (i > 0)
      p << ",";
    p << " " << mlir::cast<StringAttr>(fieldNames[i]).getValue();
    p << " : " << fieldTypes[i];
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"field_names", "def_location"});
}

LogicalResult NamedStructTypeOp::verify() {
  // Verify field_names and field_types have the same size
  auto fieldNames = getFieldNames();
  auto fieldTypes = getFieldTypes();

  if (fieldNames.size() != fieldTypes.size()) {
    return emitOpError(
               "field_names and field_types must have the same size, got ")
           << fieldNames.size() << " names and " << fieldTypes.size()
           << " types";
  }

  if (fieldNames.empty()) {
    return emitOpError("named struct must have at least one field");
  }

  // Verify all field names are strings
  for (auto nameAttr : fieldNames) {
    if (!mlir::isa<StringAttr>(nameAttr)) {
      return emitOpError("all field names must be strings");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EnumTypeOp
//===----------------------------------------------------------------------===//

ParseResult EnumTypeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse: %underlying_type ["name1", "name2", ...] [%val1, %val2, ...]
  //        {explicit_indices = [...]} def_location(loc(...))

  // Parse underlying_type operand
  OpAsmParser::UnresolvedOperand underlyingTypeOperand;
  Type unresolvedType = parsed_csl::UnresolvedType::get(parser.getContext());
  if (parser.parseOperand(underlyingTypeOperand))
    return failure();

  // Parse element names array: ["name1", "name2", ...]
  SmallVector<std::string> elementNames;
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOptionalRSquare().failed()) {
    // Non-empty list
    std::string name;
    if (parser.parseString(&name))
      return failure();
    elementNames.push_back(name);

    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseString(&name))
        return failure();
      elementNames.push_back(name);
    }

    if (parser.parseRSquare())
      return failure();
  }

  // Build element_names array attribute
  SmallVector<Attribute> nameAttrs;
  for (const auto &name : elementNames)
    nameAttrs.push_back(parser.getBuilder().getStringAttr(name));
  result.addAttribute("element_names",
                      parser.getBuilder().getArrayAttr(nameAttrs));

  // Parse explicit values array: [%val1, %val2, ...] (may be empty)
  SmallVector<OpAsmParser::UnresolvedOperand> explicitValueOperands;
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOptionalRSquare().failed()) {
    // Non-empty list
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();
    explicitValueOperands.push_back(operand);

    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(operand))
        return failure();
      explicitValueOperands.push_back(operand);
    }

    if (parser.parseRSquare())
      return failure();
  }

  // Parse optional attributes (including explicit_indices)
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse def_location
  if (parser.parseKeyword("def_location") || parser.parseLParen())
    return failure();

  LocationAttr defLocation;
  if (parser.parseAttribute(defLocation))
    return failure();
  result.addAttribute("def_location", defLocation);

  if (parser.parseRParen())
    return failure();

  // Resolve operands
  if (parser.resolveOperand(underlyingTypeOperand, unresolvedType,
                            result.operands))
    return failure();

  if (parser.resolveOperands(explicitValueOperands, unresolvedType,
                             result.operands))
    return failure();

  // Add result type
  result.addTypes(unresolvedType);

  return success();
}

void EnumTypeOp::print(OpAsmPrinter &p) {
  // Print: %underlying_type ["name1", "name2", ...] [%val1, %val2, ...]
  //        {explicit_indices = [...]} def_location(loc(...))
  p << " " << getUnderlyingType();

  // Print element names
  p << " [";
  auto elementNames = getElementNames();
  for (size_t i = 0; i < elementNames.size(); ++i) {
    if (i > 0)
      p << ", ";
    p << "\"" << mlir::cast<StringAttr>(elementNames[i]).getValue() << "\"";
  }
  p << "]";

  // Print explicit values
  p << " [";
  auto explicitValues = getExplicitValues();
  for (size_t i = 0; i < explicitValues.size(); ++i) {
    if (i > 0)
      p << ", ";
    p << explicitValues[i];
  }
  p << "]";

  // Print attributes except those we handle specially
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"element_names", "def_location", "explicit_indices"});

  // Print explicit_indices if present
  if (auto indicesAttr = getExplicitIndicesAttr()) {
    p << " {explicit_indices = [";
    auto indices = indicesAttr.getValue();
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i > 0)
        p << ", ";
      p << mlir::cast<IntegerAttr>(indices[i]).getInt();
    }
    p << "]}";
  }

  // Print def_location
  p << " def_location(";
  p.printAttribute(getDefLocation());
  p << ")";
}

LogicalResult EnumTypeOp::verify() {
  // Verify element_names is not empty
  auto elementNames = getElementNames();
  if (elementNames.empty()) {
    return emitOpError("enum must have at least one element");
  }

  // Verify all element names are strings
  for (auto nameAttr : elementNames) {
    if (!mlir::isa<StringAttr>(nameAttr)) {
      return emitOpError("all element names must be strings");
    }
  }

  // Verify explicit_indices match explicit_values count
  auto explicitValues = getExplicitValues();
  auto explicitIndices = getExplicitIndicesAttr();
  if (explicitIndices) {
    if (explicitIndices.size() != explicitValues.size()) {
      return emitOpError("explicit_indices size (")
             << explicitIndices.size() << ") must match explicit_values size ("
             << explicitValues.size() << ")";
    }

    // Verify all indices are within bounds
    for (auto indexAttr : explicitIndices) {
      int64_t idx = mlir::cast<IntegerAttr>(indexAttr).getInt();
      if (idx < 0 || static_cast<size_t>(idx) >= elementNames.size()) {
        return emitOpError("explicit index ")
               << idx << " is out of bounds for enum with "
               << elementNames.size() << " elements";
      }
    }
  } else if (!explicitValues.empty()) {
    return emitOpError("explicit_indices attribute required when "
                       "explicit_values are provided");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructStructOp
//===----------------------------------------------------------------------===//

LogicalResult ConstructStructOp::verify() {
  // Verify field_names and field_values have the same size
  auto fieldNames = getFieldNames();
  auto fieldValues = getFieldValues();

  if (fieldNames.size() != fieldValues.size()) {
    return emitOpError(
               "field_names and field_values must have the same size, got ")
           << fieldNames.size() << " names and " << fieldValues.size()
           << " values";
  }

  // Note: Empty structs (.{}) are allowed in CSL - they represent "no options"
  // when passed to builtins like @allocate_fifo

  // Verify all field names are strings
  for (auto nameAttr : fieldNames) {
    if (!mlir::isa<StringAttr>(nameAttr)) {
      return emitOpError("all field names must be strings");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FieldAccessOp
//===----------------------------------------------------------------------===//

OpFoldResult FieldAccessOp::fold(FoldAdaptor adaptor) {
  Attribute baseAttr = adaptor.getBase();
  if (!baseAttr)
    return nullptr;

  // Handle both LValueAttr and direct StructValueAttr
  csl::StructValueAttr structValue;
  bool wasLValue = false;

  if (auto lvalueAttr = mlir::dyn_cast<csl::LValueAttr>(baseAttr)) {
    // Only fold const lvalues
    if (!lvalueAttr.getType().getIsConst())
      return nullptr;
    structValue = mlir::dyn_cast<csl::StructValueAttr>(lvalueAttr.getRvalue());
    wasLValue = true;
  } else {
    structValue = mlir::dyn_cast<csl::StructValueAttr>(baseAttr);
  }

  if (!structValue)
    return nullptr;

  // Get the field value by name
  Attribute fieldValue = structValue.getField(getFieldName());
  if (!fieldValue)
    return nullptr;

  // If base was an lvalue, wrap result in LValueAttr
  if (wasLValue)
    return csl::LValueAttr::get(mlir::cast<TypedAttr>(fieldValue));

  return mlir::cast<TypedAttr>(fieldValue);
}

//===----------------------------------------------------------------------===//
// IndexAccessOp
//===----------------------------------------------------------------------===//

OpFoldResult IndexAccessOp::fold(FoldAdaptor adaptor) {
  Attribute baseAttr = adaptor.getBase();
  if (!baseAttr)
    return nullptr;

  // Handle both LValueAttr and direct value
  TypedAttr baseValue;
  bool wasLValue = false;

  if (auto lvalueAttr = mlir::dyn_cast<csl::LValueAttr>(baseAttr)) {
    // Only fold const lvalues
    if (!lvalueAttr.getType().getIsConst())
      return nullptr;
    baseValue = lvalueAttr.getRvalue();
    wasLValue = true;
  } else {
    baseValue = mlir::dyn_cast<TypedAttr>(baseAttr);
  }

  if (!baseValue)
    return nullptr;

  // Get all index values (must be constants)
  SmallVector<int64_t> indexValues;
  for (Attribute indexAttr : adaptor.getIndices()) {
    auto typedIndexAttr = llvm::dyn_cast_if_present<TypedAttr>(indexAttr);
    if (!typedIndexAttr)
      return nullptr;

    FailureOr<TypedAttr> castedIndexAttr = csl::evaluateCast(
        typedIndexAttr,
        IntegerType::get(getContext(), 32, IntegerType::Signed));

    if (failed(castedIndexAttr))
      return nullptr;

    auto indexInt = llvm::cast<IntegerAttr>(*castedIndexAttr);
    indexValues.push_back(indexInt.getValue().getSExtValue());
  }

  if (indexValues.empty())
    return nullptr;

  // Handle struct/tuple access
  if (auto structValue = dyn_cast<csl::StructValueAttr>(baseValue)) {
    if (indexValues.size() != 1)
      return nullptr;

    int64_t index = indexValues[0];
    if (index < 0 || static_cast<size_t>(index) >= structValue.getNumFields())
      return nullptr;

    Attribute fieldValue = structValue.getField(static_cast<size_t>(index));
    if (wasLValue)
      return csl::LValueAttr::get(mlir::cast<TypedAttr>(fieldValue));
    return mlir::cast<TypedAttr>(fieldValue);
  }

  // Handle array access
  if (auto arrayValue = dyn_cast<csl::ArrayValueAttr>(baseValue)) {
    TypedAttr element = arrayValue.getElement(indexValues);
    if (!element)
      return nullptr;
    if (wasLValue)
      return csl::LValueAttr::get(element);
    return element;
  }

  return nullptr;
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

  // Verify argument types are unresolved
  for (BlockArgument arg : bodyBlock.getArguments()) {
    if (!mlir::isa<UnresolvedType>(arg.getType()))
      return emitOpError("body block arguments must have unresolved type");
  }

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

  // Condition region must be terminated by YieldOp
  if (!mlir::isa<YieldOp>(condBlock.getTerminator()))
    return emitOpError("condition region must be terminated by yield");

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

LogicalResult LabeledBlockOp::verify() {
  if (getLabel().empty())
    return emitOpError("label cannot be empty");
  return success();
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

LogicalResult BreakOp::verify() {
  // Value can only be provided with a target label (labeled block breaks)
  if (getValue() && !getTargetLabel())
    return emitOpError("break with value requires a target label");
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

FailureOr<GlobalOp> parsed_csl::ModuleOp::getMemberVariable(StringRef name) {
  // Search the module body for a global variable with the given name
  for (GlobalOp op : getBody()->getOps<GlobalOp>()) {
    if (op.getSymName() == name) {
      return op;
    }
  }
  return failure();
}

FailureOr<std::variant<FuncOp, GenericFuncOp>>
parsed_csl::ModuleOp::getMemberFunc(StringRef name) {
  // Search for an instantiated function first
  for (FuncOp op : getBody()->getOps<FuncOp>()) {
    if (op.getSymName() == name) {
      return std::variant<FuncOp, GenericFuncOp>(op);
    }
  }
  // Then search for a generic function
  for (GenericFuncOp op : getBody()->getOps<GenericFuncOp>()) {
    if (op.getSymName() == name) {
      return std::variant<FuncOp, GenericFuncOp>(op);
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// TensorAccessOp
//===----------------------------------------------------------------------===//

LogicalResult TensorAccessOp::verify() {
  // Verify we have between 1 and 4 dimensions
  size_t numDims = getInductionVarNames().size();
  if (numDims < 1 || numDims > 4)
    return emitOpError("tensor_access requires 1-4 induction variables, got ")
           << numDims;

  // Verify number of extents matches number of induction variables
  if (getExtents().size() != numDims)
    return emitOpError("number of extents (")
           << getExtents().size()
           << ") must match number of induction variables (" << numDims << ")";

  // Verify the body block has the correct number of arguments
  Block &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != numDims)
    return emitOpError("body block must have ")
           << numDims << " arguments (one per induction variable), got "
           << bodyBlock.getNumArguments();

  // Verify the body is terminated by TensorAccessYieldOp
  if (bodyBlock.empty() ||
      !mlir::isa<TensorAccessYieldOp>(bodyBlock.getTerminator()))
    return emitOpError("body must be terminated by tensor_access_yield");

  return success();
}

//===----------------------------------------------------------------------===//
// TensorAccessYieldOp
//===----------------------------------------------------------------------===//

LogicalResult TensorAccessYieldOp::verify() {
  // Get the parent TensorAccessOp
  auto parentOp = (*this)->getParentOfType<TensorAccessOp>();
  if (!parentOp)
    return emitOpError("must be nested within tensor_access");

  // Note: The number of indices does NOT need to match the number of induction
  // variables. In CSL, a 1D tensor access can address a multi-dimensional array
  // (e.g., |i|{20} -> array[i, i] for diagonal access of a 2D array).
  // The proper validation that indices match array dimensions happens in
  // the comptime evaluator where we have access to the base array type.

  return success();
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
  return success();
}
