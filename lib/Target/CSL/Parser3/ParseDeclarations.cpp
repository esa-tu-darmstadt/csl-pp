//===- ParseDeclarations.cpp - Declaration Parsing ------------------------===//
//
// Top-level declaration parsing: functions, variables, params, layout,
// comptime blocks.
//
// Reference: Zig's parseFnProto, parseVarDeclProto, parseGlobalVarDecl.
//
//===----------------------------------------------------------------------===//

#include "Grammar.h"

#include "csl-mlir/Dialect/CSL/IR/CSL.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinDescriptor.h"
#include "csl-mlir/Dialect/CSL/IR/CSLBuiltinTypes.h"
#include "csl-mlir/Dialect/ParsedCSL/IR/ParsedCSL.h"
#include "mlir/IR/Attributes.h"

#include "llvm/Support/Debug.h"
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::csl;
using namespace mlir::csl::parser3;

#define DEBUG_TYPE "parser3"

using TK = TokenKind;

#define TRACE() Parser::TraceScope _trace(p, __func__)

//===----------------------------------------------------------------------===//
// extractDocString
//===----------------------------------------------------------------------===//

/// Extract doc comment text from the leading trivia of `tok`.
///
/// Walks backward through the leading trivia to find the last contiguous block
/// of comment items (DocComment `///` or LineComment `//`). "Contiguous" means
/// separated only by single-line whitespace (a Whitespace trivia with fewer
/// than two newlines). A blank line (Whitespace with 2+ newlines) breaks the
/// attachment.
///
/// Comment prefixes (`/// `, `///`, `// `, `//`) are stripped from each line.
/// Lines are joined with '\n'. Returns nullopt when no comments are found
/// directly attached to the token.
static std::optional<std::string> extractDocString(Parser &p,
                                                   const Token &tok) {
  llvm::ArrayRef<Trivia> trivia = p.leadingTrivia(tok);
  if (trivia.empty())
    return std::nullopt;

  auto isComment = [](TriviaKind k) {
    return k == TriviaKind::DocComment || k == TriviaKind::LineComment;
  };

  // Walk backwards to find the last comment in the leading trivia, skipping
  // only whitespace that doesn't contain a blank line.
  int lastDoc = -1;
  for (int i = static_cast<int>(trivia.size()) - 1; i >= 0; --i) {
    if (isComment(trivia[i].kind)) {
      lastDoc = i;
      break;
    }
    if (trivia[i].kind == TriviaKind::Whitespace) {
      // A blank line (two or more newlines) breaks direct attachment.
      if (p.triviaText(trivia[i]).count('\n') >= 2)
        return std::nullopt;
      continue;
    }
    // ContainerDocComment or anything else breaks attachment.
    return std::nullopt;
  }
  if (lastDoc < 0)
    return std::nullopt;

  // Walk further backward to find the start of the contiguous comment block.
  int firstDoc = lastDoc;
  for (int i = lastDoc - 1; i >= 0; --i) {
    if (isComment(trivia[i].kind)) {
      firstDoc = i;
    } else if (trivia[i].kind == TriviaKind::Whitespace) {
      if (p.triviaText(trivia[i]).count('\n') >= 2)
        break;
    } else {
      break;
    }
  }

  // Concatenate comment lines, stripping the leading `/// `, `///`, `// `,
  // or `//` prefix.
  std::string result;
  for (int i = firstDoc; i <= lastDoc; ++i) {
    if (!isComment(trivia[i].kind))
      continue;
    llvm::StringRef text = p.triviaText(trivia[i]);
    if (text.starts_with("/// "))
      text = text.drop_front(4);
    else if (text.starts_with("///"))
      text = text.drop_front(3);
    else if (text.starts_with("// "))
      text = text.drop_front(3);
    else if (text.starts_with("//"))
      text = text.drop_front(2);
    if (!result.empty())
      result += '\n';
    result += text;
  }
  return result.empty() ? std::nullopt : std::optional<std::string>(result);
}

//===----------------------------------------------------------------------===//
// parseFuncDecl
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseFuncDecl(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  // Capture doc comments from leading trivia of the first token before
  // consuming any tokens.
  auto docString = extractDocString(p, p.current());

  // Parse modifiers: export, extern, inline, noinline, async
  bool isExport = false;
  bool isExtern = false;
  bool isInline = false;
  bool isNoinline = false;
  bool isAsync = false;

  while (p.at(TK::KwExport) || p.at(TK::KwExtern) || p.at(TK::KwInline) ||
         p.at(TK::KwNoinline) || p.atIdent("async")) {
    if (p.eat(TK::KwExport))
      isExport = true;
    else if (p.eat(TK::KwExtern))
      isExtern = true;
    else if (p.eat(TK::KwInline))
      isInline = true;
    else if (p.eat(TK::KwNoinline))
      isNoinline = true;
    else if (p.eatIdent("async"))
      isAsync = true;
  }

  // 'fn' or 'task'
  bool isTask = p.at(TK::KwTask);
  p.advance(); // consume 'fn' or 'task'

  // Function name
  auto nameResult = p.nameRecover(ITEM_RECOVERY);
  if (failed(nameResult))
    return failure();
  llvm::StringRef funcName = p.getText(*nameResult);

  // Create GenericFuncOp with placeholder attributes (updated after parsing
  // params). We need the op to exist so we can build into its regions.
  auto genericFuncOp = parsed_csl::GenericFuncOp::create(
      builder, loc.start(), funcName,
      /*comptime_params=*/{}, /*anytype_params=*/{},
      /*num_params=*/0, isTask, isInline, isNoinline, isExport, isExtern,
      isAsync, /*stack_size=*/IntegerAttr{}, builder.getArrayAttr({}),
      docString ? builder.getStringAttr(*docString) : StringAttr{},
      /*max_recursion_depth=*/IntegerAttr{});

  // Register in global symbols early so recursive references work.
  // On conflict, declareGlobalSymbol emits the error. Continue parsing for
  // resilience.
  p.declareGlobalSymbol(genericFuncOp.getSymName(), genericFuncOp);

  auto unresolvedTy = p.getUnresolvedTy();

  // Parameter info collected while parsing.
  // name is a StringRef into the source buffer (stable, not heap-allocated),
  // so it's safe to use as a ScopedHashTable key even as the vector grows.
  struct ParamInfo {
    llvm::StringRef name;
    bool isComptime = false;
    bool isAnytype = false;
    Location loc;
    ParamInfo(Location loc) : loc(loc) {}
  };
  SmallVector<ParamInfo> params;

  // Build type region — parameters and return type are parsed here,
  // so that type IR is emitted directly into the type region.
  {
    OpBuilder::InsertionGuard guard(builder);
    Region &typeRegion = genericFuncOp.getTypeRegion();
    Block *typeBlock = builder.createBlock(&typeRegion);
    builder.setInsertionPointToStart(typeBlock);

    Parser::LocalScopeGuard typeScope(p.getLocalScope());
    SmallVector<Value> argTypeValues;

    // Parse parameter list: fn name(params...) retType
    if (p.at(TK::LParen)) {
      p.advance(); // consume '('
      while (!p.at(TK::RParen) && !p.eof()) {
        if (!params.empty()) {
          if (!p.eat(TK::Comma))
            break;
        }

        ParamInfo param(p.getLoc());
        param.isComptime = p.eat(TK::KwComptime);

        // Parameter name (if followed by ':')
        if (p.at(TK::Identifier) && p.nth(1) == TK::Colon) {
          auto nameTok = p.advance();
          param.name = p.getText(nameTok);
          p.advance(); // consume ':'
        }

        Value paramType;
        Value paramValue;

        // Check for anytype
        if (p.at(TK::KwAnytype)) {
          param.isAnytype = true;
          p.advance(); // consume 'anytype'
        } else {
          // Parse type expression
          auto maybeType = parseValueTypeExpr(p);
          if (failed(maybeType)) {
            p.emitError("expected parameter type");
            break;
          }
          paramType = *maybeType;
        }

        if (param.isComptime || param.isAnytype) {
          paramValue = typeBlock->addArgument(unresolvedTy, param.loc);
        } else {
          paramValue =
              parsed_csl::NoValueOp::create(builder, param.loc, paramType);
        }

        if (param.isComptime && !param.isAnytype) {
          paramValue = parsed_csl::CastOp::create(
              builder, param.loc, p.read(paramValue, param.loc), paramType,
              /*inferred_type=*/{},
              /*implicit=*/true);
        }

        if (param.isAnytype) {
          paramType = parsed_csl::BuiltinCallOp::create(builder, param.loc,
                                                        "type_of", paramValue,
                                                        /*inferred_type=*/{});
        }

        argTypeValues.push_back(paramType);

        params.push_back(param);

        // Insert into scope AFTER push_back so the StringRef points to the
        // stable std::string inside the vector element, not the local `param`.
        if (!params.back().name.empty())
          p.getLocalScope().insert(params.back().name, paramValue);
      }
      (void)p.expect(TK::RParen);
    } else {
      p.emitError("expected parameter list");
    }

    // Optional `linkname(expr)`, `stack_size(N)` and `max_recursion_depth(N)`
    // modifiers, in any order, between the `)` of the parameter list and the
    // return type. Each may appear at most once.
    bool sawLinkname = false;
    std::optional<Location> stackSizeLoc;
    std::optional<Location> maxRecursionDepthLoc;
    // Parse the `(N)` of a modifier, where N is an integer literal that fits
    // in u32.
    auto parseU32Argument =
        [&](llvm::StringRef modifier,
            Location modifierLoc) -> std::optional<uint32_t> {
      std::optional<uint32_t> result;
      (void)p.expect(TK::LParen);
      if (!p.at(TK::IntegerLiteral)) {
        p.emitError(llvm::Twine(modifier) + " expects an integer literal");
      } else {
        auto litTok = p.advance();
        llvm::StringRef litText = p.getText(litTok);
        std::string cleaned;
        cleaned.reserve(litText.size());
        for (char c : litText)
          if (c != '_')
            cleaned += c;
        llvm::StringRef cleanedRef(cleaned);
        uint64_t value = 0;
        if (cleanedRef.starts_with("0x") || cleanedRef.starts_with("0X"))
          value = std::stoull(cleaned.substr(2), nullptr, 16);
        else if (cleanedRef.starts_with("0b") || cleanedRef.starts_with("0B"))
          value = std::stoull(cleaned.substr(2), nullptr, 2);
        else if (cleanedRef.starts_with("0o") || cleanedRef.starts_with("0O"))
          value = std::stoull(cleaned.substr(2), nullptr, 8);
        else
          value = std::stoull(cleaned);
        if (value > std::numeric_limits<uint32_t>::max())
          p.emitError(modifierLoc,
                      llvm::Twine(modifier) + " value does not fit in u32");
        else
          result = static_cast<uint32_t>(value);
      }
      (void)p.expect(TK::RParen);
      return result;
    };
    while (p.at(TK::KwLinkname) || p.atIdent("stack_size") ||
           p.atIdent("max_recursion_depth")) {
      if (p.at(TK::KwLinkname)) {
        if (sawLinkname)
          p.emitError("duplicate 'linkname' modifier");
        sawLinkname = true;
        p.advance(); // consume 'linkname'
        (void)p.expect(TK::LParen);
        OpBuilder::InsertionGuard lnGuard(builder);
        Region &linknameRegion = genericFuncOp.getLinknameRegion();
        builder.createBlock(&linknameRegion);
        auto lnResult = parseValueExpr(p);
        if (failed(lnResult))
          return p.emitError("failed to parse linkname expression");
        parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                    *lnResult);
        (void)p.expect(TK::RParen);
      } else if (p.atIdent("stack_size")) {
        if (stackSizeLoc)
          p.emitError("duplicate 'stack_size' modifier");
        Location modifierLoc = p.getLoc(p.advance()); // consume 'stack_size'
        stackSizeLoc = modifierLoc;
        if (std::optional<uint32_t> value =
                parseU32Argument("stack_size", modifierLoc))
          genericFuncOp.setStackSizeAttr(builder.getUI32IntegerAttr(*value));
      } else {
        if (maxRecursionDepthLoc)
          p.emitError("duplicate 'max_recursion_depth' modifier");
        // consume 'max_recursion_depth'
        Location modifierLoc = p.getLoc(p.advance());
        maxRecursionDepthLoc = modifierLoc;
        std::optional<uint32_t> value =
            parseU32Argument("max_recursion_depth", modifierLoc);
        if (value && *value == 0)
          p.emitError(modifierLoc, "max_recursion_depth must be at least 1: it "
                                   "counts the calls of '" +
                                       funcName +
                                       "' that may be active at the same "
                                       "time, including the outermost one");
        else if (value)
          genericFuncOp.setMaxRecursionDepthAttr(
              builder.getUI32IntegerAttr(*value));
      }
    }
    if (stackSizeLoc && maxRecursionDepthLoc)
      p.emitWarning(*maxRecursionDepthLoc,
                    "'max_recursion_depth' is ignored because 'stack_size' "
                    "already declares the stack reach of '" +
                        funcName + "'; remove one of the two modifiers");

    // Parse return type (type expression — no binary ops, no array literals,
    // because { after the return type starts the function body)
    Value returnTypeValue;
    if (p.atTs(EXPR_FIRST) && !p.at(TK::LBrace)) {
      auto retType = parseValueTypeExpr(p);
      if (failed(retType))
        return p.emitError("failed to parse return type");
      returnTypeValue = *retType;
    } else {
      // No return type specified - use void
      auto voidType = csl::VoidType::get(&p.getContext());
      auto typeAttr = parsed_csl::TypeAttr::get(voidType);
      returnTypeValue =
          parsed_csl::BuiltinTypeOp::create(builder, loc.start(), typeAttr)
              .getResult();
    }

    auto funcTypeOp = parsed_csl::FuncTypeOp::create(
        builder, loc.start(), argTypeValues, returnTypeValue, isTask, isAsync);
    parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                funcTypeOp.getResult());
  }

  // Now update GenericFuncOp attributes with actual param info
  size_t numParams = params.size();
  SmallVector<int32_t> comptimeParamIndices;
  SmallVector<int32_t> anytypeParamIndices;
  SmallVector<Attribute> paramNameAttrs;
  for (size_t i = 0; i < numParams; ++i) {
    if (params[i].isComptime)
      comptimeParamIndices.push_back(static_cast<int32_t>(i));
    if (params[i].isAnytype)
      anytypeParamIndices.push_back(static_cast<int32_t>(i));
    paramNameAttrs.push_back(builder.getStringAttr(params[i].name));
  }
  genericFuncOp.setNumParamsAttr(
      builder.getI32IntegerAttr(static_cast<int32_t>(numParams)));
  genericFuncOp.setComptimeParamsAttr(
      builder.getDenseI32ArrayAttr(comptimeParamIndices));
  genericFuncOp.setAnytypeParamsAttr(
      builder.getDenseI32ArrayAttr(anytypeParamIndices));
  genericFuncOp.setParamNamesAttr(builder.getArrayAttr(paramNameAttrs));

  // Validate restrictions (emit errors but continue parsing for resilience)
  bool isGeneric =
      !comptimeParamIndices.empty() || !anytypeParamIndices.empty();
  if (isTask && isGeneric)
    p.emitError("tasks cannot have comptime or anytype parameters");
  if (isTask && isInline)
    p.emitError("tasks cannot have 'inline' modifier");
  if (isTask && isNoinline)
    p.emitError("tasks cannot have 'noinline' modifier");
  if (isInline && isNoinline)
    p.emitError("function cannot be both 'inline' and 'noinline'");
  if (isNoinline && isExport)
    p.emitError("exported functions cannot be 'noinline'");
  if (isNoinline && isExtern)
    p.emitError("extern functions cannot be 'noinline'");
  if (isTask && isExport)
    p.emitError("tasks cannot have 'export' modifier");
  if (isTask && isExtern)
    p.emitError("tasks cannot have 'extern' modifier");
  if (isAsync && isExtern)
    p.emitError("async functions cannot be extern");
  if (isExport && isGeneric)
    p.emitError("exported functions cannot be generic");

  // Build body region (only if function has a body)
  if (p.at(TK::LBrace)) {
    OpBuilder::InsertionGuard guard(builder);
    Region &bodyRegion = genericFuncOp.getBody();
    Block *bodyBlock = builder.createBlock(&bodyRegion);

    for (size_t i = 0; i < numParams; ++i)
      bodyBlock->addArgument(unresolvedTy, params[i].loc);

    auto savedFunc = p.getCurrentFunction();
    p.setCurrentFunction(genericFuncOp);

    builder.setInsertionPointToStart(bodyBlock);
    Parser::LocalScopeGuard bodyScope(p.getLocalScope());

    // Bind parameter names to block arguments
    for (auto [param, arg] : llvm::zip(params, bodyBlock->getArguments())) {
      if (!param.name.empty()) {
        if (failed(p.declareLocalSymbol(param.name, arg, arg.getLoc())))
          return failure();
      }
    }

    p.advance(); // consume '{'
    parseStatementList(p);
    (void)p.expect(TK::RBrace);

    p.setCurrentFunction(savedFunc);
  } else if (p.eat(TK::Semi)) {
    // Function declaration without body
    if (isExport)
      p.emitError(loc.start(), "exported functions must have a body");
  } else if (!isExtern) {
    p.emitError("expected function body or ';'");
  }

  // Update location to span the full declaration.
  genericFuncOp->setLoc(loc.toHere());

  return success();
}

//===----------------------------------------------------------------------===//
// parseVarDecl
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseVarDecl(Parser &p, bool isGlobal) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  // Parse modifiers
  bool isExport = false;
  bool isExtern = false;
  bool isComptime = false;
  bool isConst = false;

  while (true) {
    if (p.eat(TK::KwExport))
      isExport = true;
    else if (p.eat(TK::KwExtern))
      isExtern = true;
    else if (p.eat(TK::KwComptime))
      isComptime = true;
    else if (p.at(TK::KwConst)) {
      isConst = true;
      p.advance();
      break;
    } else if (p.at(TK::KwVar)) {
      p.advance();
      break;
    } else {
      break;
    }
  }

  // Variable name
  auto nameResult = p.nameRecover(ITEM_RECOVERY);
  if (failed(nameResult))
    return failure();
  llvm::StringRef varName = p.getText(*nameResult);

  // Optional type annotation: : type
  Value dynVarType;
  if (p.eat(TK::Colon)) {
    auto typeResult = parseValueTypeExpr(p);
    if (failed(typeResult))
      return failure();
    dynVarType = *typeResult;
  }

  // Optional variable attributes: align(expr), linkname(expr)
  // Parse into standalone blocks; they'll be inserted into GlobalOp regions.
  auto parseAttrBlock = [&](const char *name) -> Block * {
    auto *block = new Block();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    (void)p.expect(TK::LParen);
    auto result = parseValueExpr(p);
    if (failed(result)) {
      p.emitError(llvm::Twine("failed to parse ") + name + " expression");
      delete block;
      return nullptr;
    }
    parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(), *result);
    (void)p.expect(TK::RParen);
    return block;
  };

  Block *alignBlock = nullptr;
  Block *linknameBlock = nullptr;
  while (p.at(TK::KwAlign) || p.at(TK::KwLinkname)) {
    if (p.eat(TK::KwAlign)) {
      if (alignBlock)
        p.emitError("multiple align attributes not allowed");
      alignBlock = parseAttrBlock("align");
    } else {
      p.advance(); // consume 'linkname'
      if (linknameBlock)
        p.emitError("multiple linkname attributes not allowed");
      linknameBlock = parseAttrBlock("linkname");
    }
  }

  // Optional initializer: = expr
  bool hasInit = p.eat(TK::Eq);

  // Validate
  if (!isGlobal && (isExport || isExtern))
    return p.emitError("local variables cannot be export or extern");
  if (isExtern && hasInit)
    return p.emitError("extern variables cannot have an initializer");
  if (isExtern && !dynVarType)
    return p.emitError("extern variables must have a type");
  if (isComptime && !hasInit)
    return p.emitError(
        "comptime variable must be initialized with a comptime expression");
  if (isGlobal && isConst && !hasInit && !isExtern)
    return p.emitError("const global variable must have initializer");
  if (!dynVarType && !hasInit)
    return p.emitError("either type or initializer required for variable");

  if (isGlobal) {
    auto globalOp = parsed_csl::GlobalOp::create(
        builder, loc.start(), varName, dynVarType, isConst, isExport, isExtern,
        /*is_unbound_param=*/false,
        /*linkname=*/nullptr, /*align=*/nullptr, /*inferred_type=*/nullptr,
        /*inferred_initializer=*/nullptr);

    // Move attribute blocks into GlobalOp regions.
    if (alignBlock)
      globalOp.getAlignRegion().push_back(alignBlock);
    if (linknameBlock)
      globalOp.getLinknameRegion().push_back(linknameBlock);

    // Init region
    if (hasInit) {
      Block &initBlock = globalOp.getInitRegion().emplaceBlock();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&initBlock);
      auto initResult = parseValueExpr(p);
      if (failed(initResult))
        return failure();
      parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                  *initResult);
    }

    // Register in global symbols (emits error on conflict)
    p.declareGlobalSymbol(globalOp.getSymName(), globalOp);

    // Update location to span the full declaration.
    globalOp->setLoc(loc.toHere());
  } else {
    // Local variable
    bool requiresComptimeInit = isComptime;
    Value initializer;

    if (hasInit) {
      if (requiresComptimeInit) {
        auto comptimeLoc = p.startLoc();
        auto comptimeExprOp = parsed_csl::ComptimeExprOp::create(
            builder, comptimeLoc.start(), /*inferred_type=*/nullptr);
        Block &block = comptimeExprOp.getExprRegion().emplaceBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&block);
        auto initResult = parseValueExpr(p);
        if (failed(initResult))
          return failure();
        parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                    *initResult);
        comptimeExprOp->setLoc(comptimeLoc.toHere());
        initializer = comptimeExprOp.getResult();
      } else {
        auto initResult = parseValueExpr(p);
        if (failed(initResult))
          return failure();
        initializer = *initResult;
      }
    }

    auto varOp = parsed_csl::VariableOp::create(
        builder, loc.toHere(), p.getUnresolvedTy(),
        builder.getStringAttr(varName), dynVarType, initializer, isConst,
        isComptime, /*inferred_type=*/{});

    if (failed(p.declareLocalSymbol(varOp.getName(), varOp.getResult(),
                                    loc.start())))
      return failure();
  }

  (void)p.expect(TK::Semi);
  return success();
}

//===----------------------------------------------------------------------===//
// parseParamDecl
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseParamDecl(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  p.advance(); // consume 'param'

  auto nameResult = p.nameRecover(ITEM_RECOVERY);
  if (failed(nameResult))
    return failure();
  llvm::StringRef paramName = p.getText(*nameResult);

  // Optional type: : type
  Value dynParamType;
  if (p.eat(TK::Colon)) {
    auto typeResult = parseValueTypeExpr(p);
    if (failed(typeResult))
      return failure();
    dynParamType = *typeResult;
  }

  // Optional default value: = expr
  bool hasDefault = p.at(TK::Eq);

  // Check param bindings
  auto &paramBindings = p.getParamBindings();
  auto it = paramBindings.find(paramName);
  bool hasBinding = it != paramBindings.end();

  if (!dynParamType && !hasDefault && !hasBinding)
    return p.emitError("param '" + paramName +
                       "' has no type, default value, or binding provided");

  if (hasBinding)
    p.markParamUsed(it->first);

  auto globalOp = parsed_csl::GlobalOp::create(
      builder, loc.start(), paramName, dynParamType,
      /*is_const=*/true, /*is_export=*/false, /*is_extern=*/false,
      /*is_unbound_param=*/(!hasBinding && !hasDefault),
      /*linkname=*/nullptr, /*align=*/nullptr, /*inferred_type=*/{},
      /*inferred_initializer=*/{});

  // Populate init region. Parse the default into the init block first, then
  // replace with the binding if one was provided.
  if (hasDefault || hasBinding) {
    Block &initBlock = globalOp.getInitRegion().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&initBlock);

    if (hasDefault) {
      p.advance(); // consume '='
      auto defaultResult = parseValueExpr(p);
      if (failed(defaultResult))
        return failure();

      if (hasBinding) {
        // Binding overrides default. Erase the parsed default ops (in
        // reverse so users are erased before their operands) and emit the
        // binding constant instead.
        for (auto &op : llvm::make_early_inc_range(llvm::reverse(initBlock)))
          op.erase();
        TypedAttr boundValue = it->second.value;
        Value constValue =
            parsed_csl::ConstantOp::create(builder, loc.start(), boundValue);
        parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                    constValue);
      } else {
        parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(),
                                    *defaultResult);
      }
    } else {
      // Binding only, no default expression to consume.
      TypedAttr boundValue = it->second.value;
      Value constValue =
          parsed_csl::ConstantOp::create(builder, loc.start(), boundValue);
      parsed_csl::YieldOp::create(builder, builder.getUnknownLoc(), constValue);
    }
  }

  // Register in global symbols (emits error on conflict)
  p.declareGlobalSymbol(globalOp.getSymName(), globalOp);
  globalOp->setLoc(loc.toHere());

  (void)p.expect(TK::Semi);
  return success();
}

//===----------------------------------------------------------------------===//
// parseLayoutDecl
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseLayoutDecl(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  p.advance(); // consume 'layout'

  auto layoutBlockOp = parsed_csl::LayoutBlockOp::create(builder, loc.start());

  {
    Block &block = layoutBlockOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);
    Parser::LocalScopeGuard scope(p.getLocalScope());

    (void)p.expect(TK::LBrace);
    parseStatementList(p);
    (void)p.expect(TK::RBrace);
  }

  layoutBlockOp->setLoc(loc.toHere());
  return success();
}

//===----------------------------------------------------------------------===//
// parseComptimeBlockDecl
//===----------------------------------------------------------------------===//

LogicalResult mlir::csl::parser3::parseComptimeBlockDecl(Parser &p) {
  TRACE();
  auto &builder = p.getBuilder();
  auto loc = p.startLoc();

  p.advance(); // consume 'comptime'

  auto comptimeBlockOp =
      parsed_csl::ComptimeBlockOp::create(builder, loc.start());

  {
    Block &block = comptimeBlockOp.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);
    Parser::LocalScopeGuard scope(p.getLocalScope());

    (void)p.expect(TK::LBrace);
    parseStatementList(p);
    (void)p.expect(TK::RBrace);
  }

  comptimeBlockOp->setLoc(loc.toHere());

  // Optional trailing semicolon: `comptime { ... };`
  p.eat(TK::Semi);

  return success();
}
