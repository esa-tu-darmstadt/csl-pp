//===- SyntaxKind.td - Parser3 Syntax Kind Enum -----------*- tablegen -*-===//
//
// Enum definition for syntax range kinds used by the Parser3 parser.
// These mark syntactic constructs in the source for formatting/tooling.
//
//===----------------------------------------------------------------------===//

#ifndef PARSER3_SYNTAX_KIND
#define PARSER3_SYNTAX_KIND

include "mlir/IR/EnumAttr.td"

def Parser3_SyntaxKind : I32EnumAttr<"SyntaxKind",
    "Syntax range kind for parser", [
  // Top-level
  I32EnumAttrCase<"File",              0, "file">,

  // Declarations
  I32EnumAttrCase<"FuncDecl",          1, "func_decl">,
  I32EnumAttrCase<"ParamDecl",         2, "param_decl">,
  I32EnumAttrCase<"VarDecl",           3, "var_decl">,
  I32EnumAttrCase<"LayoutDecl",        4, "layout_decl">,
  I32EnumAttrCase<"ComptimeBlock",     5, "comptime_block">,

  // Declaration parts
  I32EnumAttrCase<"ModifierList",      6, "modifier_list">,
  I32EnumAttrCase<"ParameterList",     7, "parameter_list">,
  I32EnumAttrCase<"Parameter",         8, "parameter">,
  I32EnumAttrCase<"ReturnType",        9, "return_type">,
  I32EnumAttrCase<"FunctionBody",     10, "function_body">,

  // Statements
  I32EnumAttrCase<"StmtExpr",        11, "stmt_expr">,
  I32EnumAttrCase<"StmtAssign",      12, "stmt_assign">,
  I32EnumAttrCase<"StmtReturn",      13, "stmt_return">,
  I32EnumAttrCase<"StmtBreak",       14, "stmt_break">,
  I32EnumAttrCase<"StmtContinue",    15, "stmt_continue">,
  I32EnumAttrCase<"StmtIf",          16, "stmt_if">,
  I32EnumAttrCase<"StmtFor",         17, "stmt_for">,
  I32EnumAttrCase<"StmtWhile",       18, "stmt_while">,
  I32EnumAttrCase<"StmtAsm",         19, "stmt_asm">,
  I32EnumAttrCase<"StmtSwitch",      20, "stmt_switch">,

  // Expressions
  I32EnumAttrCase<"Expr",            21, "expr">,
  I32EnumAttrCase<"ExprCall",        22, "expr_call">,
  I32EnumAttrCase<"ExprFieldAccess", 23, "expr_field_access">,
  I32EnumAttrCase<"ExprIndex",       24, "expr_index">,
  I32EnumAttrCase<"ExprBinary",      25, "expr_binary">,
  I32EnumAttrCase<"ExprUnary",       26, "expr_unary">,
  I32EnumAttrCase<"ExprIf",          27, "expr_if">,
  I32EnumAttrCase<"ExprBlock",       28, "expr_block">,
  I32EnumAttrCase<"ExprSwitch",      29, "expr_switch">,

  // Type expressions
  I32EnumAttrCase<"ExprPtrType",     30, "expr_ptr_type">,
  I32EnumAttrCase<"ExprArrayType",   31, "expr_array_type">,
  I32EnumAttrCase<"ExprFnType",      32, "expr_fn_type">,
  I32EnumAttrCase<"ExprStructType",  33, "expr_struct_type">,
  I32EnumAttrCase<"ExprEnumType",    34, "expr_enum_type">,

  // Compound literals
  I32EnumAttrCase<"ExprStructLiteral", 35, "expr_struct_literal">,
  I32EnumAttrCase<"ExprTupleLiteral",  36, "expr_tuple_literal">,
  I32EnumAttrCase<"ExprTensorAccess",  37, "expr_tensor_access">,
  I32EnumAttrCase<"ExprBuiltinCall",   38, "expr_builtin_call">,

  // Sub-parts
  I32EnumAttrCase<"ArgList",         39, "arg_list">,
  I32EnumAttrCase<"StructField",     40, "struct_field">,
  I32EnumAttrCase<"StructFieldType", 41, "struct_field_type">,
  I32EnumAttrCase<"EnumElement",     42, "enum_element">,
  I32EnumAttrCase<"SwitchCase",      43, "switch_case">,
]> {
  let cppNamespace = "::mlir::csl::parser3";
  let genSpecializedAttr = 0;
}

#endif // PARSER3_SYNTAX_KIND
