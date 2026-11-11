# ParsedCSL Dialect TableGen Definitions

set(LLVM_TARGET_DEFINITIONS ParsedCSLBase.td)
mlir_tablegen(ParsedCSLDialect.h.inc -gen-dialect-decls -dialect=parsed_csl)
mlir_tablegen(ParsedCSLDialect.cpp.inc -gen-dialect-defs -dialect=parsed_csl)
add_public_tablegen_target(MLIRParsedCSLDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ParsedCSLTypes.td)
mlir_tablegen(ParsedCSLTypes.h.inc -gen-typedef-decls -typedefs-dialect=parsed_csl)
mlir_tablegen(ParsedCSLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=parsed_csl)
mlir_tablegen(ParsedCSLTypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(ParsedCSLTypeConstraints.cpp.inc -gen-type-constraint-defs)
add_public_tablegen_target(MLIRParsedCSLTypesIncGen)

set(LLVM_TARGET_DEFINITIONS ParsedCSLAttrs.td)
mlir_tablegen(ParsedCSLAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=parsed_csl)
mlir_tablegen(ParsedCSLAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=parsed_csl)
add_public_tablegen_target(MLIRParsedCSLAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS ParsedCSLInterfaces.td)
mlir_tablegen(ParsedCSLInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ParsedCSLInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRParsedCSLInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ParsedCSLOps.td)
mlir_tablegen(ParsedCSLOps.h.inc -gen-op-decls)
mlir_tablegen(ParsedCSLOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRParsedCSLOpsIncGen)
