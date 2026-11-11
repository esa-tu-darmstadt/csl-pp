# TypedCSL Dialect TableGen Definitions

set(LLVM_TARGET_DEFINITIONS TypedCSLBase.td)
mlir_tablegen(TypedCSLDialect.h.inc -gen-dialect-decls -dialect=typed_csl)
mlir_tablegen(TypedCSLDialect.cpp.inc -gen-dialect-defs -dialect=typed_csl)
add_public_tablegen_target(MLIRTypedCSLDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TypedCSLTypes.td)
mlir_tablegen(TypedCSLTypes.h.inc -gen-typedef-decls -typedefs-dialect=typed_csl)
mlir_tablegen(TypedCSLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=typed_csl)
mlir_tablegen(TypedCSLTypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(TypedCSLTypeConstraints.cpp.inc -gen-type-constraint-defs)
add_public_tablegen_target(MLIRTypedCSLTypesIncGen)

set(LLVM_TARGET_DEFINITIONS TypedCSLAttrs.td)
mlir_tablegen(TypedCSLAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=typed_csl)
mlir_tablegen(TypedCSLAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=typed_csl)
mlir_tablegen(TypedCSLEnums.h.inc -gen-enum-decls)
mlir_tablegen(TypedCSLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTypedCSLAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS TypedCSLInterfaces.td)
mlir_tablegen(TypedCSLInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TypedCSLInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTypedCSLInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS TypedCSLOps.td)
mlir_tablegen(TypedCSLOps.h.inc -gen-op-decls)
mlir_tablegen(TypedCSLOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTypedCSLOpsIncGen)
