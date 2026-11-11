# CSL Dialect TableGen Definitions

set(LLVM_TARGET_DEFINITIONS CSLBase.td)
mlir_tablegen(CSLDialect.h.inc -gen-dialect-decls -dialect=csl)
mlir_tablegen(CSLDialect.cpp.inc -gen-dialect-defs -dialect=csl)
add_public_tablegen_target(MLIRCSLDialectIncGen)

set(LLVM_TARGET_DEFINITIONS CSLTypes.td)
mlir_tablegen(CSLTypes.h.inc -gen-typedef-decls -typedefs-dialect=csl)
mlir_tablegen(CSLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=csl)
mlir_tablegen(CSLTypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(CSLTypeConstraints.cpp.inc -gen-type-constraint-defs)
add_public_tablegen_target(MLIRCSLTypesIncGen)

set(LLVM_TARGET_DEFINITIONS CSLAttrs.td)
mlir_tablegen(CSLAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=csl)
mlir_tablegen(CSLAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=csl)
mlir_tablegen(CSLEnums.h.inc -gen-enum-decls)
mlir_tablegen(CSLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCSLAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS CSLTypeInterfaces.td)
mlir_tablegen(CSLTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(CSLTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRCSLTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CSLOpInterfaces.td)
mlir_tablegen(CSLOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CSLOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCSLOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CSLOps.td)
mlir_tablegen(CSLOps.h.inc -gen-op-decls)
mlir_tablegen(CSLOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCSLOpsIncGen)
