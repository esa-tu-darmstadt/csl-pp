# TypedCSL Dialect

add_mlir_dialect_library(MLIRTypedCSLDialect
  TypedCSLDialect.cpp
  TypedCSLOps.cpp
  TypedCSLCanonicalize.cpp
  SharedLiteralParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/TypedCSL/IR

  DEPENDS
  MLIRTypedCSLDialectIncGen
  MLIRTypedCSLTypesIncGen
  MLIRTypedCSLAttrsIncGen
  MLIRTypedCSLInterfacesIncGen
  MLIRTypedCSLOpsIncGen
  MLIRCSLDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRCastInterfaces
  MLIRCallInterfaces
  MLIRCSLDialect
  )
