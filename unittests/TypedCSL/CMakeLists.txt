# TypedCSL Unit Tests

add_csl_mlir_unittest(CSLMLIRTypedCSLTests
  FiberStackResolverTest.cpp
)

target_link_libraries(CSLMLIRTypedCSLTests
  PRIVATE
  MLIRTypedCSLTransforms
)
