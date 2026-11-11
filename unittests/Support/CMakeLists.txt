# SIF Support Unit Tests

add_csl_mlir_unittest(CSLMLIRSifTests
  SifFileTest.cpp
)

target_link_libraries(CSLMLIRSifTests
  PRIVATE
  CSLSifReader
  Squashfs::Squashfs
  LLVMTestingSupport
)

# Make the test data directory available to tests.
target_compile_definitions(CSLMLIRSifTests PRIVATE
  "SIF_TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/testdata\""
)

# Optional SDK SIF integration test — only enabled when both
# CSL_SDK_SIF_PATH and CSL_SDK_SIF_CSL_LIBS_PATH are set.
if(CSL_SDK_SIF_PATH AND CSL_SDK_SIF_CSL_LIBS_PATH)
  target_compile_definitions(CSLMLIRSifTests PRIVATE
    CSL_SDK_SIF_TESTS_ENABLED
    "CSL_SDK_SIF_PATH=\"${CSL_SDK_SIF_PATH}\""
    "CSL_SDK_SIF_CSL_LIBS_PATH=\"${CSL_SDK_SIF_CSL_LIBS_PATH}\""
  )
endif()
