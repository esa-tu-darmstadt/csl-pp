add_custom_target(check-csl)

# The SDK provides lit itself. This build owns the source-to-site configuration
# map that LLVM's in-tree llvm-lit launcher previously supplied.
get_filename_component(CSL_LLVM_LIT_DIR "${LLVM_EXTERNAL_LIT}" DIRECTORY)
configure_file(${CSL_MLIR_SOURCE_DIR}/cmake/csl-lit.py.in
  ${CSL_MLIR_BINARY_DIR}/bin/csl-lit.py @ONLY)
set(LLVM_EXTERNAL_LIT "${CSL_MLIR_BINARY_DIR}/bin/csl-lit.py")

# Note: Here [CACHE INTERNAL ""] is used to make the set variable globally available.
set(CSL_MLIR_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
set(CSL_MLIR_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
set(CSL_MLIR_OPT_TOOL_BINARY_PATH "${CMAKE_BINARY_DIR}/bin/csl-opt" CACHE INTERNAL "")
# CSL_MLIR_SOURCE_DIR should point to the root of csl-mlir, which is the project root
set(CSL_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE INTERNAL "")
set(CSL_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/.." CACHE INTERNAL "")

# Configure main regression test suite
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py

        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Configure unit test suite
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py

        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
)

# (Re-)Set llvm-lit arguments
set(CHECK_CSL_LIT_ARGS "")

# Add verbose flag if requested
if (${CHECK_CSL_VERBOSE})
    message(STATUS "Running llvm-lit in verbose mode")
    set(CHECK_CSL_LIT_ARGS "${CHECK_CSL_LIT_ARGS};-v")
endif ()


# Test dependencies
set(CSL_MLIR_TEST_DEPENDS
  csl-opt
  csl-s2s
  csl-lsp-server
  csl-pp
  CSLMLIRUnitTests
)

# Run regression tests
add_lit_testsuite(check-csl-tests "Running check-csl regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CSL_MLIR_TEST_DEPENDS}
        ARGS ${CHECK_CSL_LIT_ARGS})
set_target_properties(check-csl-tests PROPERTIES FOLDER "Tests")

# Auto-discover test suites (creates check-csl-unit, etc.)
add_lit_testsuites(CSL_MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CSL_MLIR_TEST_DEPENDS}
)

add_dependencies(check-csl check-csl-tests)

# ---------------------------------------------------------------------------
# Integration tests (CSL SDK benchmarks/tutorials via csl-pp)
# ---------------------------------------------------------------------------
if (CSL_SDK_DIR AND CSL_EXTRAS_DIR)
    message(STATUS "CSL integration tests enabled")
    set(CSL_PP_BINARY_PATH "${CMAKE_BINARY_DIR}/bin/csl-pp")
    set(CSL_MLIR_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
    if (CSL_SDK_HOST_TOOLS_DIR)
        set(CSL_INTEGRATION_HOST_TOOLS_DIR "${CSL_SDK_HOST_TOOLS_DIR}")
    else()
        set(CSL_INTEGRATION_HOST_TOOLS_DIR "${CSL_SDK_DIR}")
    endif()
    configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/Integration/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/Integration/lit.site.cfg.py
        MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Integration/lit.cfg.py)
    add_lit_testsuite(check-csl-integration
        "Running CSL SDK integration tests"
        ${CMAKE_CURRENT_BINARY_DIR}/Integration
        DEPENDS csl-pp
        ARGS ${CHECK_CSL_LIT_ARGS}
        EXCLUDE_FROM_CHECK_ALL)
    set_target_properties(check-csl-integration PROPERTIES FOLDER "Tests")

    add_lit_testsuite(check-csl-integration-sdk
        "Running CSL SDK example integration tests"
        ${CMAKE_CURRENT_BINARY_DIR}/Integration
        DEPENDS csl-pp
        ARGS ${CHECK_CSL_LIT_ARGS} --filter-out=coro-benchmarks/
        EXCLUDE_FROM_CHECK_ALL)
    set_target_properties(check-csl-integration-sdk PROPERTIES FOLDER "Tests")

    add_lit_testsuite(check-csl-integration-coro
        "Running CSL coroutine integration tests"
        ${CMAKE_CURRENT_BINARY_DIR}/Integration
        DEPENDS csl-pp
        ARGS ${CHECK_CSL_LIT_ARGS} --filter=coro-benchmarks/
        EXCLUDE_FROM_CHECK_ALL)
    set_target_properties(check-csl-integration-coro PROPERTIES FOLDER "Tests")
else()
    message(STATUS "CSL integration tests disabled (set CSL_SDK_DIR and CSL_EXTRAS_DIR)")
endif()
