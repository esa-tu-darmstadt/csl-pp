cmake_minimum_required(VERSION 3.25)
project(csl-mlir VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard selected by the Conan profile")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(CSL_BUILD_TESTS "Build and run compiler regression and unit tests" ON)

# In debug builds, enable -fno-omit-frame-pointer and -mno-omit-leaf-frame-pointer
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer)
endif()

set(CSL_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CSL_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(CSL_RELEASE_VERSION "" CACHE STRING
    "csl++ release printed by --version, e.g. 1.0.0; empty for a development build")
set(CSL_GIT_COMMIT "" CACHE STRING
    "Commit printed by --version, with -dirty for uncommitted changes")
set(CSL_THIRD_PARTY_LICENSES_DIR "" CACHE PATH
    "Directory with the license notices of the linked dependencies, one subdirectory each")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # The build ID matches a binary to its separately published debug
  # information, for example to symbolize a backtrace of a released csl-pp.
  add_link_options(LINKER:--build-id)
  # Static builds link libstdc++ and libgcc into the executables, so that these
  # require only glibc on the machine that runs them. Shared builds keep the
  # shared runtime, which their shared libraries use as well.
  if(NOT BUILD_SHARED_LIBS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_csl_is_executable "$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>")
    add_link_options("$<${_csl_is_executable}:-static-libstdc++>"
                     "$<${_csl_is_executable}:-static-libgcc>")
  endif()
endif()

set(CSL_SDK_DIR "" CACHE PATH
    "Unpacked Cerebras SDK directory used by integration tests")
set(CSL_EXTRAS_DIR "" CACHE PATH
    "Unpacked csl-extras directory used by integration tests")
set(CSL_SDK_HOST_TOOLS_DIR "" CACHE PATH
    "Directory containing host-runnable cslc and cs_python wrappers")
set(CSL_INTEGRATION_MAX_PARALLELISM "10" CACHE STRING
    "Maximum number of SDK integration tests run concurrently")
set(CSL_INTEGRATION_TEST_TIMEOUT "600" CACHE STRING
    "Timeout in seconds for each SDK integration test")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(FindMLIR)
find_package(Squashfs REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CSL_MLIR_SOURCE_DIR}/include)
include_directories(${CSL_MLIR_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Own our export set. Upstream's helpers otherwise install these libraries into
# MLIRTargets, which belongs to the separately installed LLVM dependency.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)
set(LLVM_BUILD_TOOLS ON)

add_subdirectory(include)
add_subdirectory(lib)
# Capture library exports before AddMLIR also records tool executables.
get_property(CSLMLIR_LIBRARY_TARGETS GLOBAL PROPERTY MLIR_EXPORTS)
# The LSP support library uses LLVM's helper rather than AddMLIR.
list(APPEND CSLMLIR_LIBRARY_TARGETS CslLspServerLib)
add_subdirectory(tools)
if(CSL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()

foreach(library IN LISTS CSLMLIR_LIBRARY_TARGETS)
  target_include_directories(${library} PUBLIC
    $<BUILD_INTERFACE:${CSL_MLIR_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CSL_MLIR_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(${library} PUBLIC cxx_std_20)
  add_library(CSLMLIR::${library} ALIAS ${library})
endforeach()
install(TARGETS ${CSLMLIR_LIBRARY_TARGETS} EXPORT CSLMLIRTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS csl-opt csl-s2s csl-pp csl-lsp-server csl-lex-roundtrip
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/csl-mlir/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/csl-mlir
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.def" PATTERN "*.td")
install(DIRECTORY ${CSL_MLIR_BINARY_DIR}/include/csl-mlir/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/csl-mlir
  FILES_MATCHING PATTERN "*.inc")
# Host wrappers that run the SDK's cslc and cs_python from an OCI image.
install(PROGRAMS scripts/cerebras-sdk-oci/run-sdk-tool
  DESTINATION ${CMAKE_INSTALL_DATADIR}/csl-mlir/cerebras-sdk-oci)
install(PROGRAMS scripts/cerebras-sdk-oci/bin/cslc scripts/cerebras-sdk-oci/bin/cs_python
  DESTINATION ${CMAKE_INSTALL_DATADIR}/csl-mlir/cerebras-sdk-oci/bin)
install(FILES scripts/cerebras-sdk-oci/README.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/csl-mlir/cerebras-sdk-oci)
# The csl++ libraries, imported with --import-path=<prefix>/share/csl-mlir/libs.
install(DIRECTORY csl-libs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/csl-mlir/libs)
install(FILES LICENSE.TXT DESTINATION ${CMAKE_INSTALL_DATADIR}/csl-mlir)
if(CSL_THIRD_PARTY_LICENSES_DIR)
  install(DIRECTORY ${CSL_THIRD_PARTY_LICENSES_DIR}/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/csl-mlir/third-party-licenses)
endif()
configure_package_config_file(cmake/CSLMLIRConfig.cmake.in
  ${CSL_MLIR_BINARY_DIR}/CSLMLIRConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CSLMLIR
  PATH_VARS CMAKE_INSTALL_DATADIR)
write_basic_package_version_file(${CSL_MLIR_BINARY_DIR}/CSLMLIRConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT CSLMLIRTargets NAMESPACE CSLMLIR::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CSLMLIR)
install(FILES ${CSL_MLIR_BINARY_DIR}/CSLMLIRConfig.cmake
  ${CSL_MLIR_BINARY_DIR}/CSLMLIRConfigVersion.cmake cmake/VerifyLLVM.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CSLMLIR)

# Convenience target to build all csl-mlir tools without building all of LLVM.
add_custom_target(csl-all DEPENDS
  csl-opt csl-s2s csl-pp csl-lsp-server csl-lex-roundtrip
)

# ---------------------------------------------------------------------------
# csl-lsp-vscode: Package the VS Code extension with the bundled LSP server.
# Only allowed for static release builds to ensure a self-contained binary.
# The extension's sources in editors/vscode are copied into the build
# directory, which receives the server, the notices of the libraries linked
# into it (licenses/, so the target needs CSL_THIRD_PARTY_LICENSES_DIR) and,
# for a release, CSL_RELEASE_VERSION as the extension's version. The package
# is written to <build>/vscode/csl-lsp-<target>-<version>.vsix. esbuild bundles
# the extension's dependencies, so vsce packages no node_modules.
# ---------------------------------------------------------------------------
set(CSL_VSCODE_SOURCE_DIR "${CSL_MLIR_SOURCE_DIR}/editors/vscode")
set(CSL_VSCODE_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/vscode")
# The files of the extension; anything else in the source directory, such as
# node_modules/ of a local development setup, is not packaged.
set(CSL_VSCODE_FILES
  .vscodeignore LICENSE README.md language-configuration.json
  package.json package-lock.json tsconfig.json)
set(CSL_VSCODE_DIRECTORIES src syntaxes)

# Map CMake system info to vsce --target platform strings.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CSL_VSCE_TARGET "linux-arm64")
  else()
    set(CSL_VSCE_TARGET "linux-x64")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CSL_VSCE_TARGET "darwin-arm64")
  else()
    set(CSL_VSCE_TARGET "darwin-x64")
  endif()
else()
  set(CSL_VSCE_TARGET "unknown")
endif()

find_program(NPM_EXECUTABLE npm)
if(NPM_EXECUTABLE
   AND NOT BUILD_SHARED_LIBS
   AND CMAKE_BUILD_TYPE STREQUAL "Release"
   AND CSL_THIRD_PARTY_LICENSES_DIR)
  set(stage_commands
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CSL_VSCODE_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CSL_VSCODE_STAGE_DIR}/bin")
  foreach(file IN LISTS CSL_VSCODE_FILES)
    list(APPEND stage_commands COMMAND ${CMAKE_COMMAND} -E copy
      "${CSL_VSCODE_SOURCE_DIR}/${file}" "${CSL_VSCODE_STAGE_DIR}/${file}")
  endforeach()
  foreach(directory IN LISTS CSL_VSCODE_DIRECTORIES)
    list(APPEND stage_commands COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CSL_VSCODE_SOURCE_DIR}/${directory}" "${CSL_VSCODE_STAGE_DIR}/${directory}")
  endforeach()
  if(CSL_RELEASE_VERSION)
    list(APPEND stage_commands COMMAND ${CMAKE_COMMAND} -E chdir
      "${CSL_VSCODE_STAGE_DIR}" ${NPM_EXECUTABLE} pkg set
      "version=${CSL_RELEASE_VERSION}")
  endif()
  add_custom_target(csl-lsp-vscode
    DEPENDS csl-lsp-server
    ${stage_commands}
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:csl-lsp-server>" "${CSL_VSCODE_STAGE_DIR}/bin/csl-lsp-server"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CSL_THIRD_PARTY_LICENSES_DIR}" "${CSL_VSCODE_STAGE_DIR}/licenses"
    COMMAND ${CMAKE_COMMAND} -E chdir "${CSL_VSCODE_STAGE_DIR}"
      ${NPM_EXECUTABLE} ci --no-audit --no-fund
    COMMAND ${CMAKE_COMMAND} -E chdir "${CSL_VSCODE_STAGE_DIR}"
      ${NPM_EXECUTABLE} run compile
    COMMAND ${CMAKE_COMMAND} -E chdir "${CSL_VSCODE_STAGE_DIR}"
      ${NPM_EXECUTABLE} exec -- vsce package --no-dependencies
      --target ${CSL_VSCE_TARGET}
    COMMENT "Packaging CSL VS Code extension for ${CSL_VSCE_TARGET}"
    VERBATIM
  )
else()
  if(NOT NPM_EXECUTABLE)
    message(STATUS "csl-lsp-vscode: npm not found, target disabled")
  elseif(BUILD_SHARED_LIBS)
    message(STATUS "csl-lsp-vscode: requires static build (BUILD_SHARED_LIBS=OFF)")
  elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "csl-lsp-vscode: requires Release build (CMAKE_BUILD_TYPE=Release)")
  else()
    message(STATUS "csl-lsp-vscode: requires CSL_THIRD_PARTY_LICENSES_DIR (set by the Conan build)")
  endif()
endif()
