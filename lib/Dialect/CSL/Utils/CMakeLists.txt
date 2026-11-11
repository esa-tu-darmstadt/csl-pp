# CSL Utils Library

# The coroutine runtime is compiled into the binary rather than read from disk.
# It is an implementation detail of the transpiler -- user programs never name
# it -- so it must resolve identically regardless of where csl-pp is invoked
# from or installed to.
set(CORO_RUNTIME_DIR
  ${PROJECT_SOURCE_DIR}/lib/Dialect/TypedCSL/Transforms/Fiber/runtime)
set(CORO_RUNTIME_INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/CoroRuntimeSources)

set(CORO_RUNTIME_INCS)
foreach(entry "wse2.csl:kFiberRuntimeWse2" "wse3.csl:kFiberRuntimeWse3"
              "resources.csl:kAllocRuntime")
  string(REPLACE ":" ";" entry ${entry})
  list(GET entry 0 _file)
  list(GET entry 1 _symbol)
  set(_out ${CORO_RUNTIME_INC_DIR}/${_file}.inc)
  add_custom_command(
    OUTPUT ${_out}
    COMMAND ${CMAKE_COMMAND}
      -DINPUT=${CORO_RUNTIME_DIR}/${_file}
      -DOUTPUT=${_out}
      -DSYMBOL=${_symbol}
      -P ${PROJECT_SOURCE_DIR}/cmake/EmbedCslSource.cmake
    DEPENDS ${CORO_RUNTIME_DIR}/${_file}
            ${PROJECT_SOURCE_DIR}/cmake/EmbedCslSource.cmake
    COMMENT "Embedding ${_file}"
    VERBATIM)
  list(APPEND CORO_RUNTIME_INCS ${_out})
endforeach()

add_custom_target(CoroRuntimeSources DEPENDS ${CORO_RUNTIME_INCS})

# The release and commit printed by --version. configure_file rewrites the
# file only when they change, so a new commit recompiles only Version.cpp.
configure_file(VersionInfo.inc.in ${CMAKE_CURRENT_BINARY_DIR}/VersionInfo.inc
  @ONLY)

add_mlir_library(MLIRCSLUtils
  CommandLineOptions.cpp
  CoroRuntimeSources.cpp
  CslModuleResolver.cpp
  ParameterParsing.cpp
  ResourceBitmaps.cpp
  Version.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/csl-mlir/Dialect/CSL/Utils

  DEPENDS
  CoroRuntimeSources

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CSLSifReader
  MLIRCSLDialect
  MLIRSupport
)

target_include_directories(MLIRCSLUtils PRIVATE ${CORO_RUNTIME_INC_DIR}
  ${CMAKE_CURRENT_BINARY_DIR})
