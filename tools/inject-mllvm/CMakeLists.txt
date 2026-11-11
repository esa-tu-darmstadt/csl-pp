# libinject_mllvm.so — LD_PRELOAD interposer that appends
# `-fstack-size-section` (and any other flags listed in inject_mllvm.c) to
# every clang::CompilerInvocation produced inside cslc-backend.  csl-pp loads
# it into cslc so that per-PE ELFs carry a `.stack_sizes` section, from which
# it sizes the stacks of async tasks.
#
# cslc-backend runs inside the Cerebras SDK container, whose glibc is older
# than the host's. inject_mllvm.c binds its glibc symbols to old versions, and
# a post-build check rejects any symbol version the container's glibc 2.30
# does not provide, so the host compiler can build it.

add_library(inject_mllvm MODULE inject_mllvm.c)
set_target_properties(inject_mllvm PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "inject_mllvm"
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  C_VISIBILITY_PRESET default)
target_link_libraries(inject_mllvm PRIVATE ${CMAKE_DL_LIBS})

add_custom_command(TARGET inject_mllvm POST_BUILD
  COMMAND "${CMAKE_COMMAND}"
          -DOBJDUMP=${CMAKE_OBJDUMP}
          -DLIBRARY=$<TARGET_FILE:inject_mllvm>
          -DMAX_GLIBC_MINOR=30
          -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckGlibcVersions.cmake"
  VERBATIM)

# csl-pp looks for the library in ../lib relative to its own executable.
install(TARGETS inject_mllvm
  LIBRARY DESTINATION lib
  COMPONENT inject-mllvm)
