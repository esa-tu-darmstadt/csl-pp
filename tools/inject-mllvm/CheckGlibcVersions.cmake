# Fails if LIBRARY needs a glibc symbol version newer than 2.MAX_GLIBC_MINOR,
# which the Cerebras SDK container would not provide.
#
#   cmake -DOBJDUMP=<objdump> -DLIBRARY=<library> -DMAX_GLIBC_MINOR=<n>
#         -P CheckGlibcVersions.cmake

if(NOT OBJDUMP)
  message(FATAL_ERROR "no objdump to check the glibc versions of ${LIBRARY}")
endif()

execute_process(
  COMMAND "${OBJDUMP}" -T "${LIBRARY}"
  OUTPUT_VARIABLE symbols
  RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "'${OBJDUMP} -T ${LIBRARY}' failed")
endif()

string(REGEX MATCHALL "GLIBC_2\\.[0-9]+" versions "${symbols}")
list(REMOVE_DUPLICATES versions)
foreach(version IN LISTS versions)
  string(REGEX REPLACE "GLIBC_2\\." "" minor "${version}")
  if(minor GREATER MAX_GLIBC_MINOR)
    message(FATAL_ERROR
      "${LIBRARY} needs ${version}, but the Cerebras SDK container only "
      "provides glibc 2.${MAX_GLIBC_MINOR}; bind the symbol to an older "
      "version in inject_mllvm.c")
  endif()
endforeach()
