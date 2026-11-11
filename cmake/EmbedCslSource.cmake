# Turn a .csl file into a C++ raw string literal so the compiler can carry its
# own runtime sources in the binary. Keeping the .csl files as the on-disk
# source of truth (rather than hand-maintaining C++ literals) means they stay
# editable, diffable, and syntax-highlighted like any other CSL.
#
# Invoked at build time as:
#   cmake -DINPUT=<file.csl> -DOUTPUT=<file.inc> -DSYMBOL=<name>
#         -P EmbedCslSource.cmake

file(READ "${INPUT}" _contents)

# A raw string literal needs a delimiter that cannot appear in the payload.
# `)csl"` is not valid CSL, so it cannot occur in these files.
set(_delim "csl")
if(_contents MATCHES "\\)${_delim}\"")
  message(FATAL_ERROR
    "${INPUT} contains the raw-string delimiter )${_delim}\" - pick another")
endif()

file(WRITE "${OUTPUT}"
"// Generated from ${INPUT} by EmbedCslSource.cmake. Do not edit.\n"
"static constexpr const char *${SYMBOL} = R\"${_delim}(${_contents})${_delim}\";\n")
