# Reject a different SDK even when its LLVM release number is identical.
function(csl_verify_llvm_revision expected_revision)
  set(manifest "${LLVM_INSTALL_PREFIX}/share/llvm-sdk/provenance.json")
  if(NOT EXISTS "${manifest}")
    message(FATAL_ERROR "LLVM is missing Conan SDK provenance: ${manifest}")
  endif()
  file(READ "${manifest}" provenance)
  string(JSON schema ERROR_VARIABLE schema_error GET "${provenance}" schema)
  string(JSON revision ERROR_VARIABLE revision_error GET "${provenance}" revision)
  if(schema_error OR revision_error OR NOT schema STREQUAL "1" OR
     NOT revision STREQUAL expected_revision)
    message(FATAL_ERROR "csl-mlir requires LLVM gitlink ${expected_revision}; SDK records '${revision}'.")
  endif()
endfunction()
