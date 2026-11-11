# CSL-MLIR Unit Tests

add_custom_target(CSLMLIRUnitTests)
set_target_properties(CSLMLIRUnitTests PROPERTIES FOLDER "CSLMLIR Unit Tests")

# Function to add unit tests that automatically get added to CSLMLIRUnitTests target
function(add_csl_mlir_unittest test_dirname)
  add_unittest(CSLMLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

# Add subdirectories
add_subdirectory(Support)
add_subdirectory(TypedCSL)
