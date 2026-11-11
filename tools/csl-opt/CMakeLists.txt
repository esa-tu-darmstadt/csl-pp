set (LIBS
    MLIRIR
    MLIROptLib
    MLIRCSLPipelines
    MLIRCSLUtils
    MLIRParsedCSLDialect
    MLIRTypedCSLDialect
    MLIRParsedCSLTransforms
    MLIRTypedCSLTransforms
    MLIRParsedCSLToTypedCSL
)

add_llvm_executable(csl-opt csl-opt.cpp
    DEPENDS 
    ${LIBS}
)

# llvm_update_compile_flags(csl-opt)
target_link_libraries(csl-opt PRIVATE ${LIBS})

# mlir_check_all_link_libraries(csl-opt)
