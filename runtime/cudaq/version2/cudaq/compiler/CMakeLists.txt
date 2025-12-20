# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS
  native
  Support
  Core
)

add_mlir_library(cudaq-compiler SHARED 
    # module.cpp
    mlir_module.cpp
    llvm_module.cpp
    execution_engine.cpp
    compiler.cpp
    opt.cpp
    translate.cpp
    arg_conversion.cpp
    utils.cpp
)

target_compile_options(cudaq-compiler PRIVATE -fexceptions)

# Link against MLIR libraries
target_link_libraries(cudaq-compiler
    PRIVATE
    fmt::fmt-header-only
    spdlog::spdlog
    MLIRParser
    MLIRIR
    MLIRPass
    MLIRSupport
    QuakeDialect
    OptCodeGen
    MLIROptLib
    MLIRArithDialect 
    MLIRFuncDialect
    CCDialect
    OptTransforms 
    MLIRMathDialect
    MLIRTransforms
    MLIRExecutionEngine
    MLIRTranslateLib
    MLIRTargetLLVMIRExport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    LLVMSupport
    LLVMCore
    LLVMTarget
    LLVMMC
    LLVMPasses
    LLVMAnalysis
    LLVMTransformUtils
    LLVMBitWriter
    LLVMOrcJIT
)

target_link_options(cudaq-compiler PRIVATE -Wl,--no-as-needed)

# Include directories
target_include_directories(cudaq-compiler PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/cudaq/version2
)
