# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)

## ----- Runtime MLIR Library -----
## We need support in the runtime for compiler-level things,
## but we have to collate this functionality in one place to avoid
## LLVM starting up static Options more than once. Create a library
## here called cudaq-mlir-runtime that cudaq-builder and cudaq-rest-qpu
## can link to and get MLIR/LLVM things uniformly

add_library(cudaq-mlir-runtime
  SHARED
    ArgumentConversion.cpp
    CodeGenConfig.cpp
    Environment.cpp
    JIT.cpp
    RuntimeMLIR.cpp
    RuntimeCppMLIR.cpp
    RunTheKernel.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-mlir-runtime)

target_include_directories(cudaq-mlir-runtime
  PRIVATE 
    ${CMAKE_SOURCE_DIR}/runtime/common)

target_link_libraries(cudaq-mlir-runtime
  PUBLIC
    nvqir
    CCDialect
    QuakeDialect
    OptCodeGen
    OptTransforms
    MLIRTranslateLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTranslateLib
    MLIRSupport
    MLIROptLib
    MLIRExecutionEngine
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
  PRIVATE
    cudaq
    cudaq-logger
)

install(TARGETS cudaq-mlir-runtime DESTINATION lib)

