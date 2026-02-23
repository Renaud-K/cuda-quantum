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
## LLVM starting up static Options more than once.
add_library(cudaq-mlir-runtime
  SHARED
    ArgumentConversion.cpp
    Compiler.cpp
    JIT.cpp
    RuntimeMLIR.cpp
    RuntimeCppMLIR.cpp
    LayoutInfo.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-mlir-runtime)

target_include_directories(cudaq-mlir-runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
)

target_link_libraries(cudaq-mlir-runtime
  PRIVATE
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
    cudaq-common
    CUDAQTargetConfigUtil
    cudaq-logger
    fmt::fmt-header-only
)

install(TARGETS cudaq-mlir-runtime DESTINATION lib)
