# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#include_directories(SYSTEM
#  ${LLVM_INCLUDE_DIRS}
#  ${MLIR_INCLUDE_DIRS})

add_library(llvm-support-headers INTERFACE)
target_include_directories(llvm-support-headers
  SYSTEM
  INTERFACE
    ${LLVM_INCLUDE_DIRS}
)

add_library(mlir-support-headers INTERFACE)
target_include_directories(mlir-support-headers
  SYSTEM
  INTERFACE
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(mlir-support-headers
  INTERFACE
    llvm-support-headers
)

add_subdirectory(logger)
add_subdirectory(common)
add_subdirectory(nvqir)
add_subdirectory(cudaq)
add_subdirectory(test)

# Install
# ==============================================================================

install (DIRECTORY cudaq DESTINATION include 
            FILES_MATCHING PATTERN "*.h" 
            PATTERN "nlopt-src" EXCLUDE)
install (DIRECTORY include/cudaq DESTINATION include FILES_MATCHING PATTERN "*.h")
install (DIRECTORY common DESTINATION include FILES_MATCHING PATTERN "*.h")
install (FILES nvqir/CircuitSimulator.h
               nvqir/QIRTypes.h
               nvqir/Gates.h
        DESTINATION include/nvqir)
install (FILES cudaq.h DESTINATION include)
