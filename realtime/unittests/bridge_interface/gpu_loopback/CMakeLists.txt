# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# GPU data plane tests over a test-only loopback provider
# ==============================================================================
# The DEVICE + UNIFIED dispatcher end to end -- three device images linked at
# runtime -- without a NIC.  Needs a GPU at run time; skips without one.

# Test-only provider: its device hooks are an embedded image, as an
# out-of-tree provider would ship them.
add_library(cudaq-realtime-bridge-gpu-loopback SHARED bridge_impl.cpp)
target_include_directories(cudaq-realtime-bridge-gpu-loopback PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAQ_REALTIME_INCLUDE_DIR})
target_link_libraries(cudaq-realtime-bridge-gpu-loopback PRIVATE CUDA::cudart)
cudaq_realtime_add_device_code(cudaq-realtime-bridge-gpu-loopback
  SYMBOL gpu_loopback_transport_images
  SOURCES gpu_loopback_transport.cu
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

# The application: plain host code plus its handler image.
add_executable(test_gpu_dataplane test_gpu_dataplane.cpp)
target_include_directories(test_gpu_dataplane PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR})
target_compile_definitions(test_gpu_dataplane PRIVATE
  GPU_LOOPBACK_PROVIDER_PATH="$<TARGET_FILE:cudaq-realtime-bridge-gpu-loopback>")
target_link_libraries(test_gpu_dataplane PRIVATE
  GTest::gtest_main
  cudaq-realtime
  CUDA::cudart)
cudaq_realtime_add_device_code(test_gpu_dataplane
  SYMBOL loopback_handler_images
  SOURCES loopback_handlers.cu)

# The provider is dlopen'd, not linked.
add_dependencies(test_gpu_dataplane cudaq-realtime-bridge-gpu-loopback)

cudaq_gtest_discover_tests(test_gpu_dataplane DISCOVERY_TIMEOUT 120)
message(STATUS "  - test_gpu_dataplane (DEVICE unified over a GPU data plane)")
