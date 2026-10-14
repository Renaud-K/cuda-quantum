# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# cudaq_realtime_add_device_code(<target> SYMBOL <name> SOURCES <src>...
#                                [ARCHITECTURES <arch>...]
#                                [INCLUDE_DIRECTORIES <dir>...]
#                                [COMPILE_DEFINITIONS <def>...])
#
# Builds device code that the DEVICE + UNIFIED dispatcher links at runtime
# (a transport's device hooks, or an application's DEVICE_CALL handlers) and
# embeds it in <target>.  Each source becomes one relocatable fatbin with SASS
# and LTO-IR for every architecture, as realtime/docs/device_code_contract.md
# requires; the sources are never device-linked.  <target> then has
#
#   extern "C" const cudaq_device_image_t <name>[];
#   extern "C" const size_t <name>_count;
#
# declared in the generated header <name>.h on its include path, ready to pass
# to cudaq_dispatcher_set_device_code (or to return from get_gpu_dataplane).
# Both symbols have hidden visibility.
#
# ARCHITECTURES defaults to CUDA_TARGET_ARCHS, then CMAKE_CUDA_ARCHITECTURES.
# This is a convenience: any build that follows the contract works, CMake or
# not.

include_guard(GLOBAL)

function(cudaq_realtime_add_device_code target)
  cmake_parse_arguments(PARSE_ARGV 1 arg ""
    "SYMBOL" "SOURCES;ARCHITECTURES;INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS")
  if(NOT arg_SYMBOL OR NOT arg_SOURCES)
    message(FATAL_ERROR
      "cudaq_realtime_add_device_code(${target}) needs SYMBOL and SOURCES")
  endif()

  set(archs ${arg_ARCHITECTURES})
  if(NOT archs)
    if(CUDA_TARGET_ARCHS)
      set(archs ${CUDA_TARGET_ARCHS})
    else()
      set(archs ${CMAKE_CUDA_ARCHITECTURES})
    endif()
  endif()
  list(TRANSFORM archs REPLACE "-.*$" "")
  if(NOT archs OR archs MATCHES "^(OFF|all|all-major|native)$")
    message(FATAL_ERROR
      "cudaq_realtime_add_device_code(${target}): no explicit CUDA "
      "architectures; pass ARCHITECTURES")
  endif()

  set(fatbins ${target}-${arg_SYMBOL}-device-code)
  add_library(${fatbins} OBJECT ${arg_SOURCES})
  set_target_properties(${fatbins} PROPERTIES
    CUDA_FATBIN_COMPILATION ON
    CUDA_ARCHITECTURES OFF)
  target_compile_options(${fatbins} PRIVATE -rdc=true)
  foreach(arch IN LISTS archs)
    target_compile_options(${fatbins} PRIVATE
      "--generate-code=arch=compute_${arch},code=[sm_${arch},lto_${arch}]")
  endforeach()
  if(CUDAQ_REALTIME_INCLUDE_DIR)
    target_include_directories(${fatbins} PRIVATE ${CUDAQ_REALTIME_INCLUDE_DIR})
  endif()
  target_include_directories(${fatbins} PRIVATE ${arg_INCLUDE_DIRECTORIES})
  target_compile_definitions(${fatbins} PRIVATE ${arg_COMPILE_DEFINITIONS})

  set(gen_dir ${CMAKE_CURRENT_BINARY_DIR}/${target}-device-code)
  set(gen_src ${gen_dir}/${arg_SYMBOL}.cpp)
  set(gen_hdr ${gen_dir}/${arg_SYMBOL}.h)
  set(script ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CudaqRealtimeEmbedDeviceCode.cmake)
  add_custom_command(
    OUTPUT ${gen_src} ${gen_hdr}
    COMMAND ${CMAKE_COMMAND}
      -DSYMBOL=${arg_SYMBOL} -DSOURCE=${gen_src} -DHEADER=${gen_hdr}
      "-DFATBINS=$<JOIN:$<TARGET_OBJECTS:${fatbins}>,|>"
      -P ${script}
    DEPENDS ${fatbins} $<TARGET_OBJECTS:${fatbins}> ${script}
    COMMENT "Embedding device code ${arg_SYMBOL} in ${target}"
    VERBATIM)
  target_sources(${target} PRIVATE ${gen_src})
  target_include_directories(${target} PRIVATE ${gen_dir})
  if(CUDAQ_REALTIME_INCLUDE_DIR)
    target_include_directories(${target} PRIVATE ${CUDAQ_REALTIME_INCLUDE_DIR})
  endif()
endfunction()
