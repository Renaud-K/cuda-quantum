/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file device_code.h
/// @brief Device images the DEVICE + UNIFIED dispatcher links at runtime.
///
/// The unified GPU dispatch kernel, the transport's device hooks and the
/// application's DEVICE_CALL handlers are built separately, usually in
/// different repositories, and only meet in one CUDA module that
/// cudaq_dispatcher_set_device_code links with nvJitLink.  Each party hands
/// its device code over as one or more `cudaq_device_image_t`.  What an image
/// must contain is specified in realtime/docs/device_code_contract.md.
///
/// This header has no CUDA dependency, so code that only embeds an image
/// (e.g. the sources cudaq_realtime_add_device_code generates) can include it.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// One relocatable fatbin (`nvcc -fatbin -rdc=true`), preferably carrying
/// LTO-IR for the target architectures.  Opaque bytes to the caller.
typedef struct {
  const void *data;
  size_t size;
} cudaq_device_image_t;

#ifdef __cplusplus
}
#endif

/// Prefix of the pointer globals CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION
/// defines; cudaq_dispatcher_get_device_function looks names up under it.
#define CUDAQ_REALTIME_DEVICE_FUNCTION_PREFIX "cudaq_rt_device_fn_"

#ifdef __CUDACC__
/// Export a DEVICE_CALL handler from a handler image, in the translation unit
/// that defines it.  CUDA can look up globals in a module by name but not
/// device functions, so this stores the handler's address in a global that
/// cudaq_dispatcher_get_device_function reads back from the linked module.
/// The image must be compiled with -rdc=true for the address to be valid.
#define CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(fn)                              \
  extern "C" __device__ void *cudaq_rt_device_fn_##fn = (void *)&fn;
#endif
