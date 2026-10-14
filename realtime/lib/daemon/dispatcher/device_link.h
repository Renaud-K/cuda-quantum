/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <string>
#include <vector>

namespace cudaq::realtime {

/// A device image to link, with the name nvJitLink uses for it in errors.
struct NamedDeviceImage {
  cudaq_device_image_t image;
  std::string name;
};

/// Link `images` into one module for the current CUDA device with nvJitLink
/// (device LTO, so the transport hooks inline into the kernel) and load it.
/// Prints the linker's error log to stderr on failure.
cudaq_status_t link_device_images(const std::vector<NamedDeviceImage> &images,
                                  cudaLibrary_t *out_library);

} // namespace cudaq::realtime
