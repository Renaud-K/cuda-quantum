/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "device_link.h"

#include <cstdio>
#include <nvJitLink.h>

namespace cudaq::realtime {
namespace {

void print_error_log(nvJitLinkHandle handle) {
  size_t size = 0;
  if (nvJitLinkGetErrorLogSize(handle, &size) != NVJITLINK_SUCCESS || size <= 1)
    return;
  std::vector<char> log(size);
  if (nvJitLinkGetErrorLog(handle, log.data()) == NVJITLINK_SUCCESS)
    fprintf(stderr, "%s\n", log.data());
}

// Says what failed, which nvJitLink version did the linking, and the linker's
// own log, which names the offending input and symbol.  An input rejected
// outright is most often one built by a newer CUDA toolkit, which nvJitLink
// reports as bare "bad input" for SASS, so that case gets a hint.
bool linked(nvJitLinkResult result, nvJitLinkHandle handle, const char *what,
            bool adding_input) {
  if (result == NVJITLINK_SUCCESS)
    return true;
  unsigned major = 0, minor = 0;
  nvJitLinkVersion(&major, &minor);
  fprintf(stderr,
          "ERROR: cudaq_dispatcher_set_device_code: nvJitLink %u.%u failed "
          "%s (nvJitLinkResult %d)%s\n",
          major, minor, what, static_cast<int>(result),
          adding_input ? "; images must come from a CUDA toolkit no newer "
                         "than this nvJitLink"
                       : "");
  print_error_log(handle);
  return false;
}

struct LinkerGuard {
  nvJitLinkHandle handle = nullptr;
  ~LinkerGuard() {
    if (handle)
      nvJitLinkDestroy(&handle);
  }
};

} // namespace

cudaq_status_t link_device_images(const std::vector<NamedDeviceImage> &images,
                                  cudaLibrary_t *out_library) {
  int device = 0;
  int major = 0;
  int minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  const std::string arch =
      "-arch=sm_" + std::to_string(major) + std::to_string(minor);
  const char *options[] = {arch.c_str(), "-lto"};

  LinkerGuard linker;
  if (nvJitLinkCreate(&linker.handle, 2, options) != NVJITLINK_SUCCESS) {
    fprintf(stderr, "ERROR: cudaq_dispatcher_set_device_code: "
                    "nvJitLinkCreate failed\n");
    linker.handle = nullptr;
    return CUDAQ_ERR_INTERNAL;
  }
  for (const auto &input : images) {
    const std::string what = "adding the " + input.name;
    if (!linked(nvJitLinkAddData(linker.handle, NVJITLINK_INPUT_FATBIN,
                                 input.image.data, input.image.size,
                                 input.name.c_str()),
                linker.handle, what.c_str(), true))
      return CUDAQ_ERR_INVALID_ARG;
  }
  const std::string what = "linking for " + arch.substr(6);
  if (!linked(nvJitLinkComplete(linker.handle), linker.handle, what.c_str(),
              false))
    return CUDAQ_ERR_INVALID_ARG;

  size_t cubin_size = 0;
  if (nvJitLinkGetLinkedCubinSize(linker.handle, &cubin_size) !=
      NVJITLINK_SUCCESS)
    return CUDAQ_ERR_INTERNAL;
  std::vector<char> cubin(cubin_size);
  if (nvJitLinkGetLinkedCubin(linker.handle, cubin.data()) != NVJITLINK_SUCCESS)
    return CUDAQ_ERR_INTERNAL;

  cudaLibrary_t library = nullptr;
  const cudaError_t err = cudaLibraryLoadData(&library, cubin.data(), nullptr,
                                              nullptr, 0, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_set_device_code: loading the linked "
            "module failed: %s\n",
            cudaGetErrorString(err));
    return CUDAQ_ERR_CUDA;
  }
  *out_library = library;
  return CUDAQ_OK;
}

} // namespace cudaq::realtime
