/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file bridge_impl.cpp
/// @brief Test-only bridge provider with a GPU data plane and no network.
///
/// Its rings are mapped pinned host memory: the test plays the wire by
/// writing requests into the RX ring and reading answers out of the TX ring,
/// whose host views it gets from get_transport_context(RING_BUFFER).  The
/// device side is gpu_loopback_transport.cu, served as a device image through
/// get_gpu_dataplane, so the DEVICE + UNIFIED dispatcher can be exercised end
/// to end (three images linked at runtime) without a NIC.

#include "gpu_loopback_ctx.h"
#include "gpu_loopback_transport_images.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <cstring>
#include <cuda_runtime.h>
#include <new>

namespace {

constexpr std::uint32_t kNumSlots = 8;
constexpr std::uint32_t kSlotSize = 256;

struct LoopbackBridge {
  std::uint64_t *rx_flags = nullptr; ///< Host views, mapped
  std::uint64_t *tx_flags = nullptr;
  std::uint8_t *rx_data = nullptr;
  std::uint8_t *tx_data = nullptr;
  gpu_loopback_ctx device_view{};
  void *device_ctx = nullptr; ///< Staged copy of device_view

  ~LoopbackBridge() {
    cudaFree(device_ctx);
    cudaFreeHost(rx_flags);
    cudaFreeHost(tx_flags);
    cudaFreeHost(rx_data);
    cudaFreeHost(tx_data);
  }
};

template <typename T>
bool alloc_mapped(T **host, T **device, size_t bytes) {
  void *h = nullptr;
  void *d = nullptr;
  if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  *host = static_cast<T *>(h);
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess)
    return false;
  std::memset(h, 0, bytes);
  *device = static_cast<T *>(d);
  return true;
}

} // namespace

extern "C" {

static cudaq_status_t loopback_create(cudaq_realtime_bridge_handle_t *handle,
                                      int, char **) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *b = new (std::nothrow) LoopbackBridge();
  if (!b)
    return CUDAQ_ERR_INTERNAL;
  auto &dv = b->device_view;
  std::uint64_t *d_rx_flags = nullptr;
  std::uint64_t *d_tx_flags = nullptr;
  const size_t flags = kNumSlots * sizeof(std::uint64_t);
  const size_t data = kNumSlots * kSlotSize;
  const bool ok = alloc_mapped(&b->rx_flags, &d_rx_flags, flags) &&
                  alloc_mapped(&b->tx_flags, &d_tx_flags, flags) &&
                  alloc_mapped(&b->rx_data, &dv.rx_data, data) &&
                  alloc_mapped(&b->tx_data, &dv.tx_data, data) &&
                  cudaMalloc(&b->device_ctx, sizeof(dv)) == cudaSuccess;
  dv.rx_flags = d_rx_flags;
  dv.tx_flags = d_tx_flags;
  dv.stride = kSlotSize;
  dv.num_slots = kNumSlots;
  if (!ok || cudaMemcpy(b->device_ctx, &dv, sizeof(dv),
                        cudaMemcpyHostToDevice) != cudaSuccess) {
    delete b;
    return CUDAQ_ERR_CUDA;
  }
  *handle = b;
  return CUDAQ_OK;
}

static cudaq_status_t loopback_destroy(cudaq_realtime_bridge_handle_t handle) {
  delete static_cast<LoopbackBridge *>(handle);
  return CUDAQ_OK;
}

// RING_BUFFER is how the test reaches the rings to play the wire; the
// dispatcher only ever asks for the GPU data plane.
static cudaq_status_t
loopback_get_transport_context(cudaq_realtime_bridge_handle_t handle,
                               cudaq_realtime_transport_context_t type,
                               void *out) {
  if (!handle || !out)
    return CUDAQ_ERR_INVALID_ARG;
  if (type != RING_BUFFER)
    return CUDAQ_ERR_UNSUPPORTED;
  auto *b = static_cast<LoopbackBridge *>(handle);
  auto *ring = static_cast<cudaq_ringbuffer_t *>(out);
  ring->rx_flags = b->device_view.rx_flags;
  ring->tx_flags = b->device_view.tx_flags;
  ring->rx_data = b->device_view.rx_data;
  ring->tx_data = b->device_view.tx_data;
  ring->rx_stride_sz = kSlotSize;
  ring->tx_stride_sz = kSlotSize;
  ring->rx_flags_host = b->rx_flags;
  ring->tx_flags_host = b->tx_flags;
  ring->rx_data_host = b->rx_data;
  ring->tx_data_host = b->tx_data;
  return CUDAQ_OK;
}

static cudaq_status_t loopback_noop(cudaq_realtime_bridge_handle_t handle) {
  return handle ? CUDAQ_OK : CUDAQ_ERR_INVALID_ARG;
}

static cudaq_status_t
loopback_get_ring_geometry(cudaq_realtime_bridge_handle_t handle,
                           uint32_t *out_num_slots, uint32_t *out_slot_size) {
  if (!handle || !out_num_slots || !out_slot_size)
    return CUDAQ_ERR_INVALID_ARG;
  *out_num_slots = kNumSlots;
  *out_slot_size = kSlotSize;
  return CUDAQ_OK;
}

static cudaq_status_t
loopback_get_gpu_dataplane(cudaq_realtime_bridge_handle_t handle,
                           cudaq_gpu_dataplane_t *out) {
  if (!handle || !out)
    return CUDAQ_ERR_INVALID_ARG;
  out->ctx = static_cast<LoopbackBridge *>(handle)->device_ctx;
  out->device_code = gpu_loopback_transport_images;
  out->device_code_count = gpu_loopback_transport_images_count;
  return CUDAQ_OK;
}

cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      loopback_create,
      loopback_destroy,
      loopback_get_transport_context,
      /*connect=*/loopback_noop,
      /*launch=*/loopback_noop,
      /*disconnect=*/loopback_noop,
      /*get_cpu_dataplane=*/nullptr,
      /*get_endpoint_info=*/nullptr,
      loopback_get_ring_geometry,
      /*set_function_table=*/nullptr,
      loopback_get_gpu_dataplane,
  };
  return &interface;
}
}
