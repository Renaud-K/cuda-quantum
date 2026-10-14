/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_loopback_transport.cu
/// @brief Device hooks of the loopback test provider, over rings in mapped
/// host memory.  Built as a device image that the provider returns from
/// get_gpu_dataplane, exactly as an out-of-tree provider would ship it.

#include "gpu_loopback_ctx.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"

namespace {

struct loopback_session {
  gpu_loopback_ctx ring;
  volatile int *shutdown_flag;
  std::uint32_t cursor;
};

static_assert(sizeof(loopback_session) <= CUDAQ_DEV_TRANSPORT_SESSION_BYTES &&
                  alignof(loopback_session) <=
                      CUDAQ_DEV_TRANSPORT_SESSION_ALIGN,
              "loopback_session must fit the kernel's session storage");

} // namespace

extern "C" __device__ void *
cudaq_dev_transport_attach(void *storage, void *ctx,
                           volatile int *shutdown_flag) {
  if (ctx == nullptr)
    return nullptr;
  auto &s = *static_cast<loopback_session *>(storage);
  s.ring = *static_cast<const gpu_loopback_ctx *>(ctx);
  s.shutdown_flag = shutdown_flag;
  s.cursor = 0;
  return &s;
}

extern "C" __device__ cudaq_rx_dev_status_t
cudaq_dev_rx_poll(void *session, void **out_request, void **out_response) {
  auto &s = *static_cast<loopback_session *>(session);
  while (s.ring.rx_flags[s.cursor] == 0) {
    if (*s.shutdown_flag)
      return CUDAQ_RX_DEV_SHUTDOWN;
  }
  __threadfence_system();
  *out_request = s.ring.rx_data + s.cursor * s.ring.stride;
  *out_response = s.ring.tx_data + s.cursor * s.ring.stride;
  return CUDAQ_RX_DEV_READY;
}

extern "C" __device__ void cudaq_dev_tx_publish(void *session, void *response) {
  auto &s = *static_cast<loopback_session *>(session);
  const auto slot = static_cast<std::uint32_t>(
      (static_cast<std::uint8_t *>(response) - s.ring.tx_data) / s.ring.stride);
  __threadfence_system();
  s.ring.tx_flags[slot] = reinterpret_cast<std::uint64_t>(response);
  s.ring.rx_flags[slot] = 0;
  s.cursor = (s.cursor + 1) % s.ring.num_slots;
}
