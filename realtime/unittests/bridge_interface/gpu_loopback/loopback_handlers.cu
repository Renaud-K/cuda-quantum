/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file loopback_handlers.cu
/// @brief The test's handler image: one DEVICE_CALL handler whose increment
/// is module state the test sets in the linked module.

#include "cudaq/realtime/daemon/dispatcher/device_code.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

extern "C" __device__ unsigned loopback_increment = 0;

namespace {

/// Adds `loopback_increment` to every payload byte.
__device__ int loopback_add(const void *rx_slot, void *tx_slot,
                            std::size_t slot_size) {
  const auto *header = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  const auto *in = reinterpret_cast<const std::uint8_t *>(header + 1);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  auto *out = reinterpret_cast<std::uint8_t *>(response + 1);
  const std::uint32_t capacity =
      slot_size - sizeof(cudaq::realtime::RPCResponse);
  const std::uint32_t len =
      header->arg_len < capacity ? header->arg_len : capacity;
  for (std::uint32_t i = 0; i < len; ++i)
    out[i] = static_cast<std::uint8_t>(in[i] + loopback_increment);
  response->result_len = len;
  return 0;
}

} // namespace

CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(loopback_add)
