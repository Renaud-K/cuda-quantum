/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

/// What the loopback provider's device hooks read, staged in device memory.
/// The rings live in mapped pinned host memory: the test writes a request and
/// sets rx_flags[slot] from the host, and the hooks answer through tx_flags.
struct gpu_loopback_ctx {
  volatile std::uint64_t *rx_flags; ///< Device views of the mapped rings
  volatile std::uint64_t *tx_flags;
  std::uint8_t *rx_data;
  std::uint8_t *tx_data;
  std::size_t stride;
  std::uint32_t num_slots;
};
