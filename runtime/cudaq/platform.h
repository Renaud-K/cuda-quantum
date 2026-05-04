/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/builder/kernel_builder.h"
#include "cudaq/platform/quantum_platform.h"

namespace cudaq {
quantum_platform *getQuantumPlatformInternal();

/// @brief Return the quantum platform provided by the linked platform library
/// @return
inline quantum_platform &get_platform() {
  return *getQuantumPlatformInternal();
}

/// @brief Return a typed view of the quantum platform after validating that
/// the QPU at index 0 is an instance of @p QPUType.
template <typename QPUType>
typed_platform<QPUType> get_platform() {
  auto &platform = get_platform();
  platform.assertQPUType<QPUType>();
  return typed_platform<QPUType>(platform);
}

/// @brief Return the number of QPUs (at runtime)
inline std::size_t platform_num_qpus() {
  return getQuantumPlatformInternal()->num_qpus();
}

/// @brief Return true if the quantum platform is remote.
inline bool is_remote_platform() {
  return getQuantumPlatformInternal()->is_remote();
}

/// @brief Return true if the quantum platform is a remote simulator.
inline bool is_remote_simulator_platform() {
  return getQuantumPlatformInternal()
      ->get_remote_capabilities()
      .isRemoteSimulator;
}

/// @brief Return true if the quantum platform is emulated.
inline bool is_emulated_platform() {
  return getQuantumPlatformInternal()->is_emulated();
}

// Declare this function, implemented elsewhere
std::string getQIR(const std::string &);

} // namespace cudaq

#include "cudaq/platform/qpu_types.h"
#if 0
#ifdef NVQPP_TARGET_QPU_TYPE
namespace cudaq::__internal__ {
/// Validates that the QPU instantiated by the platform matches the concrete
/// type expected for this target. Runs during static initialization, after
/// TargetSetter.
struct QPUTypeValidator {
  QPUTypeValidator() {
    auto *platform = getQuantumPlatformInternal();
    if (platform->num_qpus() > 0)
      platform->assertQPUType<NVQPP_TARGET_QPU_TYPE>();
  }
};
inline QPUTypeValidator qpuTypeValidator;
} // namespace cudaq::__internal__
#endif
#endif
