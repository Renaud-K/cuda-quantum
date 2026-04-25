/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>

namespace cudaq {
class ExecutionContext;

namespace platform {

/// @brief Execute the given function within the given execution context,
/// delegating to the current quantum_platform. This free function avoids
/// a header dependency on platform.h from QPU implementation headers.
void with_execution_context(ExecutionContext &ctx, std::function<void()> f);

} // namespace platform
} // namespace cudaq
