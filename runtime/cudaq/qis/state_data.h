/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/operators/matrix.h"
#include <algorithm>
#include <bitset>
#include <complex>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace cudaq {

/// Enum to specify the initial quantum state.
enum class InitialState { ZERO, UNIFORM };

/// @brief Encapsulates a list of tensors (data pointer and dimensions).
// Note: tensor data is expected in column-major.
using TensorStateData =
    std::vector<std::pair<const void *, std::vector<std::size_t>>>;
/// @brief state_data is a variant type
/// encoding different forms of user state vector data
/// we support.
using state_data = std::variant<std::vector<std::complex<double>>,
                                std::vector<std::complex<float>>,
                                std::pair<std::complex<double> *, std::size_t>,
                                std::pair<std::complex<float> *, std::size_t>,
                                complex_matrix, TensorStateData>;
} // namespace cudaq
