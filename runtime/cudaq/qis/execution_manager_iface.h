/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cstddef>
#include <vector>
#pragma once

namespace cudaq {
/// A QuditInfo is a type encoding the number of \a levels and the \a id of the
/// qudit to the ExecutionManager.
struct QuditInfo {
  std::size_t levels = 0;
  std::size_t id = 0;
  QuditInfo(std::size_t _levels, std::size_t _id) : levels(_levels), id(_id) {}
  bool operator==(const QuditInfo &other) const {
    return levels == other.levels && id == other.id;
  }
};

class state;

namespace execution_manager_iface {
std::size_t allocateQudit(std::size_t quditLevels = 2);
void returnQudit(const QuditInfo &q);
void initializeState(const std::vector<QuditInfo> &targets, const state &state);
} // namespace execution_manager_iface
} // namespace cudaq
