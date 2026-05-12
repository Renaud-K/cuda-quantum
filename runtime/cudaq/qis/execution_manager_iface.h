/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/operators.h"
#include "cudaq/qis/measure_result.h"
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

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
class kraus_channel;

/// Free-function interface to the active execution manager. These are thin
/// forwarders to the current `ExecutionManager *` returned by
/// `cudaq::getExecutionManager()`. User-kernel headers (qubit_qis.h, qudit.h,
/// etc.) call these free functions instead of including the heavy
/// `execution_manager.h`, which keeps that header out of their transitive
/// include set.
namespace execution_manager {
std::size_t allocateQudit(std::size_t quditLevels = 2);
void returnQudit(const QuditInfo &q);
void initializeState(const std::vector<QuditInfo> &targets, const state &state);

void apply(std::string_view gateName, const std::vector<double> &params,
           const std::vector<QuditInfo> &controls,
           const std::vector<QuditInfo> &targets, bool isAdjoint = false,
           const spin_op_term op = cudaq::spin_op::identity());
void applyNoise(const kraus_channel &channel,
                const std::vector<QuditInfo> &targets);
void reset(const QuditInfo &target);
int measure(const QuditInfo &target, const std::string &registerName = "");
SpinMeasureResult measure(const cudaq::spin_op &op);
void startCtrlRegion(const std::vector<std::size_t> &controlQubits);
void endCtrlRegion(std::size_t nControls);
void startAdjointRegion();
void endAdjointRegion();
} // namespace execution_manager
} // namespace cudaq
