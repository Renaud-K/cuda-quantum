/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform/qpu.h"

namespace cudaq {

/// The DefaultQPU models a simulated QPU by specifically
/// targeting the QIS ExecutionManager.
class DefaultQPU : public QPU {
public:
  DefaultQPU() = default;
  ~DefaultQPU() override;

  void enqueue(QuantumTask &task) override;

  KernelThunkResultType
  launchKernel(const std::string &name, KernelThunkType kernelFunc, void *args,
               std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override;

  void configureExecutionContext(ExecutionContext &context) const override;
  void beginExecution() override;
  void endExecution() override;
  void finalizeExecutionContext(ExecutionContext &context) const override;
};

} // namespace cudaq
