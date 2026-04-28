/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu.h"
#include "common/ExecutionContext.h"
#include "common/Timing.h"
#include "cudaq/operators.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstring>

using namespace cudaq_internal::compiler;
using namespace cudaq;

CUDAQ_INSTANTIATE_REGISTRY(cudaq::ModuleLauncher::RegistryType)

/// Execute a JIT-compiled kernel with provided arguments.
///
/// Handles argument marshaling via `argsCreator` (if not fully specialized) and
/// result buffer allocation.
KernelThunkResultType
launchCompiledModule(const cudaq::CompiledModule &compiled,
                     const std::vector<void *> &rawArgs) {
  auto funcPtr = compiled.getJit()->getFn();
  const auto &resultInfo = compiled.getResultInfo();
  if (!compiled.isFullySpecialized()) {
    // Pack args at runtime via argsCreator, then call the thunk.
    auto argsCreator = compiled.getArgsCreator();
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    // If the kernel has a result, copy it from the packed buffer into
    // rawArgs.back() (where the caller expects to find it).
    if (resultInfo.hasResult()) {
      auto offset = compiled.getReturnOffset().value();
      std::memcpy(rawArgs.back(), static_cast<char *>(buff) + offset,
                  resultInfo.getBufferSize());
    }
    std::free(buff);
    return {nullptr, 0};
  }
  if (resultInfo.hasResult()) {
    // Fully specialized with result: rawArgs.back() is the pre-allocated
    // result buffer; pass it directly to the thunk.
    void *buff = const_cast<void *>(rawArgs.back());
    return reinterpret_cast<KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
  }
  // Fully specialized, no result.
  funcPtr();
  return {nullptr, 0};
}

cudaq::KernelThunkResultType
QPU::launchModule(const std::string &name, mlir::ModuleOp &module,
                  const std::vector<void *> &rawArgs) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `launchModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchModule", name);
  auto compiled = launcher->compileModule(name, module, rawArgs, true);
  return launchCompiledModule(compiled, rawArgs);
}

CompiledModule QPU::specializeModule(const std::string &name,
                                     mlir::ModuleOp &module,
                                     const std::vector<void *> &rawArgs,
                                     bool isEntryPoint) {
  auto launcher = registry::get<ModuleLauncher>("default");
  if (!launcher)
    throw std::runtime_error(
        "No ModuleLauncher registered with name 'default'. This may be a "
        "result of attempting to use `specializeModule` outside Python.");
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::specializeModule", name);
  return launcher->compileModule(name, module, rawArgs, isEntryPoint);
}

void QPU::handleObservation(ExecutionContext &context) const {
  // The reason for the 2 if checks is simply to do a flushGateQueue() before
  // initiating the trace.
  bool execute = context.name == "observe";
  if (execute) {
    ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                           "handleObservation flushGateQueue()");
    getExecutionManager()->flushGateQueue();
  }
  if (execute) {
    ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                           "QPU::handleObservation (after flush)");
    double sum = 0.0;
    if (!context.spin.has_value())
      throw std::runtime_error("[QPU] Observe ExecutionContext specified "
                               "without a cudaq::spin_op.");

    std::vector<cudaq::ExecutionResult> results;
    cudaq::spin_op &H = context.spin.value();
    assert(cudaq::spin_op::canonicalize(H) == H);

    // If the backend supports the observe task, let it compute the
    // expectation value instead of manually looping over terms, applying
    // basis change ops, and computing <ZZ..ZZZ>
    if (context.canHandleObserve) {
      auto [exp, data] = cudaq::measure(H);
      context.expectationValue = exp;
      context.result = data;
    } else {

      // Loop over each term and compute coeff * <term>
      for (const auto &term : H) {
        if (term.is_identity())
          sum += term.evaluate_coefficient().real();
        else {
          // This takes a longer time for the first iteration unless
          // flushGateQueue() is called above.
          auto [exp, data] = cudaq::measure(term);
          results.emplace_back(data.to_map(), term.get_term_id(), exp);
          sum += term.evaluate_coefficient().real() * exp;
        }
      };

      context.expectationValue = sum;
      context.result = cudaq::sample_result(sum, results);
    }
  }
}
