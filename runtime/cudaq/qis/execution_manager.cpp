/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "common/ExecutionContext.h"
#include "common/PluginUtils.h"
#include "cudaq/algorithms/policy_cpos.h"
#include "cudaq/algorithms/policy_dispatch.h"
#include "cudaq/qis/state.h"
#include "execution_manager_iface.h"

using namespace cudaq;

static ExecutionManager *executionManager;

void cudaq::setExecutionManagerInternal(ExecutionManager *em) {
  CUDAQ_INFO("external caller setting the execution manager.");
  executionManager = em;
}

void cudaq::resetExecutionManagerInternal() {
  CUDAQ_INFO("external caller clearing the execution manager.");
  executionManager = nullptr;
}

ExecutionManager *cudaq::getExecutionManagerInternal() {
  return executionManager;
}

ExecutionManager *cudaq::detail::getExecutionManagerFromContext() {
  auto ctx = getExecutionContext();
  if (ctx)
    return ctx->executionManager;
  return nullptr;
}

void ExecutionManager::finalizeExecutionContext(ExecutionContext &ctx) {
  policies::withPolicy(ctx.name, [&](auto policy) {
    policies::visitResult(
        [&]() { return cudaq::finalize_execution_manager(*this, policy, ctx); },
        [&](sample_result &&r) { ctx.result = std::move(r); },
        [&](policies::void_result &&r) {});
  });
}

void ExecutionManager::initializeState(const std::vector<QuditInfo> &targets,
                                       const state &state) {
  initializeState(targets, state.internal.get());
}

std::size_t
cudaq::execution_manager::allocateQudit(std::size_t quditLevels) {
  return getExecutionManager()->allocateQudit(quditLevels);
}

void cudaq::execution_manager::returnQudit(const QuditInfo &q) {
  return getExecutionManager()->returnQudit(q);
}

void cudaq::execution_manager::initializeState(
    const std::vector<QuditInfo> &targets, const state &state) {
  getExecutionManager()->initializeState(targets, state);
}

void cudaq::execution_manager::apply(
    std::string_view gateName, const std::vector<double> &params,
    const std::vector<QuditInfo> &controls,
    const std::vector<QuditInfo> &targets, bool isAdjoint,
    const spin_op_term op) {
  getExecutionManager()->apply(gateName, params, controls, targets, isAdjoint,
                               op);
}

void cudaq::execution_manager::applyNoise(
    const kraus_channel &channel, const std::vector<QuditInfo> &targets) {
  getExecutionManager()->applyNoise(channel, targets);
}

void cudaq::execution_manager::reset(const QuditInfo &target) {
  getExecutionManager()->reset(target);
}

int cudaq::execution_manager::measure(const QuditInfo &target,
                                            const std::string &registerName) {
  return getExecutionManager()->measure(target, registerName);
}

cudaq::SpinMeasureResult
cudaq::execution_manager::measure(const cudaq::spin_op &op) {
  return getExecutionManager()->measure(op);
}

void cudaq::execution_manager::startCtrlRegion(
    const std::vector<std::size_t> &controlQubits) {
  getExecutionManager()->startCtrlRegion(controlQubits);
}

void cudaq::execution_manager::endCtrlRegion(std::size_t nControls) {
  getExecutionManager()->endCtrlRegion(nControls);
}

void cudaq::execution_manager::startAdjointRegion() {
  getExecutionManager()->startAdjointRegion();
}

void cudaq::execution_manager::endAdjointRegion() {
  getExecutionManager()->endAdjointRegion();
}
