/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "opt.h"
#include "mlir_module_impl.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace cudaq::compiler {

// ==========================================================================
// New API with mlir::module
// ==========================================================================

mlir_compilation_result optimizer::run(mlir_module &&module_input,
                                       const std::string &pass_pipeline) {
  // Call non-consuming version
  auto err = run(module_input, pass_pipeline);
  mlir_compilation_result result;
  result.module_result = std::move(module_input);
  result.error_message = err;
  return result;
}

std::optional<std::string> optimizer::run(mlir_module &module_input,
                                          const std::string &pass_pipeline) {
  // Get the MLIR module from the wrapper
  auto *impl = module_input.get_impl();
  if (!impl) {
    return "Invalid MLIR module for optimization";
  }

  auto mlir_module = cudaq::compiler::get_mlir_module(impl);
  if (!mlir_module) {
    return "Empty MLIR module";
  }

  // Check if pass pipeline is empty (identity transform)
  if (pass_pipeline.empty()) {
    return std::nullopt;
  }

  // Create a pass manager
  ::mlir::PassManager pm(mlir_module->getContext());

  // Parse the pass pipeline
  if (::mlir::failed(::mlir::parsePassPipeline(pass_pipeline, pm))) {
    return "Failed to parse pass pipeline: " + pass_pipeline;
  }

  // Run the pass manager
  if (::mlir::failed(pm.run(mlir_module))) {
    return "Pass manager execution failed";
  }

  return std::nullopt;
}

} // namespace cudaq::compiler
