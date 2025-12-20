/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "compiler.h"
#include "mlir_module.h"
#include <optional>
#include <string>

namespace cudaq::compiler {

/// \brief Optimization engine for MLIR modules
///
/// Applies MLIR pass pipelines to optimize quantum kernels. Supports both
/// consuming and non-consuming patterns for flexible ownership management.
///
/// \b Pass Pipeline Format:
/// Pass pipelines use MLIR textual format, e.g.:
/// - `"func.func(canonicalize)"` - Canonicalize functions
/// - `"func.func(canonicalize,cse)"` - Canonicalize then CSE
/// - `"builtin.module(func.func(mem2reg,canonicalize))"` - Nested passes
///
/// \b Example:
/// \code
/// cudaq::compiler::mlir_module mod(quake_code);
/// 
/// // Non-consuming: mutate in place
/// auto error = optimizer::run(mod, "func.func(canonicalize)");
/// if (error) {
///   std::cerr << "Optimization failed: " << *error << std::endl;
/// }
/// 
/// // Consuming: returns new module
/// auto result = optimizer::run(std::move(mod), "func.func(cse)");
/// if (result.success()) {
///   auto optimized_mod = result.take_module();
/// }
/// \endcode
class optimizer {
public:
  /// \brief Run optimization passes on an MLIR module (consuming)
  ///
  /// Takes ownership of the module, applies optimization passes, and returns
  /// a new result with the optimized module. Use this when you want to
  /// continue a compilation pipeline with move semantics.
  ///
  /// \note This consuming version internally calls the non-consuming version,
  ///       as optimization is performed in-place on the MLIR module regardless
  ///       of the call signature. The move semantics indicate ownership transfer.
  ///
  /// \param module_input Input MLIR module to optimize (consumed via move)
  /// \param pass_pipeline MLIR pass pipeline string
  /// \return mlir_compilation_result with optimized module or error
  static mlir_compilation_result run(mlir_module &&module_input,
                                     const std::string &pass_pipeline);

  /// \brief Run optimization passes on an MLIR module (non-consuming)
  ///
  /// Mutates the module in place to apply optimization passes. Use this
  /// when you want to optimize a module multiple times or inspect it
  /// between optimization stages.
  ///
  /// \param module_input Input MLIR module to optimize (mutated in place)
  /// \param pass_pipeline MLIR pass pipeline string
  /// \return std::nullopt on success, error message string on failure
  static std::optional<std::string> run(mlir_module &module_input,
                                        const std::string &pass_pipeline);
};

} // namespace cudaq::compiler
