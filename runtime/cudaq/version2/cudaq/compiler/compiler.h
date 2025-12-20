/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file compiler.h
/// \brief JIT compiler API for MLIR/QIR compilation
///
/// Provides a clean, PIMPL-based interface for compiling quantum kernels.
/// The compiler operates on separate mlir_module and llvm_module types
/// to enforce type safety and clear ownership semantics.
///
/// \b Basic Usage:
/// \code
/// #include "cudaq/compiler/compiler.h"
/// 
/// // Parse Quake IR to MLIR
/// cudaq::compiler::mlir_module mod(quake_code);
/// 
/// // Create compiler instance
/// cudaq::compiler::compiler comp;
/// 
/// // Optimize (non-consuming - mutates in place)
/// auto opt_error = comp.optimize(mod, "func.func(canonicalize)");
/// if (opt_error) {
///   std::cerr << "Optimization failed: " << *opt_error << std::endl;
/// }
/// 
/// // Translate to QIR (consuming - takes ownership)
/// auto trans_result = comp.translate(std::move(mod), "qir-base");
/// if (!trans_result.success()) {
///   std::cerr << "Translation failed: " << trans_result.error_message.value() << std::endl;
/// }
/// 
/// // Extract LLVM module
/// if (trans_result.is_llvm_module()) {
///   auto llvm_mod = trans_result.take_llvm_module();
///   
///   // Encode for remote execution
///   std::string bitcode = llvm_mod.encode_to_base64_bitcode();
///   
///   // Or execute locally
///   cudaq::compiler::execution_engine engine(std::move(llvm_mod));
///   auto *fn_ptr = engine.get_symbol("kernel_name");
/// }
/// \endcode

#pragma once

#include "llvm_module.h"
#include "mlir_module.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cudaq::compiler {
class execution_engine;

// ============================================================================
// Result Types
// ============================================================================

/// \brief Result of an MLIR compilation operation
///
/// Returned by consuming compiler operations (those taking mlir_module&&).
/// Contains either a successfully compiled MLIR module or an error message.
/// Use success() to check status before accessing the module.
///
/// \b Error Semantics:
/// - error_message.has_value() == true: Operation failed, error message available
/// - error_message.has_value() == false: Operation succeeded, module is valid
/// - Always check success() or is_error() before accessing module
///
/// \b Example:
/// \code
/// auto result = compiler.optimize(std::move(mod), "func.func(cse)");
/// if (result.success()) {
///   auto optimized = result.take_module();
///   // Continue compilation pipeline...
/// } else {
///   std::cerr << "Error: " << result.error_message.value() << std::endl;
/// }
/// \endcode
struct mlir_compilation_result {
  mlir_module module_result;
  std::optional<std::string> error_message;

  /// \brief Check if compilation succeeded
  /// \return true if no error, false if error occurred
  bool success() const { return !error_message.has_value(); }

  /// \brief Check if compilation failed (inverse of success())
  /// \return true if error occurred, false if no error
  bool is_error() const { return error_message.has_value(); }

  /// \brief Get the compiled module (only valid if success())
  /// \pre success() must be true
  mlir_module &get_module() { return module_result; }

  /// \brief Get the compiled module (only valid if success())
  /// \pre success() must be true
  const mlir_module &get_module() const { return module_result; }

  /// \brief Move the module out of the result
  /// \pre success() must be true
  mlir_module take_module() { return std::move(module_result); }
};

/// \brief Result of a translation operation
///
/// Returned by compiler::translate() methods. Contains a variant of either
/// an llvm_module (for QIR targets) or a std::string (for text-based targets
/// like OpenQASM). Use is_llvm_module() and is_string() to determine the
/// result type before extraction.
///
/// \b Error Semantics:
/// - error_message.has_value() == true: Operation failed, error message available
/// - error_message.has_value() == false: Operation succeeded, result is valid
/// - Always check success() or is_error() before accessing result
/// - Always check is_llvm_module() or is_string() before extracting value
///
/// \b Example:
/// \code
/// auto result = compiler.translate(mod, "qir-base");
/// if (result.success()) {
///   if (result.is_llvm_module()) {
///     auto llvm_mod = result.take_llvm_module();
///     // Use LLVM module...
///   } else if (result.is_string()) {
///     std::string text = result.get_string();
///     // Use text output (e.g., OpenQASM)...
///   }
/// } else {
///   std::cerr << "Translation failed: " << result.error_message.value() << std::endl;
/// }
/// \endcode
struct translate_result {
  std::optional<std::variant<llvm_module, std::string>> result;
  std::optional<std::string> error_message;

  // Explicit move operations (required due to move-only llvm_module in variant)
  translate_result() = default;
  translate_result(translate_result &&) = default;
  translate_result &operator=(translate_result &&) = default;
  
  // Delete copy operations (due to move-only llvm_module in variant)
  translate_result(const translate_result &) = delete;
  translate_result &operator=(const translate_result &) = delete;

  /// \brief Check if translation succeeded
  /// \return true if no error, false if error occurred
  bool success() const { return !error_message.has_value(); }

  /// \brief Check if translation failed (inverse of success())
  /// \return true if error occurred, false if no error
  bool is_error() const { return error_message.has_value(); }

  /// \brief Check if result is an LLVM module
  /// \return true if result contains llvm_module
  bool is_llvm_module() const {
    return result.has_value() && std::holds_alternative<llvm_module>(*result);
  }

  /// \brief Check if result is a string
  /// \return true if result contains std::string
  bool is_string() const {
    return result.has_value() && std::holds_alternative<std::string>(*result);
  }

  /// \brief Get LLVM module reference (only valid if is_llvm_module())
  /// \pre success() && is_llvm_module() must both be true
  /// \throws std::runtime_error if result is empty or contains wrong type
  /// \return Reference to the LLVM module in the result
  llvm_module &get_llvm_module() {
    if (!result.has_value()) {
      throw std::runtime_error(
          "Cannot get_llvm_module(): result is empty. Check success() first.");
    }
    if (!std::holds_alternative<llvm_module>(*result)) {
      throw std::runtime_error(
          "Cannot get_llvm_module(): result does not contain llvm_module. "
          "Check is_llvm_module() first.");
    }
    return std::get<llvm_module>(*result);
  }

  /// \brief Get LLVM module by moving (only valid if is_llvm_module())
  /// \pre success() && is_llvm_module() must both be true
  /// \throws std::runtime_error if result is empty or contains wrong type
  /// \return LLVM module moved out of result
  llvm_module take_llvm_module() {
    if (!result.has_value()) {
      throw std::runtime_error(
          "Cannot take_llvm_module(): result is empty. Check success() first.");
    }
    if (!std::holds_alternative<llvm_module>(*result)) {
      throw std::runtime_error(
          "Cannot take_llvm_module(): result does not contain llvm_module. "
          "Check is_llvm_module() first.");
    }
    return std::move(std::get<llvm_module>(*result));
  }

  /// \brief Get string result (only valid if is_string())
  /// \pre success() && is_string() must both be true
  /// \throws std::runtime_error if result is empty or contains wrong type
  /// \return Copy of the string in the result
  std::string get_string() const {
    if (!result.has_value()) {
      throw std::runtime_error(
          "Cannot get_string(): result is empty. Check success() first.");
    }
    if (!std::holds_alternative<std::string>(*result)) {
      throw std::runtime_error(
          "Cannot get_string(): result does not contain string. "
          "Check is_string() first.");
    }
    return std::get<std::string>(*result);
  }

  /// \brief Get string reference (only valid if is_string())
  /// \pre success() && is_string() must both be true
  /// \throws std::runtime_error if result is empty or contains wrong type
  /// \return Reference to the string in the result
  std::string &get_string_ref() {
    if (!result.has_value()) {
      throw std::runtime_error(
          "Cannot get_string_ref(): result is empty. Check success() first.");
    }
    if (!std::holds_alternative<std::string>(*result)) {
      throw std::runtime_error(
          "Cannot get_string_ref(): result does not contain string. "
          "Check is_string() first.");
    }
    return std::get<std::string>(*result);
  }
};

/// \brief Variant for kernel compilation results in QPU context
///
/// Returned by qpu::compile_kernel_impl(). Can be either:
/// - std::function<void()>: Ready-to-execute kernel (when return_functor=true)
/// - translate_result: Compiled LLVM/string for remote execution (when return_functor=false)
///
/// The behavior is controlled by jit_options::return_functor flag in the QPU.
///
/// \b Example (in QPU implementation):
/// \code
/// auto result = compile_kernel_impl(kernel_name, module, args);
/// if (std::holds_alternative<std::function<void()>>(result)) {
///   // Execute immediately
///   auto kernel_fn = std::get<std::function<void()>>(result);
///   kernel_fn();
/// } else {
///   // Extract translate_result for remote execution
///   auto trans_result = std::get<translate_result>(result);
///   auto llvm_mod = trans_result.take_llvm_module();
///   // Send to remote backend...
/// }
/// \endcode
using kernel_compilation_result =
    std::variant<std::function<void()>, translate_result>;

// ============================================================================
// Compiler Class
// ============================================================================

/// \brief JIT compiler for quantum kernels
///
/// Provides optimization, translation, and combined compilation pipelines.
/// Uses PIMPL idiom to hide implementation details.
///
/// \b API Pattern:
/// - Consuming (&&) operations return a result with a new module
/// - Non-consuming (&) operations mutate in place and return optional error
class compiler {
  struct impl;
  std::unique_ptr<impl> impl_;

public:
  compiler();
  ~compiler();

  // ==========================================================================
  // Argument Synthesis
  // ==========================================================================

  /// \brief Synthesize arguments for a kernel (consuming version)
  ///
  /// Takes ownership of the module, synthesizes arguments, and returns a new
  /// result with the updated module.
  ///
  /// \param kernelName Name of the kernel
  /// \param module_input Input MLIR module (consumed)
  /// \param rawArgs Raw arguments to synthesize
  /// \return mlir_compilation_result with synthesized module or error
  mlir_compilation_result synthesize_arguments(
      const std::string &kernelName, mlir_module &&module_input,
      const std::vector<void *> &rawArgs);

  /// \brief Synthesize arguments for a kernel (non-consuming version)
  ///
  /// Mutates the module in place to synthesize arguments.
  ///
  /// \param kernelName Name of the kernel
  /// \param module_input Input MLIR module (mutated in place)
  /// \param rawArgs Raw arguments to synthesize
  /// \return std::nullopt on success, error message on failure
  std::optional<std::string>
  synthesize_arguments(const std::string &kernelName,
                       mlir_module &module_input,
                       const std::vector<void *> &rawArgs);

  // ==========================================================================
  // Optimization
  // ==========================================================================

  /// \brief Run optimization passes on an MLIR module (consuming version)
  ///
  /// Takes ownership of the module, applies optimization passes, and returns
  /// a new result with the optimized module.
  ///
  /// \param module_input Input MLIR module (consumed)
  /// \param pass_pipeline MLIR pass pipeline string
  /// \return mlir_compilation_result with optimized module or error
  mlir_compilation_result optimize(mlir_module &&module_input,
                                   const std::string &pass_pipeline);

  /// \brief Run optimization passes on an MLIR module (non-consuming version)
  ///
  /// Mutates the module in place to apply optimization passes.
  ///
  /// \param module_input Input MLIR module (mutated in place)
  /// \param pass_pipeline MLIR pass pipeline string
  /// \return std::nullopt on success, error message on failure
  std::optional<std::string> optimize(mlir_module &module_input,
                                      const std::string &pass_pipeline);

  // ==========================================================================
  // Translation
  // ==========================================================================

  /// \brief Translate MLIR module to target format (consuming version)
  ///
  /// Takes ownership of the MLIR module and translates it to the target
  /// format (LLVM module for QIR, string for OpenQASM, etc.).
  ///
  /// \param module_input Input MLIR module (consumed)
  /// \param target Translation target (e.g., "qir", "qir-base", "openqasm2")
  /// \return translate_result with LLVM module or string
  translate_result translate(mlir_module &&module_input,
                             const std::string &target);

  /// \brief Translate MLIR module to target format (preserving version)
  ///
  /// Clones the MLIR module internally and translates it to the target format.
  /// The original module remains unchanged.
  ///
  /// \param module_input Input MLIR module (preserved)
  /// \param target Translation target (e.g., "qir", "qir-base", "openqasm2")
  /// \return translate_result with LLVM module or string
  translate_result translate(const mlir_module &module_input,
                             const std::string &target);

  // ==========================================================================
  // Combined Pipeline
  // ==========================================================================

  /// \brief Combined compilation pipeline (parse, optimize, translate)
  ///
  /// Convenience method that performs the complete compilation pipeline:
  /// 1. Parse quake_code to MLIR
  /// 2. Run optimization passes
  /// 3. Translate to target format
  ///
  /// \param quake_code Input Quake code as string
  /// \param pass_pipeline MLIR pass pipeline string
  /// \param target Translation target (e.g., "qir-base")
  /// \return translate_result with final output
  translate_result compile(const std::string &quake_code,
                           const std::string &pass_pipeline,
                           const std::string &target);

  // ==========================================================================
  // Kernel Validation (NEW)
  // ==========================================================================

  /// \brief Validate a kernel against policy constraints
  ///
  /// Analyzes the MLIR module to detect kernel characteristics, then checks
  /// them against the policy's constraints. This allows execution policies
  /// to declare compatibility requirements and catch violations early.
  ///
  /// \b Example:
  /// \code
  /// // Get policy constraints
  /// auto constraints = cudaq::sample_policy::get_constraints();
  /// 
  /// // Validate kernel
  /// auto validation = compiler.validate_kernel(mlir_mod, constraints, "my_kernel");
  /// if (!validation.success()) {
  ///   std::cerr << validation.get_detailed_error() << std::endl;
  ///   return;
  /// }
  /// \endcode
  ///
  /// \param module The MLIR module containing the kernel
  /// \param constraints Policy constraints to validate against
  /// \param kernel_name Optional specific kernel to validate (validates all if empty)
  /// \return validation_result indicating success or failure with detailed error
  ///
  /// \note This is a forward declaration. Include kernel_validator.h for full definition.
  struct validation_result;
  struct policy_constraints;
  validation_result validate_kernel(const mlir_module &module,
                                   const policy_constraints &constraints,
                                   const std::string &kernel_name = "");

  /// \brief Analyze a kernel's characteristics without validation
  ///
  /// Useful for introspection or custom validation logic. Returns detailed
  /// information about what operations and patterns are present in the kernel.
  ///
  /// \b Example:
  /// \code
  /// auto traits = compiler.analyze_kernel(mlir_mod, "my_kernel");
  /// if (traits.has(kernel_characteristic::has_measurements)) {
  ///   std::cout << "Kernel has " << traits.num_measurements << " measurements\n";
  /// }
  /// \endcode
  ///
  /// \param module The MLIR module containing the kernel
  /// \param kernel_name Optional specific kernel to analyze (analyzes all if empty)
  /// \return kernel_traits describing the kernel's properties
  ///
  /// \note This is a forward declaration. Include kernel_validator.h for full definition.
  struct kernel_traits;
  kernel_traits analyze_kernel(const mlir_module &module,
                               const std::string &kernel_name = "");
};

} // namespace cudaq::compiler
