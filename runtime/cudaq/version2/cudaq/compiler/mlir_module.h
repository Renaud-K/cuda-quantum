/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file mlir_module.h
/// \brief MLIR module wrapper with pimpl pattern for header isolation

#pragma once

#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
}

namespace cudaq::compiler {

/// \brief Forward-declared implementation struct (defined in mlir_module_impl.h)
struct mlir_module_impl;

/// \brief Opaque wrapper for MLIR modules using Pimpl idiom
///
/// Hides MLIR types (ModuleOp, MLIRContext) from the public interface to
/// avoid forcing MLIR header includes in user code. This type is separate
/// from llvm_module to enforce clear ownership semantics and type safety.
///
/// \b Ownership Semantics:
/// - **Owning**: Created from string (parses MLIR text), owns ModuleOp and MLIRContext
///   - Supports move operations
///   - Deallocates resources on destruction
/// - **Non-owning**: Created from ModuleOp reference, does NOT own data
///   - **Move constructors/assignment are DELETED for non-owning modules**
///   - Safe to use as temporary wrapper for existing ModuleOp
///
/// \b Example Usage:
/// \code
/// // Owning module - parsed from Quake IR string
/// cudaq::compiler::mlir_module mod(quake_code);
/// 
/// // Serialize to string
/// std::string mlir_str = mod.to_string();
/// 
/// // Get hash for caching
/// std::size_t h = mod.hash();
/// 
/// // Non-owning wrapper (from existing ModuleOp)
/// mlir::ModuleOp existing_op = /* ... */;
/// cudaq::compiler::mlir_module wrapper(existing_op);  // References, doesn't own
/// \endcode
class mlir_module {
public:
  /// \brief Default constructor (empty MLIR module, owning)
  mlir_module();

  ~mlir_module();

  /// \brief Move constructor (only for owning modules)
  /// \throws std::logic_error if attempting to move a non-owning module
  /// \note Non-owning modules cannot be moved to prevent dangling references
  mlir_module(mlir_module &&other);

  /// \brief Move assignment (only for owning modules)
  /// \throws std::logic_error if attempting to move a non-owning module
  /// \note Non-owning modules cannot be moved to prevent dangling references
  mlir_module &operator=(mlir_module &&other);

  // Delete copy operations
  mlir_module(const mlir_module &) = delete;
  mlir_module &operator=(const mlir_module &) = delete;

  /// \brief Parse MLIR code from string (owning)
  /// \param code MLIR textual representation to parse
  /// \param keep_thunk Whether to keep the thunk operation
  explicit mlir_module(const std::string &code);

  /// \brief Construct module from existing MLIR ModuleOp (non-owning)
  /// \param mod MLIR ModuleOp to wrap without taking ownership
  /// \note Move operations are disabled for non-owning modules
  explicit mlir_module(::mlir::ModuleOp mod);

  /// \brief Serialize module to string
  /// \return Textual representation of MLIR module
  std::string to_string() const;

  /// \brief Get hash of the module for caching
  /// \return Hash of the module
  std::size_t hash() const;

  /// \brief Internal access for implementation (non-const)
  /// \return Raw pointer to impl (for use in compiler implementation)
  mlir_module_impl *get_impl() { return pimpl_.get(); }

  /// \brief Internal access for implementation (const)
  /// \return Raw const pointer to impl (for use in compiler implementation)
  const mlir_module_impl *get_impl() const { return pimpl_.get(); }

  /// \brief Check if this module owns its data
  /// \return true if owning, false if non-owning
  bool owns_data() const;

  /// \brief Remove the thunk operation for the given kernel name
  /// \param kernel_name Name of the kernel
  /// \note This is a non-consuming operation
  void remove_thunk(const std::string& kernel_name);

  /// \brief Disable CUDA-Q passes registration
  /// \details Static method to disable CUDA-Q passes registration
  static void disableCudaqPassRegistration();

private:
  std::unique_ptr<mlir_module_impl> pimpl_; ///< Pimpl pointer to hide MLIR types
};

} // namespace cudaq::compiler::mlir

