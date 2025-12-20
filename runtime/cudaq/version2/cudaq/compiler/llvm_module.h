/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file llvm_module.h
/// \brief LLVM module wrapper with pimpl pattern for header isolation

#pragma once

#include <memory>
#include <string>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace cudaq::compiler {

/// \brief Forward-declared implementation struct (defined in llvm_module_impl.h)
struct llvm_module_impl;

/// \brief Opaque wrapper for LLVM modules using Pimpl idiom
///
/// Hides LLVM types (Module, LLVMContext) from the public interface to
/// avoid forcing LLVM header includes in user code. This type is separate
/// from mlir_module to enforce clear type safety and ownership semantics.
///
/// \b Ownership Semantics:
/// - **Always owning**: Takes ownership of LLVM Module and LLVMContext
/// - **Move-only**: Copy operations are deleted, only move allowed
/// - Deallocates resources (Module, Context) on destruction
///
/// Typically created as the result of translation from MLIR to LLVM IR
/// via the compiler::translate() method.
///
/// \b Example Usage:
/// \code
/// // Created from translation
/// cudaq::compiler::mlir_module mlir_mod(quake_code);
/// auto trans_result = compiler.translate(std::move(mlir_mod), "qir");
/// cudaq::compiler::llvm_module llvm_mod = trans_result.take_llvm_module();
/// 
/// // Serialize to LLVM IR text
/// std::string llvm_ir = llvm_mod.to_string();
/// 
/// // Encode to base64 bitcode for remote backends
/// std::string base64 = llvm_mod.encode_to_base64_bitcode();
/// 
/// // Move into execution engine
/// cudaq::compiler::execution_engine engine(std::move(llvm_mod));
/// \endcode
class llvm_module {
public:
  /// \brief Construct LLVM module from LLVM Module and Context (takes ownership)
  /// \param llvm_module LLVM Module to take ownership of
  /// \param llvm_context LLVM Context to take ownership of
  llvm_module(std::unique_ptr<::llvm::Module> llvm_module,
         std::unique_ptr<::llvm::LLVMContext> llvm_context);

  ~llvm_module();

  /// \brief Move constructor
  llvm_module(llvm_module &&other) noexcept;

  /// \brief Move assignment
  llvm_module &operator=(llvm_module &&other) noexcept;

  // Delete copy operations (LLVM modules are move-only)
  llvm_module(const llvm_module &) = delete;
  llvm_module &operator=(const llvm_module &) = delete;

  /// \brief Serialize module to LLVM IR string
  /// \return Textual representation of LLVM module
  std::string to_string() const;

  /// \brief Encode module to base64-encoded bitcode
  /// \return Base64-encoded LLVM bitcode
  std::string encode_to_base64_bitcode() const;

  /// \brief Extract output names from QIR LLVM module
  /// \return JSON object with output name mappings
  std::string extract_output_names() const;

  /// \brief Internal access for implementation
  /// \return Raw pointer to impl (for use in compiler implementation)
  llvm_module_impl *get_impl() { return pimpl_.get(); }

  /// \brief Internal access for implementation (const)
  /// \return Raw const pointer to impl (for use in compiler implementation)
  const llvm_module_impl *get_impl() const { return pimpl_.get(); }

private:
  std::unique_ptr<llvm_module_impl> pimpl_; ///< Pimpl pointer to hide LLVM types
};

} // namespace cudaq::compiler::llvm

