/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file llvm_module_impl.h
/// \brief Internal implementation details for llvm::module (private header)

#pragma once

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>

namespace cudaq::compiler {

// Forward declare the module class
class module;

// Implementation struct for LLVM modules (always owning)
struct llvm_module_impl {
  std::unique_ptr<::llvm::LLVMContext> llvm_context;
  std::unique_ptr<::llvm::Module> llvm_module;

  // Constructor
  llvm_module_impl(std::unique_ptr<::llvm::Module> mod,
              std::unique_ptr<::llvm::LLVMContext> ctx);
};

} // namespace cudaq::compiler::llvm

