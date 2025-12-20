/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm_module.h"
#include "llvm_module_impl.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/raw_ostream.h"

namespace cudaq::compiler {

// module_impl implementations
llvm_module_impl::llvm_module_impl(std::unique_ptr<::llvm::Module> mod,
                                   std::unique_ptr<::llvm::LLVMContext> ctx)
    : llvm_context(std::move(ctx)), llvm_module(std::move(mod)) {}

// module implementations
llvm_module::llvm_module(std::unique_ptr<::llvm::Module> llvm_module,
                         std::unique_ptr<::llvm::LLVMContext> llvm_context)
    : pimpl_(std::make_unique<llvm_module_impl>(std::move(llvm_module),
                                                std::move(llvm_context))) {}

llvm_module::~llvm_module() = default;

llvm_module::llvm_module(llvm_module &&other) noexcept = default;

llvm_module &llvm_module::operator=(llvm_module &&other) noexcept = default;

std::string llvm_module::to_string() const {
  if (!pimpl_ || !pimpl_->llvm_module) {
    return "";
  }

  std::string result;
  ::llvm::raw_string_ostream os(result);
  pimpl_->llvm_module->print(os, nullptr);
  return result;
}

std::string llvm_module::encode_to_base64_bitcode() const {
  if (!pimpl_ || !pimpl_->llvm_module) {
    return "";
  }

  // Write bitcode to string buffer
  ::llvm::SmallString<1024> bitCodeMem;
  ::llvm::raw_svector_ostream os(bitCodeMem);
  ::llvm::WriteBitcodeToFile(*pimpl_->llvm_module, os);

  // Encode to base64
  return ::llvm::encodeBase64(bitCodeMem.str());
}

std::string llvm_module::extract_output_names() const {
  if (!pimpl_ || !pimpl_->llvm_module) {
    return "";
  }
 
  return compiler::extract_output_names(*this);
}
} // namespace cudaq::compiler
