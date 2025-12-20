/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mlir_module.h"
#include "mlir_module_impl.h"

#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include <mutex>

namespace cudaq::compiler {

static bool m_disableCudaqPassRegistration = false;

void mlir_module::disableCudaqPassRegistration() {
  m_disableCudaqPassRegistration = true;
}

// Helper to get MLIR ModuleOp from impl
::mlir::ModuleOp get_mlir_module(mlir_module_impl *impl) {
  if (!impl->owns_data) {
    return std::get<::mlir::ModuleOp>(impl->content);
  } else {
    auto &owning_pair =
        std::get<std::pair<::mlir::OwningOpRef<::mlir::ModuleOp>,
                           std::unique_ptr<::mlir::MLIRContext>>>(
            impl->content);
    return *owning_pair.first;
  }
}

// module_impl implementations
mlir_module_impl::mlir_module_impl() {
  // Global one-time initialization (thread-safe)
  static std::once_flag once;
  std::call_once(once, [&]() {
    if (!m_disableCudaqPassRegistration) {
      cudaq::registerAllPasses();
    }
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });

  // Create and configure context
  auto context = std::make_unique<::mlir::MLIRContext>();

  // Register dialects for this context
  ::mlir::DialectRegistry registry;
  cudaq::registerAllDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  // Register LLVM dialect translation for this context
  ::mlir::registerLLVMDialectTranslation(*context);

  ::mlir::OwningOpRef<::mlir::ModuleOp> mlir_module;
  content = std::make_pair(std::move(mlir_module), std::move(context));
  owns_data = true;
}

mlir_module_impl::mlir_module_impl(::mlir::ModuleOp mod)
    : content(mod), owns_data(false) {}

mlir_module_impl::mlir_module_impl(::mlir::OwningOpRef<::mlir::ModuleOp> mod,
                                   std::unique_ptr<::mlir::MLIRContext> ctx)
    : content(std::make_pair(std::move(mod), std::move(ctx))), owns_data(true) {
}

std::size_t mlir_module_impl::hash() const {
  auto hash = llvm::hash_code{0};
  if (std::holds_alternative<::mlir::ModuleOp>(content)) {
    std::get<::mlir::ModuleOp>(content)->walk([&hash](::mlir::Operation *op) {
      hash = llvm::hash_combine(hash,
                                ::mlir::OperationEquivalence::computeHash(op));
    });
  } else if (std::holds_alternative<
                 std::pair<::mlir::OwningOpRef<::mlir::ModuleOp>,
                           std::unique_ptr<::mlir::MLIRContext>>>(content)) {
    auto &mod =
        std::get<std::pair<::mlir::OwningOpRef<::mlir::ModuleOp>,
                           std::unique_ptr<::mlir::MLIRContext>>>(content)
            .first;
    (*mod)->walk([&hash](::mlir::Operation *op) {
      hash = llvm::hash_combine(hash,
                                ::mlir::OperationEquivalence::computeHash(op));
    });
  }
  return static_cast<size_t>(hash);
}

// module implementations
mlir_module::mlir_module() : pimpl_(std::make_unique<mlir_module_impl>()) {}

mlir_module::~mlir_module() = default;

mlir_module::mlir_module(mlir_module &&other) {
  // Only owning modules can be moved
  if (other.pimpl_ && !other.pimpl_->owns_data) {
    throw std::logic_error(
        "Cannot move a non-owning mlir_module. Non-owning modules reference "
        "external data and must not be moved to prevent dangling references.");
  }
  pimpl_ = std::move(other.pimpl_);
}

mlir_module &mlir_module::operator=(mlir_module &&other) {
  // Only owning modules can be moved
  if (other.pimpl_ && !other.pimpl_->owns_data) {
    throw std::logic_error(
        "Cannot move a non-owning mlir_module. Non-owning modules reference "
        "external data and must not be moved to prevent dangling references.");
  }
  if (this != &other) {
    pimpl_ = std::move(other.pimpl_);
  }
  return *this;
}

mlir_module::mlir_module(const std::string &code)
    : pimpl_(std::make_unique<mlir_module_impl>()) {
  // Parse MLIR code
  auto &owning_pair = std::get<std::pair<::mlir::OwningOpRef<::mlir::ModuleOp>,
                                         std::unique_ptr<::mlir::MLIRContext>>>(
      pimpl_->content);
  auto &context = owning_pair.second;
  auto &moduleOp = owning_pair.first;

  moduleOp = ::mlir::parseSourceString<::mlir::ModuleOp>(code, context.get());

  if (!moduleOp) {
    throw std::runtime_error("Failed to parse MLIR code");
  }
  pimpl_->owns_data = true;
}

void mlir_module::remove_thunk(const std::string &kernel_name) {
  if (!pimpl_) {
    throw std::runtime_error(
        "Cannot remove thunk: mlir_module has null implementation");
  }

  auto mlir_module = get_mlir_module(pimpl_.get());
  auto *context = mlir_module.getContext();

  // Remove the FuncOp with the name "<kernel_name>.thunk"
  std::string thunkName;
  mlir_module->walk([&](mlir::func::FuncOp op) {
    if (op.getName().ends_with(".thunk")) {
      thunkName = op.getName().str();
      op->erase();
    }
  });

  // No thunk, just return
  if (thunkName.empty())
    return;

  mlir::PassManager pm(context);
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (::mlir::failed(pm.run(mlir_module)))
    throw std::runtime_error("Failed to remove thunk operation for kernel. (" +
                             thunkName + ".)");
}

mlir_module::mlir_module(::mlir::ModuleOp mod)
    : pimpl_(std::make_unique<mlir_module_impl>(mod)) {}

std::string mlir_module::to_string() const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Cannot convert to string: mlir_module has null implementation");
  }

  std::string result;
  llvm::raw_string_ostream os(result);

  auto mlir_module = get_mlir_module(pimpl_.get());
  if (!mlir_module) {
    throw std::runtime_error(
        "Cannot convert to string: mlir_module contains null ModuleOp");
  }

  mlir_module->print(os);
  os.flush();

  return result;
}

std::size_t mlir_module::hash() const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Cannot compute hash: mlir_module has null implementation");
  }
  return pimpl_->hash();
}

bool mlir_module::owns_data() const {
  if (!pimpl_) {
    return false;
  }
  return pimpl_->owns_data;
}

} // namespace cudaq::compiler
