/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "translate.h"
#include "mlir_module_impl.h"
#include "utils.h"

#include "cudaq/utils/string_utils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

namespace cudaq::compiler {

// ==========================================================================
// New API with mlir::module
// ==========================================================================

translate_result translator::run(mlir_module &&module_input,
                                  const std::string &target) {
  // Note: We still clone the module even when consuming it because the
  // translation process modifies the module and we want to preserve the
  // original semantics. The move parameter indicates ownership transfer
  // to this function, but we preserve the module for potential reuse.
  return run(static_cast<const mlir_module &>(module_input), target);
}

translate_result translator::run(const mlir_module &module_input,
                                  const std::string &target) {
  translate_result result;

  if (target.empty()) {
    result.error_message = "Translation target string is empty";
    return result;
  }

  // Initialize translation registry (internally guarded, safe to call multiple times)
  cudaq::compiler::initializeTranslations();

  // Get the MLIR module from the wrapper
  // const_cast is safe here because we're only reading the module for cloning
  auto *impl = const_cast<mlir_module_impl*>(module_input.get_impl());
  if (!impl) {
    result.error_message = "Invalid MLIR module for translation";
    return result;
  }

  auto mlir_module = cudaq::compiler::get_mlir_module(impl);
  if (!mlir_module) {
    result.error_message = "Empty MLIR module";
    return result;
  }

  // Parse target to extract base name (e.g., "qir-base:profile" -> "qir-base")
  auto splitTarget = cudaq::split(target, ':');
  auto target_key = target;
  if (splitTarget.size() > 1 &&
      splitTarget[0].find("qir") != std::string::npos)
    target_key = splitTarget[0];

  // Look up the translation function
  auto &registry = cudaq::compiler::getTranslationRegistry();
  auto it = registry.find(target_key);
  if (it == registry.end()) {
    result.error_message = "Unknown translation target: " + target;
    return result;
  }

  // Clone the MLIR module so we don't modify the original
  // Note: We use string serialization + parsing for cloning because it ensures
  // the cloned module has its own context, avoiding lifetime issues with
  // non-owning modules where the original context may be destroyed.
  std::string mlirStr;
  ::llvm::raw_string_ostream os(mlirStr);
  mlir_module->print(os);
  os.flush();

  auto clonedModule = ::mlir::parseSourceString<::mlir::ModuleOp>(
      ::llvm::StringRef(mlirStr), mlir_module->getContext());

  if (!clonedModule) {
    result.error_message = "Failed to clone MLIR module for translation";
    return result;
  }

  // Run the translation
  auto [llvmContext, llvmModule] =
      it->second(*clonedModule, *mlir_module->getContext());
  if (!llvmModule) {
    result.error_message = "Translation to " + target + " failed";
    return result;
  }

  // Create LLVM module wrapper
  llvm_module llvm_mod(std::move(llvmModule), std::move(llvmContext));
  result.result = std::move(llvm_mod);
  return result;
}

} // namespace cudaq::compiler
