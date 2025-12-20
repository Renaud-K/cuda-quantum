/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "compiler.h"
#include "arg_conversion.h"
#include "mlir_module_impl.h"
#include "opt.h"
#include "translate.h"

#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::compiler {

struct compiler::impl {
  // New API with mlir::module
  mlir_compilation_result
  synthesize_arguments(const std::string &kernelName,
                       mlir_module &&module_input,
                       const std::vector<void *> &rawArgs);

  std::optional<std::string>
  synthesize_arguments(const std::string &kernelName,
                       mlir_module &module_input,
                       const std::vector<void *> &rawArgs);

  mlir_compilation_result optimize(mlir_module &&module_input,
                                   const std::string &pass_pipeline);

  std::optional<std::string> optimize(mlir_module &module_input,
                                      const std::string &pass_pipeline);

  translate_result translate(mlir_module &&module_input,
                             const std::string &target);

  translate_result translate(const mlir_module &module_input,
                             const std::string &target);

  translate_result compile(const std::string &quake_code,
                           const std::string &pass_pipeline,
                           const std::string &target);
};

compiler::compiler() : impl_(std::make_unique<impl>()) {}
compiler::~compiler() = default;

// ==========================================================================
// New API: Argument Synthesis
// ==========================================================================

mlir_compilation_result
compiler::synthesize_arguments(const std::string &kernelName,
                               mlir_module &&module_input,
                               const std::vector<void *> &rawArgs) {
  return impl_->synthesize_arguments(kernelName, std::move(module_input),
                                     rawArgs);
}

std::optional<std::string>
compiler::synthesize_arguments(const std::string &kernelName,
                               mlir_module &module_input,
                               const std::vector<void *> &rawArgs) {
  return impl_->synthesize_arguments(kernelName, module_input, rawArgs);
}

// ==========================================================================
// New API: Optimization
// ==========================================================================

mlir_compilation_result compiler::optimize(mlir_module &&module_input,
                                           const std::string &pass_pipeline) {
  return impl_->optimize(std::move(module_input), pass_pipeline);
}

std::optional<std::string> compiler::optimize(mlir_module &module_input,
                                              const std::string &pass_pipeline) {
  return impl_->optimize(module_input, pass_pipeline);
}

// ==========================================================================
// New API: Translation
// ==========================================================================

translate_result compiler::translate(mlir_module &&module_input,
                                     const std::string &target) {
  return impl_->translate(std::move(module_input), target);
}

translate_result compiler::translate(const mlir_module &module_input,
                                     const std::string &target) {
  return impl_->translate(module_input, target);
}

// ==========================================================================
// New API: Combined Pipeline
// ==========================================================================

translate_result compiler::compile(const std::string &quake_code,
                                   const std::string &pass_pipeline,
                                   const std::string &target) {
  return impl_->compile(quake_code, pass_pipeline, target);
}

// ==========================================================================
// Implementation: New API
// ==========================================================================

mlir_compilation_result
compiler::impl::synthesize_arguments(const std::string &kernelName,
                                     mlir_module &&module_input,
                                     const std::vector<void *> &rawArgs) {
  // Call non-consuming version to perform synthesis
  auto err = synthesize_arguments(kernelName, module_input, rawArgs);
  
  mlir_compilation_result result;
  // Always move the module into result, even if error occurred
  // The module may be partially modified but is still valid
  result.module_result = std::move(module_input);
  result.error_message = err;
  return result;
}

std::optional<std::string>
compiler::impl::synthesize_arguments(const std::string &kernelName,
                                     mlir_module &module_input,
                                     const std::vector<void *> &rawArgs) {
  // Get the MLIR module from the wrapper
  auto *impl = module_input.get_impl();
  if (!impl) {
    return "Invalid MLIR module for synthesis: null implementation";
  }

  auto mlir_module = cudaq::compiler::get_mlir_module(impl);
  if (!mlir_module) {
    return "Invalid MLIR module for synthesis: null ModuleOp";
  }

  auto *context = mlir_module.getContext();
  if (!context) {
    return "Invalid MLIR module for synthesis: null MLIRContext";
  }

  cudaq::opt::ArgumentConverter argCon(kernelName, mlir_module);
  argCon.gen(rawArgs);
  ::mlir::PassManager pm(context);

  // Store kernel and substitution strings on the stack
  ::mlir::SmallVector<std::string> kernels;
  ::mlir::SmallVector<std::string> substs;
  for (auto *kInfo : argCon.getKernelSubstitutions()) {
    std::string kernName =
        cudaq::runtime::cudaqGenPrefixName + kInfo->getKernelName().str();
    kernels.emplace_back(kernName);
    std::string substBuff;
    ::llvm::raw_string_ostream ss(substBuff);
    ss << kInfo->getSubstitutionModule();
    substs.emplace_back(substBuff);
  }

  // Collect references for the argument synthesis
  ::mlir::SmallVector<::mlir::StringRef> kernelRefs{kernels.begin(),
                                                     kernels.end()};
  ::mlir::SmallVector<::mlir::StringRef> substRefs{substs.begin(),
                                                    substs.end()};
  pm.addPass(cudaq::opt::createArgumentSynthesisPass(kernelRefs, substRefs));
  pm.addPass(::mlir::createSymbolDCEPass());
  pm.addPass(::mlir::createCanonicalizerPass());

  if (::mlir::failed(pm.run(mlir_module))) {
    return "Pass manager execution failed";
  }

  return std::nullopt;
}

mlir_compilation_result compiler::impl::optimize(mlir_module &&module_input,
                                                 const std::string &pass_pipeline) {
  return optimizer::run(std::move(module_input), pass_pipeline);
}

std::optional<std::string>
compiler::impl::optimize(mlir_module &module_input,
                         const std::string &pass_pipeline) {
  return optimizer::run(module_input, pass_pipeline);
}

translate_result compiler::impl::translate(mlir_module &&module_input,
                                           const std::string &target) {
  return translator::run(std::move(module_input), target);
}

translate_result compiler::impl::translate(const mlir_module &module_input,
                                           const std::string &target) {
  return translator::run(module_input, target);
}

translate_result compiler::impl::compile(const std::string &quake_code,
                                         const std::string &pass_pipeline,
                                         const std::string &target) {
  translate_result result;

  // Parse the quake code
  try {
    mlir_module parsed_module(quake_code);

    // Run optimization
    if (auto err = optimize(parsed_module, pass_pipeline)) {
      result.error_message = "Optimization failed: " + *err;
      return result;
    }

    // Run translation
    result = translate(parsed_module, target);
    return result;

  } catch (const std::exception &e) {
    result.error_message = std::string("Compilation failed: ") + e.what();
    return result;
  }
}

} // namespace cudaq::compiler
