/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_engine.h"
#include "llvm_module_impl.h"
#include "mlir_module_impl.h"

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

namespace cudaq::compiler {

struct execution_engine::impl {
  std::unique_ptr<::mlir::ExecutionEngine> mlirJit;
  std::unique_ptr<::llvm::orc::LLJIT> llvmJit;
};

// Helper function to load shared libraries
static void load_shared_libraries(const std::vector<std::string> &shared_libraries) {
  std::string errMsg;
  if (::llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr, &errMsg)) {
    throw std::runtime_error("Failed to load process symbols: " + errMsg);
  }

  for (const auto &lib : shared_libraries) {
    if (::llvm::sys::DynamicLibrary::LoadLibraryPermanently(lib.c_str(),
                                                            &errMsg)) {
      throw std::runtime_error("Failed to load shared library '" + lib +
                               "': " + errMsg);
    }
  }
}

// ==========================================================================
// New API: LLVM module constructors
// ==========================================================================

execution_engine::execution_engine(
    llvm_module &&module_input,
    const std::vector<std::string> &shared_libraries)
    : pimpl_(std::make_unique<impl>()) {

  // Initialize native target
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  load_shared_libraries(shared_libraries);

  auto *mod_impl = module_input.get_impl();
  if (!mod_impl || !mod_impl->llvm_module || !mod_impl->llvm_context) {
    throw std::runtime_error("Invalid LLVM module/context");
  }

  // Create the LLJIT builder
  auto jitBuilder = ::llvm::orc::LLJITBuilder();
  auto jitOrError = jitBuilder.create();
  if (!jitOrError) {
    throw std::runtime_error("Failed to create LLJIT: " +
                             ::llvm::toString(jitOrError.takeError()));
  }
  auto jit = std::move(*jitOrError);

  // Enable lookup of process symbols (including loaded shared libraries)
  jit->getMainJITDylib().addGenerator(cantFail(
      ::llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix())));

  // Create a ThreadSafeModule
  ::llvm::orc::ThreadSafeContext tsc(std::move(mod_impl->llvm_context));
  ::llvm::orc::ThreadSafeModule tsm(std::move(mod_impl->llvm_module), tsc);

  // Add the module to the JIT
  if (auto err = jit->addIRModule(std::move(tsm))) {
    throw std::runtime_error("Failed to add IR module to LLJIT: " +
                             ::llvm::toString(std::move(err)));
  }

  pimpl_->llvmJit = std::move(jit);
}

// ==========================================================================
// New API: MLIR module constructors
// ==========================================================================

execution_engine::execution_engine(
    mlir_module &&module_input,
    const std::vector<std::string> &shared_libraries)
    : pimpl_(std::make_unique<impl>()) {

  // Initialize native target
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  load_shared_libraries(shared_libraries);

  auto *mod_impl = module_input.get_impl();
  if (!mod_impl) {
    throw std::runtime_error("Invalid MLIR module implementation");
  }

  auto mlir_module = cudaq::compiler::get_mlir_module(mod_impl);
  if (!mlir_module) {
    throw std::runtime_error("Empty MLIR module");
  }

  ::mlir::ExecutionEngineOptions opts;

  std::vector<::llvm::StringRef> libPaths;
  libPaths.reserve(shared_libraries.size());
  for (const auto &lib : shared_libraries)
    libPaths.push_back(lib);

  opts.sharedLibPaths = libPaths;

  auto jitOrError = ::mlir::ExecutionEngine::create(mlir_module, opts);
  if (!jitOrError) {
    auto err = jitOrError.takeError();
    std::string errMsg;
    ::llvm::raw_string_ostream os(errMsg);
    os << err;
    throw std::runtime_error("Failed to create MLIR ExecutionEngine: " +
                             errMsg);
  }

  pimpl_->mlirJit = std::move(jitOrError.get());
}

execution_engine::execution_engine(
    mlir_module &module_input,
    const std::vector<std::string> &shared_libraries)
    : pimpl_(std::make_unique<impl>()) {

  // Initialize native target
  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  load_shared_libraries(shared_libraries);

  auto *mod_impl = module_input.get_impl();
  if (!mod_impl) {
    throw std::runtime_error("Invalid MLIR module implementation");
  }

  auto mlir_module = cudaq::compiler::get_mlir_module(mod_impl);
  if (!mlir_module) {
    throw std::runtime_error("Empty MLIR module");
  }

  ::mlir::ExecutionEngineOptions opts;

  std::vector<::llvm::StringRef> libPaths;
  libPaths.reserve(shared_libraries.size());
  for (const auto &lib : shared_libraries)
    libPaths.push_back(lib);

  opts.sharedLibPaths = libPaths;

  auto jitOrError = ::mlir::ExecutionEngine::create(mlir_module, opts);
  if (!jitOrError) {
    auto err = jitOrError.takeError();
    std::string errMsg;
    ::llvm::raw_string_ostream os(errMsg);
    os << err;
    throw std::runtime_error("Failed to create MLIR ExecutionEngine: " +
                             errMsg);
  }

  pimpl_->mlirJit = std::move(jitOrError.get());
}

execution_engine::~execution_engine() = default;

void *execution_engine::get_symbol(const std::string &name) {
  if (pimpl_ && pimpl_->mlirJit) {
    auto sym_or_err = pimpl_->mlirJit->lookup(name);
    if (!sym_or_err) {
      auto E = sym_or_err.takeError();
      throw std::runtime_error("Failed to look up symbol: " +
                               ::llvm::toString(std::move(E)));
    }
    return reinterpret_cast<void *>(*sym_or_err);
  }

  if (pimpl_ && pimpl_->llvmJit) {
    auto sym_or_err = pimpl_->llvmJit->lookup(name);
    if (!sym_or_err) {
      auto E = sym_or_err.takeError();
      throw std::runtime_error("Failed to look up symbol: " +
                               ::llvm::toString(std::move(E)));
    }
    return reinterpret_cast<void *>(sym_or_err->getValue());
  }

  return nullptr;
}

static constexpr int NUM_JIT_CACHE_ITEMS_TO_RETAIN = 100;

execution_cache::~execution_cache() {
  std::scoped_lock<std::mutex> lock(mutex);
  for (auto &[k, v] : cacheMap)
    v.execEngine.reset();
  cacheMap.clear();
}

bool execution_cache::has_engine(std::size_t hashkey) {
  std::scoped_lock<std::mutex> lock(mutex);
  return cacheMap.count(hashkey);
}

void execution_cache::cache(std::size_t hash,
                            std::unique_ptr<execution_engine> &&engine) {
  std::scoped_lock<std::mutex> lock(mutex);

  lruList.push_back(hash);

  // If adding a new item would exceed our cache limit, then remove the least
  // recently used item (at the head of the list).
  if (cacheMap.size() >= NUM_JIT_CACHE_ITEMS_TO_RETAIN) {
    auto hashToRemove = lruList.begin();
    auto it = cacheMap.find(*hashToRemove);
    it->second.execEngine.reset();
    lruList.erase(hashToRemove);
    cacheMap.erase(it);
  }

  cacheMap.insert(
      {hash, MapItemType{std::move(engine), std::prev(lruList.end())}});
}

execution_engine *execution_cache::get_engine(std::size_t hash) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto &item = cacheMap.at(hash);

  // Move item.lruListIt to the end of the list to indicate that it is being
  // used right now.
  lruList.splice(lruList.end(), lruList, item.lruListIt);

  return item.execEngine.get();
}

} // namespace cudaq::compiler
