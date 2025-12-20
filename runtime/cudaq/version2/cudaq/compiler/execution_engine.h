/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file execution_engine.h
/// \brief JIT execution engine for compiled modules
///
/// Provides a wrapper around LLVM's JIT engine (LLJIT) to execute compiled
/// code. Supports loading shared libraries and retrieving function pointers.

#pragma once

#include "llvm_module.h"
#include "mlir_module.h"
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::compiler {

/// \brief JIT execution engine
///
/// Wraps LLVM's LLJIT to execute compiled quantum kernels. Takes either
/// an llvm_module (QIR) or mlir_module (LLVM dialect MLIR) and prepares
/// it for in-process execution. Supports dynamic linking of shared libraries
/// for runtime functions and quantum gate implementations.
///
/// \b Example Usage (from LLVM module):
/// \code
/// // Full compilation pipeline: MLIR → optimization → translation → execution
/// cudaq::compiler::compiler comp;
/// cudaq::compiler::mlir_module mlir_mod(quake_code);
/// 
/// // Optimize and translate to QIR
/// auto trans_result = comp.translate(std::move(mlir_mod), "qir");
/// auto llvm_mod = trans_result.take_llvm_module();
/// 
/// // Create execution engine with runtime libraries
/// std::vector<std::string> libs = {"/path/to/libqir-runtime.so"};
/// cudaq::compiler::execution_engine engine(std::move(llvm_mod), libs);
/// 
/// // Get kernel function pointer and execute
/// auto *kernel_fn = engine.get_symbol("my_kernel_name");
/// auto kernel = reinterpret_cast<void(*)()>(kernel_fn);
/// kernel();  // Execute the quantum kernel
/// \endcode
///
/// \b Example Usage (from MLIR module with LLVM dialect):
/// \code
/// // Direct execution from MLIR containing LLVM dialect IR
/// // (skips separate translation step)
/// cudaq::compiler::mlir_module mlir_llvm_mod(llvm_dialect_ir);
/// 
/// std::vector<std::string> libs = {"/path/to/runtime.so"};
/// cudaq::compiler::execution_engine engine(std::move(mlir_llvm_mod), libs);
/// 
/// // Execute as before
/// auto *fn = engine.get_symbol("kernel_name");
/// reinterpret_cast<void(*)()>(fn)();
/// \endcode
class execution_engine {
private:
  struct impl;
  std::unique_ptr<impl> pimpl_;

public:

  /// \brief Construct JIT engine from an LLVM module (consuming)
  ///
  /// Takes ownership of the llvm_module and creates a JIT engine for
  /// immediate execution. Use this for the typical execution path.
  ///
  /// \param module_input The LLVM module (consumed via move)
  /// \param shared_libraries List of paths to shared libraries to load
  /// \throws std::runtime_error if JIT creation fails
  /// \note Non-consuming constructor is not provided as LLVM module cloning
  ///       is complex and not generally supported
  execution_engine(llvm_module &&module_input,
                   const std::vector<std::string> &shared_libraries = {});

  /// \brief Construct JIT engine from an MLIR module (consuming)
  ///
  /// Takes ownership of the mlir_module (must contain LLVM dialect IR)
  /// and creates a JIT engine. Useful when skipping the translation step.
  ///
  /// \param module_input The MLIR module with LLVM dialect (consumed via move)
  /// \param shared_libraries List of paths to shared libraries to load
  /// \throws std::runtime_error if JIT creation fails
  execution_engine(mlir_module &&module_input,
                   const std::vector<std::string> &shared_libraries = {});

  /// \brief Construct JIT engine from an MLIR module (non-consuming)
  ///
  /// References the mlir_module (must contain LLVM dialect IR) without
  /// taking ownership. The module must outlive the execution_engine instance.
  ///
  /// \param module_input The MLIR module with LLVM dialect (referenced)
  /// \param shared_libraries List of paths to shared libraries to load
  /// \throws std::runtime_error if JIT creation fails
  execution_engine(mlir_module &module_input,
                   const std::vector<std::string> &shared_libraries = {});

  ~execution_engine();

  /// \brief Get a pointer to a symbol (function) in the JITed code
  ///
  /// Looks up a symbol by name in the JITed code. Typically used to
  /// retrieve kernel entry points for execution.
  ///
  /// \param name The mangled name of the symbol to look up
  /// \return Pointer to the symbol, or nullptr if not found
  /// \throws std::runtime_error if symbol lookup fails due to JIT error
  /// \note Returns nullptr for missing symbols (expected behavior), but
  ///       throws exception for actual lookup failures (unexpected errors)
  void *get_symbol(const std::string &name);
};

/// \brief The execution_cache is a utility class for storing ExecutionEngine
/// pointers keyed on the hash for the string representation of the original
/// MLIR ModuleOp.
///
/// \b Thread Safety:
/// - **Thread-safe**: All public methods are protected by an internal mutex.
/// - Safe to use concurrently from multiple threads without external synchronization.
/// - Uses LRU eviction policy with configurable cache size.
class execution_cache {
protected:
  // Implement a Least Recently Used cache based on the JIT hash.
  std::list<std::size_t> lruList;

  // A given JIT hash has an associated MapItemType, which contains pointers to
  // the execution engine and to the LRU iterator that is used to track which
  // engine is the least recently used.
  struct MapItemType {
    std::unique_ptr<execution_engine> execEngine = nullptr;
    std::list<std::size_t>::iterator lruListIt;
  };
  std::unordered_map<std::size_t, MapItemType> cacheMap;

  std::mutex mutex;

public:
  execution_cache() = default;
  ~execution_cache();

  void cache(std::size_t hash, std::unique_ptr<execution_engine> &&engine);
  bool has_engine(std::size_t hash);
  execution_engine *get_engine(std::size_t hash);
};

} // namespace cudaq::compiler
