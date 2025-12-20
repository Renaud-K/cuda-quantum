/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file mlir_module_impl.h
/// \brief Internal implementation details for mlir::module (private header)

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <memory>
#include <variant>

namespace cudaq::compiler {

// Forward declare the module class
class module;

// Implementation struct for MLIR modules
struct mlir_module_impl {
  // Variant to hold either:
  // 1. Owning: OwningOpRef + MLIRContext
  // 2. Non-owning: Just ModuleOp reference
  std::variant<std::pair<::mlir::OwningOpRef<::mlir::ModuleOp>,
                         std::unique_ptr<::mlir::MLIRContext>>,
               ::mlir::ModuleOp>
      content;

  bool owns_data = true;

  // Constructor declarations (defined in mlir_module.cpp)
  mlir_module_impl();
  mlir_module_impl(::mlir::ModuleOp mod); // Non-owning
  mlir_module_impl(::mlir::OwningOpRef<::mlir::ModuleOp> mod,
              std::unique_ptr<::mlir::MLIRContext> ctx); // Owning

  std::size_t hash() const;
};

// Helper to get ModuleOp from impl
::mlir::ModuleOp get_mlir_module(mlir_module_impl *impl);

} // namespace cudaq::compiler::mlir

