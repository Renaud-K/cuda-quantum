/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file utils.h
/// \brief Utility functions for QIR compilation and processing
///
/// Provides helper functions for working with compiled QIR modules,
/// including bitcode encoding and output name extraction.

#pragma once

#include "llvm_module.h"
#include "mlir_module.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// Forward declarations
namespace llvm {
class LLVMContext;
class Module;
}

namespace mlir {
class ModuleOp;
class MLIRContext;
}

namespace cudaq::compiler {

/// \brief Encode LLVM module to base64-encoded bitcode
///
/// Takes an llvm_module and encodes it as base64 bitcode suitable for
/// transmission to remote quantum backends. This is the standard format
/// for sending compiled QIR to hardware providers.
///
/// \param module_input LLVM module to encode
/// \return Base64-encoded bitcode string
///
/// \b Example:
/// \code
/// auto trans_result = compiler.translate(mlir_mod, "qir-base");
/// auto llvm_mod = trans_result.take_llvm_module();
/// std::string bitcode = encode_to_base64_bitcode(llvm_mod);
/// // Send bitcode to remote backend...
/// \endcode
std::string encode_to_base64_bitcode(const llvm_module &module_input);

/// \brief Extract output names from QIR LLVM module
///
/// Parses the QIR entry point function attributes to extract the
/// output name mappings, which describe how measurement results
/// are named in the compiled code. Returns a JSON string mapping
/// output indices to their semantic names.
///
/// \param module_input LLVM module containing QIR code
/// \return JSON object with output name mappings
///
/// \b Example:
/// \code
/// auto trans_result = compiler.translate(mlir_mod, "qir");
/// auto llvm_mod = trans_result.take_llvm_module();
/// std::string output_json = extract_output_names(llvm_mod);
/// // Parse JSON to get output mappings...
/// \endcode
std::string extract_output_names(const llvm_module &module_input);

// ============================================================================
// Translation Registry
// ============================================================================

/// \brief Translation function type: MLIR → (LLVMContext, LLVM Module)
using TranslationFunction =
    std::function<std::pair<std::unique_ptr<::llvm::LLVMContext>,
                            std::unique_ptr<::llvm::Module>>(
        ::mlir::ModuleOp, ::mlir::MLIRContext &)>;

/// \brief Get the global translation registry
///
/// Returns a registry mapping translation target names (e.g., "qir",
/// "qir-base") to translation functions.
std::unordered_map<std::string, TranslationFunction> &getTranslationRegistry();

/// \brief Initialize all built-in translations (QIR profiles, etc.)
///
/// This must be called before using translate functions. It registers
/// standard QIR profiles like "qir", "qir-base", and "qir-adaptive".
void initializeTranslations();

} // namespace cudaq::compiler
