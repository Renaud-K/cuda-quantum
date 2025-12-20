/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "compiler.h"
#include "llvm_module.h"
#include "mlir_module.h"
#include <string>

namespace cudaq::compiler {

/// \brief Translation engine for MLIR modules
///
/// Translates MLIR (Quake IR) to various target formats. Supports both
/// consuming and non-consuming patterns for flexible ownership management.
///
/// \b Supported Targets:
/// - **"qir"**: Full QIR profile (returns llvm_module)
/// - **"qir-base"**: QIR Base profile, no classical control (returns llvm_module)
/// - **"qir-adaptive"**: QIR Adaptive profile with classical control (returns llvm_module)
/// - **"openqasm2"**: OpenQASM 2.0 text format (returns std::string)
///
/// The result is wrapped in a translate_result which contains a variant
/// of either an llvm_module or a std::string, depending on the target.
///
/// \b Example:
/// \code
/// cudaq::compiler::mlir_module mod(quake_code);
/// 
/// // Translate to QIR (returns LLVM module)
/// auto qir_result = translator::run(mod, "qir-base");
/// if (qir_result.success() && qir_result.is_llvm_module()) {
///   auto llvm_mod = qir_result.take_llvm_module();
///   std::string bitcode = llvm_mod.encode_to_base64_bitcode();
/// }
/// 
/// // Translate to OpenQASM (returns string)
/// auto qasm_result = translator::run(mod, "openqasm2");
/// if (qasm_result.success() && qasm_result.is_string()) {
///   std::string qasm_code = qasm_result.get_string();
/// }
/// \endcode
class translator {
public:
  /// \brief Translate MLIR module to target format (consuming)
  ///
  /// Translates the MLIR module to the target format. For QIR targets,
  /// returns an llvm_module. For other targets like OpenQASM, returns a string.
  /// Use this when you're done with the MLIR module and want to move it.
  ///
  /// \param module_input Input MLIR module to translate (consumed via move)
  /// \param target Translation target (e.g., "qir", "qir-base", "openqasm2")
  /// \return translate_result containing either llvm_module or std::string
  static translate_result run(mlir_module &&module_input,
                              const std::string &target);

  /// \brief Translate MLIR module to target format (preserving)
  ///
  /// Clones the MLIR module internally and translates it to the target format.
  /// The original module remains unchanged. Use this when you need to preserve
  /// the MLIR module for further processing or inspection.
  ///
  /// \param module_input Input MLIR module to translate (preserved)
  /// \param target Translation target (e.g., "qir", "qir-base", "openqasm2")
  /// \return translate_result containing either llvm_module or std::string
  static translate_result run(const mlir_module &module_input,
                              const std::string &target);
};

} // namespace cudaq::compiler
