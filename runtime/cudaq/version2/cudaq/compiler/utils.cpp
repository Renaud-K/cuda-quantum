/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "utils.h"
#include "llvm_module_impl.h"
#include "mlir_module_impl.h"
#include "nlohmann/json.hpp"

#include "cudaq/Optimizer/CodeGen/OptUtils.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <mutex>

namespace cudaq::compiler {

enum struct QirVersion { version_0_1, version_1_0 };

/// \brief Configuration for code generation translation
///
/// Parsed from profile strings like "qir-base", "qir-adaptive".
/// Controls validation, optimization, and code generation behavior.
struct CodeGenConfig {
  // Profile name
  std::string profile;
  // True if this is a QIR profile.
  bool isQIRProfile = false;
  // QIR profile version enum
  QirVersion version = QirVersion::version_1_0;
  // QIR profile major version
  std::uint32_t qir_major_version = 1;
  // QIR profile minor version
  std::uint32_t qir_minor_version = 0;
  // True if this is an adaptive QIR profile.
  bool isAdaptiveProfile = false;
  // True if this is a base QIR profile.
  bool isBaseProfile = false;
  // True if integer computation is enabled.
  bool integerComputations = false;
  // True if floating-point computation is enabled.
  bool floatComputations = false;
  // True if QIR output to log is enabled.
  bool outputLog = false;
  // True if we should erase stacksave/stackrestore instructions.
  bool eraseStackBounding = false;
  // True if we should erase measurement result recording functions.
  bool eraseRecordCalls = false;
  // True if we should bypass instruction validation, i.e., allow all
  // instructions.
  bool allowAllInstructions = false;
};

std::vector<std::string> splitString(const std::string &s,
                                     const char delimiter) {
  std::vector<std::string> tokens;
  size_t pos_start = 0, pos_end;
  std::string token;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + 1;
    tokens.push_back(token);
  }
  tokens.push_back(s.substr(pos_start)); // Add the last token
  return tokens;
}

std::tuple<std::string, std::string, std::vector<std::string>>
parseCodeGenTranslationString(const std::string &settingStr) {
  auto transportFields = splitString(settingStr, ':');
  auto size = transportFields.size();
  if (size == 1)
    return {transportFields[0], {}, {}};
  if (size == 2)
    return {transportFields[0], transportFields[1], {}};
  if (size == 3) {
    auto options = splitString(transportFields[2], ',');
    return {transportFields[0], transportFields[1], options};
  }
  throw std::runtime_error("Invalid codegen-emission string: " + settingStr);
}

CodeGenConfig parseCodeGenTranslation(const std::string &codegenTranslation) {
  auto [codeGenName, codeGenVersion, codeGenOptions] =
      parseCodeGenTranslationString(codegenTranslation);

  if (codeGenName.find("qir") == codeGenName.npos)
    return {.profile = codeGenName};

  CodeGenConfig config = {
      .profile = codeGenName,
      .isQIRProfile = true,
      .isAdaptiveProfile = codeGenName == "qir-adaptive",
      .isBaseProfile = codeGenName == "qir-base",
  };

  // Default version for base profile is 1.0
  if (config.isBaseProfile) {
    config.version = QirVersion::version_1_0;
    config.qir_major_version = 1;
    config.qir_minor_version = 0;
  }

  if (config.isAdaptiveProfile) {
    for (auto option : codeGenOptions) {
      if (option == "int_computations") {
        config.integerComputations = true;
      } else if (option == "float_computations") {
        config.floatComputations = true;
      } else if (option == "output_log") {
        config.outputLog = true;
      } else if (option == "erase_stack_bounding") {
        config.eraseStackBounding = true;
      } else if (option == "erase_record_calls") {
        config.eraseRecordCalls = true;
      } else if (option == "allow_all_instructions") {
        config.allowAllInstructions = true;
      } else {
        throw std::runtime_error("Invalid option '" + option + "' for '" +
                                 codeGenName + "' codegen.");
      }
    }
  } else {
    if (!codeGenOptions.empty())
      throw std::runtime_error("Invalid codegen-emission '" +
                               codegenTranslation +
                               "'. Extra options are not supported for '" +
                               codeGenName + "' codegen.");
  }

  if (config.isAdaptiveProfile) {
    // If no version is specified, using the lowest version
    if (codeGenVersion.empty() || codeGenVersion == "0.1") {
      config.version = QirVersion::version_0_1;
      config.qir_major_version = 0;
      config.qir_minor_version = 1;
    } else if (codeGenVersion == "1.0") {
      config.version = QirVersion::version_1_0;
      config.qir_major_version = 1;
      config.qir_minor_version = 0;
    } else {
      throw std::runtime_error("Unsupported QIR version '" + codeGenVersion +
                               "', codegen setting: " + codegenTranslation);
    }
  }

  return config;
}

// ============================================================================
// Target Triple Setup
// ============================================================================

bool setupTargetTriple(llvm::Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    return false;

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine)
    return false;

  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  return true;
}

// ============================================================================
// LLVM Optimization
// ============================================================================

void optimizeLLVM(llvm::Module *module) {
  auto optPipeline = cudaq::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(module)) {
    llvm::consumeError(std::move(err));
    throw std::runtime_error("Failed to optimize LLVM IR");
  }

  // Remove memory attributes from entry_point functions because the optimizer
  // sometimes applies it to degenerate cases (empty programs), and IonQ cannot
  // support that.
  for (llvm::Function &func : *module)
    if (func.hasFnAttribute("entry_point"))
      func.removeFnAttr(llvm::Attribute::Memory);
}

// ============================================================================
// QIR Attribute Application
// ============================================================================

void applyWriteOnlyAttributes(llvm::Module *llvmModule) {
  // Note that we only need to inspect QIRMeasureBody because MeasureCallConv
  // and MeasureToRegisterCallConv have already been called, so only
  // QIRMeasureBody remains.
  const unsigned int arg_num = 1;

  // Apply attribute to measurement function declaration
  if (auto func = llvmModule->getFunction(cudaq::opt::QIRMeasureBody)) {
    func->addParamAttr(arg_num, llvm::Attribute::WriteOnly);
  }

  // Apply to measurement function calls
  for (llvm::Function &func : *llvmModule)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block) {
        auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst);
        if (callInst && callInst->getCalledFunction()) {
          auto calledFunc = callInst->getCalledFunction();
          auto funcName = calledFunc->getName();
          if (funcName == cudaq::opt::QIRMeasureBody)
            callInst->addParamAttr(arg_num, llvm::Attribute::WriteOnly);
        }
      }
}

// ============================================================================
// Range Verification
// ============================================================================

// Convert a `nullptr` or `inttoptr (i64 1 to Ptr)` into an integer
static std::size_t getArgAsInteger(llvm::Value *arg) {
  std::size_t ret = 0; // handles the nullptr case
  // Handle constant integer case
  if (auto constInt = llvm::dyn_cast<llvm::ConstantInt>(arg)) {
    ret = constInt->getZExtValue();
  }
  // In newer LLVM, ConstantExpr was removed, so we check for PtrToInt
  // instructions
  else if (auto constValue = llvm::dyn_cast<llvm::Constant>(arg)) {
    // Try to get the value through constant folding if possible
    if (auto intVal = llvm::dyn_cast<llvm::ConstantInt>(constValue))
      ret = intVal->getZExtValue();
  }
  return ret;
}

#define CHECK_RANGE(_check_var, _limit_var)                                    \
  do {                                                                         \
    if (_check_var >= _limit_var) {                                            \
      llvm::errs() << #_check_var << " [" << _check_var                        \
                   << "] is >= " << #_limit_var << " [" << _limit_var          \
                   << "]\n";                                                   \
      return mlir::failure();                                                  \
    }                                                                          \
  } while (0)

mlir::LogicalResult verifyQubitAndResultRanges(llvm::Module *llvmModule) {
  std::size_t required_num_qubits = 0;
  std::size_t required_num_results = 0;
  for (llvm::Function &func : *llvmModule) {
    if (func.hasFnAttribute("entry_point")) {
      constexpr auto NotFound = std::numeric_limits<std::uint64_t>::max();
      required_num_qubits = func.getFnAttributeAsParsedInteger(
          cudaq::opt::qir0_1::RequiredQubitsAttrName, NotFound);
      if (required_num_qubits == NotFound)
        required_num_qubits = func.getFnAttributeAsParsedInteger(
            cudaq::opt::qir1_0::RequiredQubitsAttrName, 0);
      required_num_results = func.getFnAttributeAsParsedInteger(
          cudaq::opt::qir0_1::RequiredResultsAttrName, NotFound);
      if (required_num_results == NotFound)
        required_num_results = func.getFnAttributeAsParsedInteger(
            cudaq::opt::qir1_0::RequiredResultsAttrName, 0);
      break; // no need to keep looking
    }
  }
  for (llvm::Function &func : *llvmModule) {
    for (llvm::BasicBlock &block : func) {
      for (llvm::Instruction &inst : block) {
        if (auto callInst = llvm::dyn_cast_or_null<llvm::CallBase>(&inst)) {
          if (auto func = callInst->getCalledFunction()) {
            // All results must be in range for output recording functions
            if (func->getName() == cudaq::opt::QIRRecordOutput) {
              auto result = getArgAsInteger(callInst->getArgOperand(0));
              CHECK_RANGE(result, required_num_results);
            }
            // All qubits and results must be in range for measurements
            else if (func->getName() == cudaq::opt::QIRMeasureBody) {
              auto qubit = getArgAsInteger(callInst->getArgOperand(0));
              auto result = getArgAsInteger(callInst->getArgOperand(1));
              CHECK_RANGE(qubit, required_num_qubits);
              CHECK_RANGE(result, required_num_results);
            }
          }
        }
      }
    }
  }
  return mlir::success();
}

// ============================================================================
// Code Pattern Filtering
// ============================================================================

mlir::LogicalResult filterSpecificCodePatterns(llvm::Module *llvmModule,
                                               CodeGenConfig &config) {
  bool erasePatterns = config.outputLog;
  bool eraseStackBounding = config.eraseStackBounding;
  bool eraseResultRecordCalls = config.eraseRecordCalls;

  if (erasePatterns || eraseStackBounding || eraseResultRecordCalls) {
    llvm::SmallVector<llvm::Instruction *> eraseInst;
    for (llvm::Function &func : *llvmModule)
      for (llvm::BasicBlock &block : func)
        for (llvm::Instruction &inst : block)
          if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
            auto *calledFunc = call->getCalledFunction();
            if (!calledFunc)
              continue;
            auto name = calledFunc->getGlobalIdentifier();
            if (eraseStackBounding && calledFunc->isIntrinsic() &&
                (name.find("llvm.stacksave") != std::string::npos ||
                 name.find("llvm.stackrestore") != std::string::npos))
              eraseInst.push_back(&inst);
            if (eraseResultRecordCalls && name == cudaq::opt::QIRRecordOutput)
              eraseInst.push_back(&inst);
          }
    for (auto *insn : eraseInst) {
      if (insn->hasNUsesOrMore(1)) {
        // Use PoisonValue in newer LLVM versions (UndefValue was deprecated)
        insn->replaceAllUsesWith(llvm::PoisonValue::get(insn->getType()));
      }
      insn->eraseFromParent();
    }
  }
  return mlir::success();
}

// ============================================================================
// QIR Profile Validation
// ============================================================================

static bool isValidIntegerArithmeticInstruction(llvm::Instruction &inst) {
  const auto isValidIntegerBinaryInst = [](const auto &inst) {
    if (!llvm::isa<llvm::BinaryOperator>(inst))
      return false;
    const auto opCode = inst.getOpcode();
    static const std::vector<int> integerOps = {
        llvm::BinaryOperator::Add,  llvm::BinaryOperator::Sub,
        llvm::BinaryOperator::Mul,  llvm::BinaryOperator::UDiv,
        llvm::BinaryOperator::SDiv, llvm::BinaryOperator::URem,
        llvm::BinaryOperator::SRem, llvm::BinaryOperator::And,
        llvm::BinaryOperator::Or,   llvm::BinaryOperator::Xor,
        llvm::BinaryOperator::Shl,  llvm::BinaryOperator::LShr,
        llvm::BinaryOperator::AShr};
    return std::find(integerOps.begin(), integerOps.end(), opCode) !=
           integerOps.end();
  };

  return isValidIntegerBinaryInst(inst) ||
         llvm::isa<llvm::ICmpInst, llvm::ZExtInst, llvm::SExtInst,
                   llvm::TruncInst, llvm::SelectInst, llvm::PHINode>(inst);
}

static bool isValidFloatingArithmeticInstruction(llvm::Instruction &inst) {
  const auto isValidFloatBinaryInst = [](const auto &inst) {
    if (!llvm::isa<llvm::BinaryOperator>(inst))
      return false;
    const auto opCode = inst.getOpcode();
    static const std::vector<int> floatOps = {
        llvm::BinaryOperator::FAdd, llvm::BinaryOperator::FSub,
        llvm::BinaryOperator::FMul, llvm::BinaryOperator::FDiv,
        llvm::BinaryOperator::FRem};
    return std::find(floatOps.begin(), floatOps.end(), opCode) !=
           floatOps.end();
  };

  return isValidFloatBinaryInst(inst) ||
         llvm::isa<llvm::FCmpInst, llvm::FPExtInst, llvm::FPTruncInst,
                   llvm::FPToSIInst, llvm::FPToUIInst, llvm::SIToFPInst,
                   llvm::UIToFPInst, llvm::SelectInst, llvm::PHINode>(inst);
}

mlir::LogicalResult validateQIRProfile(llvm::Module *llvmModule,
                                       const CodeGenConfig &config) {
  if (!config.isQIRProfile)
    return mlir::success();

  // Profile-specific validation
  bool isAdaptiveProfile = config.profile.find("adaptive") != std::string::npos;
  bool allowIntegerCompute = false; // Could be parsed from config
  bool allowFloatingCompute = false;

  for (llvm::Function &func : *llvmModule) {
    for (llvm::BasicBlock &block : func) {
      for (llvm::Instruction &inst : block) {
        // Check for invalid instructions based on profile
        if (!isAdaptiveProfile) {
          // Base profile: very restricted
          if (llvm::isa<llvm::BinaryOperator, llvm::ICmpInst, llvm::FCmpInst>(
                  inst)) {
            llvm::errs() << "Base profile does not support arithmetic: " << inst
                         << "\n";
            return mlir::failure();
          }
        } else {
          // Adaptive profile: check integer/float compute capabilities
          if (llvm::isa<llvm::BinaryOperator>(inst)) {
            if (inst.getType()->isIntOrIntVectorTy()) {
              if (!allowIntegerCompute &&
                  !isValidIntegerArithmeticInstruction(inst)) {
                llvm::errs()
                    << "Adaptive profile without integer compute capability: "
                    << inst << "\n";
                return mlir::failure();
              }
            } else if (inst.getType()->isFPOrFPVectorTy()) {
              if (!allowFloatingCompute &&
                  !isValidFloatingArithmeticInstruction(inst)) {
                llvm::errs()
                    << "Adaptive profile without floating compute capability: "
                    << inst << "\n";
                return mlir::failure();
              }
            }
          }
        }
      }
    }
  }

  return mlir::success();
}

std::string encode_to_base64_bitcode(const llvm_module &module_input) {
  // Extract LLVM module from wrapper
  auto *impl = module_input.get_impl();
  auto &llvmModule = impl->llvm_module;

  if (!llvmModule)
    return "";

  // Write bitcode to string buffer
  llvm::SmallString<1024> bitCodeMem;
  llvm::raw_svector_ostream os(bitCodeMem);
  llvm::WriteBitcodeToFile(*llvmModule, os);

  // Encode to base64
  return llvm::encodeBase64(bitCodeMem.str());
}

std::string extract_output_names(const llvm_module &module_input) {
  // Extract LLVM module from wrapper
  auto *impl = module_input.get_impl();
  auto &llvmModule = impl->llvm_module;

  if (!llvmModule)
    return nlohmann::json::object().dump();

  // Find the entry point function
  llvm::Function *entryPoint = nullptr;
  for (auto &F : *llvmModule) {
    if (F.hasFnAttribute(cudaq::opt::QIREntryPointAttrName)) {
      entryPoint = &F;
      break;
    }
  }

  if (!entryPoint)
    return nlohmann::json::object().dump();

  // Get the output_names attribute
  if (auto attr =
          entryPoint->getFnAttribute(cudaq::opt::QIROutputNamesAttrName);
      attr.isStringAttribute()) {
    std::string outputNamesStr = attr.getValueAsString().str();
    try {
      return nlohmann::json::parse(outputNamesStr).dump();
    } catch (const std::exception &) {
      return nlohmann::json::object().dump();
    }
  }

  return nlohmann::json::object().dump();
}

// ============================================================================
// Translation Registry
// ============================================================================

std::unordered_map<std::string, TranslationFunction> &getTranslationRegistry() {
  static std::unordered_map<std::string, TranslationFunction> registry;
  return registry;
}

void registerTranslation(const std::string &name, TranslationFunction func) {
  getTranslationRegistry()[name] = std::move(func);
}

void addAdaptiveProfileFlags(llvm::Module *llvmModule,
                             llvm::LLVMContext &llvmContext,
                             const CodeGenConfig &config) {
  if (!llvmModule)
    return;

  auto falseValue =
      llvm::ConstantInt::getFalse(llvm::Type::getInt1Ty(llvmContext));
  auto trueValue =
      llvm::ConstantInt::getTrue(llvm::Type::getInt1Ty(llvmContext));

  // Add dynamic management flags (common to all profiles)
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRDynamicQubitsManagementFlagName,
                            falseValue);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRDynamicResultManagementFlagName,
                            falseValue);

  if (config.isAdaptiveProfile) {
    if (config.version == QirVersion::version_0_1) {
      // QIR 0.1 adaptive profile flags
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::QubitResettingFlagName,
                                trueValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::ClassicalIntsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::ClassicalFloatsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(
          llvm::Module::ModFlagBehavior::Error,
          cudaq::opt::qir0_1::ClassicalFixedPointsFlagName, falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::UserFunctionsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::DynamicFloatArgsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::ExternFunctionsFlagName,
                                falseValue);
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir0_1::BackwardsBranchingFlagName,
                                falseValue);
    } else {
      // QIR 1.0 adaptive profile flags
      if (config.integerComputations) {
        llvm::Constant *intPrecisionValue =
            llvm::ConstantDataArray::getString(llvmContext, "i64", false);
        llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                  cudaq::opt::qir1_0::IntComputationsFlagName,
                                  intPrecisionValue);
      }
      if (config.floatComputations) {
        llvm::Constant *floatPrecisionValue =
            llvm::ConstantDataArray::getString(llvmContext, "f64", false);
        llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                  cudaq::opt::qir1_0::FloatComputationsFlagName,
                                  floatPrecisionValue);
      }
      auto backwardsBranchingValue = llvm::ConstantInt::getIntegerValue(
          llvm::Type::getIntNTy(llvmContext, 2), llvm::APInt(2, 0, false));
      llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                cudaq::opt::qir1_0::BackwardsBranchingFlagName,
                                backwardsBranchingValue);
    }
  }
}

// ============================================================================
// QIR Profile Translation Implementation
// ============================================================================
std::pair<std::unique_ptr<llvm::LLVMContext>, std::unique_ptr<llvm::Module>>
qirProfileTranslation(const std::string &qirProfile, mlir::ModuleOp moduleOp,
                      mlir::MLIRContext &context, bool printIR = false,
                      bool printIntermediateMLIR = false,
                      bool printStats = false) {

  auto config = parseCodeGenTranslation(qirProfile);
  if (!config.isQIRProfile) {
    llvm::errs() << "Unexpected codegen profile while translating to QIR: "
                 << config.profile << "\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Create pass manager
  mlir::PassManager pm(&context);
  if (printIntermediateMLIR)
    pm.enableIRPrinting();
  if (printStats)
    pm.enableStatistics();

  // Check if module contains WireSet operations
  bool containsWireSet =
      moduleOp
          .walk<mlir::WalkOrder::PreOrder>([](quake::WireSetOp wireSetOp) {
            return mlir::WalkResult::interrupt();
          })
          .wasInterrupted();

  // Add appropriate pipeline
  if (containsWireSet)
    cudaq::opt::addWiresetToProfileQIRPipeline(pm, config.profile);
  else
    cudaq::opt::addAOTPipelineConvertToQIR(pm, qirProfile);

  // Run the pass pipeline
  if (mlir::failed(pm.run(moduleOp))) {
    llvm::errs() << "Failed to run QIR conversion pipeline\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Translate MLIR to LLVM IR
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  llvmContext->setOpaquePointers(false);
  auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp, *llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR to LLVM IR\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Apply required attributes for the profile
  applyWriteOnlyAttributes(llvmModule.get());

  // Add required module flags for QIR
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                            cudaq::opt::QIRMajorVersionFlagName,
                            config.qir_major_version);
  llvmModule->addModuleFlag(llvm::Module::ModFlagBehavior::Max,
                            cudaq::opt::QIRMinorVersionFlagName,
                            config.qir_minor_version);

  // Add adaptive profile flags
  addAdaptiveProfileFlags(llvmModule.get(), *llvmContext, config);

  // Setup target triple
  if (!setupTargetTriple(llvmModule.get())) {
    llvm::errs() << "Failed to setup target triple\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Filter code patterns if needed
  if (mlir::failed(filterSpecificCodePatterns(llvmModule.get(), config))) {
    llvm::errs() << "Failed to filter code patterns\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Verify qubit and result ranges
  if (mlir::failed(verifyQubitAndResultRanges(llvmModule.get()))) {
    llvm::errs() << "Failed qubit/result range verification\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Validate profile compliance
  if (mlir::failed(validateQIRProfile(llvmModule.get(), config))) {
    llvm::errs() << "Failed QIR profile validation\n";
    return std::make_pair(nullptr, nullptr);
  }

  // Optimize if not printing intermediate IR
  if (!printIntermediateMLIR) {
    optimizeLLVM(llvmModule.get());
  }

  // Print IR if requested
  if (printIR) {
    llvmModule->print(llvm::errs(), nullptr);
  }

  return std::make_pair(std::move(llvmContext), std::move(llvmModule));
}

// ============================================================================
// Specific Translation Functions
// ============================================================================

static std::pair<std::unique_ptr<llvm::LLVMContext>,
                 std::unique_ptr<llvm::Module>>
translateToQIR(mlir::ModuleOp moduleOp, mlir::MLIRContext &context) {
  return qirProfileTranslation("qir", moduleOp, context);
}

static std::pair<std::unique_ptr<llvm::LLVMContext>,
                 std::unique_ptr<llvm::Module>>
translateToQIRBase(mlir::ModuleOp moduleOp, mlir::MLIRContext &context) {
  return qirProfileTranslation("qir-base", moduleOp, context);
}

static std::pair<std::unique_ptr<llvm::LLVMContext>,
                 std::unique_ptr<llvm::Module>>
translateToQIRAdaptive(mlir::ModuleOp moduleOp, mlir::MLIRContext &context) {
  return qirProfileTranslation("qir-adaptive", moduleOp, context);
}

// ============================================================================
// Translation Registration
// ============================================================================

void initializeTranslations() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    registerTranslation("qir", translateToQIR);
    registerTranslation("qir-base", translateToQIRBase);
    registerTranslation("qir-adaptive", translateToQIRAdaptive);

    // Future translations can be added here:
    // registerTranslation("qasm2", translateToQASM2);
    // registerTranslation("iqm", translateToIQM);
    // registerTranslation("llvm", translateToLLVM);
  });
}

} // namespace cudaq::compiler
