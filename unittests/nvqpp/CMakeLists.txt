# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Phase 2 of the nvq++ unittests migration (see
# docs/refactor/investigations/nvqpp-unittests-migration.md in the refactor
# workspace): every executable defined in this subdirectory (and its
# add_subdirectory() descendants, e.g. backends/) is compiled AND linked by
# nvq++ instead of the ambient g++. This must not leak to the parent
# unittests/ directory.
#
# A few subdirectories here define plain host C++ SHARED plugin libraries
# rather than test executables (e.g. qudit/simple_qudit's cudaq-em-qudit,
# backends/quake_backend's server-helper stubs) -- those must keep compiling
# with the ambient host compiler, not nvq++ (nvq++ is an executable compiler
# driver: it always assembles a full CUDA-Q runtime link line, which is wrong
# for a lean plugin .so). Capture the ambient compiler here so those
# subdirectories can restore it for themselves before their own
# add_library() calls.
set(CUDAQ_HOST_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_COMPILER "${CMAKE_BINARY_DIR}/bin/nvq++")
set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

# ccache (enabled project-wide via CMAKE_CXX_COMPILER_LAUNCHER, see
# scripts/build_cudaq.sh) cannot cache nvq++ invocations: nvq++.in has no
# support for -E, so when ccache probes it with "-E -o foo.ii" to build its
# manifest, nvq++ ignores that intent and runs its normal quake+compile+link
# pipeline instead, treating foo.ii as the final link output. That fails (or
# at best produces the wrong kind of artifact), so ccache marks every call
# uncacheable and just re-runs the real nvq++ every time -- pure overhead,
# never a hit. Disable the launcher for this subtree only.
set(CMAKE_CXX_COMPILER_LAUNCHER "")
set(CMAKE_CXX_COMPILER_LAUNCHER "")

# CUDAQ_LIBRARY_MODE (set directory-wide in the parent unittests/CMakeLists.txt
# for the now-retired g++ + inline-qubit_qis.h compilation path) must not be
# defined for sources compiled by nvq++: nvq++ runs the real AST bridge
# (cudaq-quake) for every TU, so __qpu__ kernels always get real Quake IR
# here, not host-inlined gate/measure bodies.
get_directory_property(defs COMPILE_DEFINITIONS)
list(REMOVE_ITEM defs CUDAQ_LIBRARY_MODE)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${defs}")

# COMPILE_OPTIONS inherited from the top-level CMakeLists.txt include a few
# GCC-only warning suppressions (-Wno-restrict, -Wno-stringop-overflow,
# -Wno-stringop-overread) added there because that check runs against the
# ambient host compiler (see CMakeLists.txt around CUDAQ_HAS_WNO_RESTRICT).
# nvq++'s bundled clang++ doesn't recognize them and (with this toolchain's
# default of erroring on unknown warning options) fails every TU. Strip just
# those GCC-specific entries here, same pattern as CUDAQ_LIBRARY_MODE above;
# leave the rest (color diagnostics, -Wno-error=deprecated-declarations)
# alone since clang understands those too.
get_directory_property(opts COMPILE_OPTIONS)
list(REMOVE_ITEM opts 
  -Wno-restrict 
  -Wno-stringop-overflow 
  -Wno-stringop-overread
  --param=evrp-mode=legacy
)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${opts}")

# INCLUDE_DIRECTORIES inherited from the top-level CMakeLists.txt --
# LLVM/Clang/MLIR headers and cudaq/include's compiler-internal (AST bridge)
# headers (for building cudaq-quake/cudaq-opt/etc. themselves), and
# tpls/json/single_include[_fwd] -- are irrelevant to sources compiled by
# nvq++: nvq++ never runs those TUs through the MLIR-based compiler
# internals, and supplies its own consistent -I set for the CUDA-Q runtime
# headers internally (see the INCLUDES logic in nvq++.in). Clear them here so
# nvq++ invocations for this subtree aren't cluttered with -I/-isystem paths
# no test source in this subtree includes. Targets that still need
# unittests/'s own CUDAQTestUtils.h (normally reached via the parent
# unittests/CMakeLists.txt's directory-wide include_directories()) must add
# ".." to their own target_include_directories() -- see call sites below.
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/..")

include(AddNvqppExecutable.cmake)

set(CUDAQ_RUNTIME_TEST_SOURCES
  # Integration tests
  integration/adjoint_tester.cpp
  integration/builder_tester.cpp
  integration/ccnot_tester.cpp
  integration/deuteron_variational_tester.cpp
  integration/draw_tester.cpp
  integration/ghz_nisq_tester.cpp
  integration/gradient_tester.cpp
  integration/grover_test.cpp
  integration/qpe_ftqc.cpp
  integration/qpe_nisq.cpp
  integration/qubit_allocation.cpp
  integration/vqe_tester.cpp
  integration/bug67_vqe_then_sample.cpp
  integration/bug77_vqe_with_shots.cpp
  integration/bug116_cusv_measure_bug.cpp
  integration/async_tester.cpp
  integration/negative_controls_tester.cpp
  integration/observe_result_tester.cpp
  integration/noise_tester.cpp
  integration/get_state_tester.cpp
  integration/encoding_tester.cpp
  integration/measure_reset_tester.cpp
  integration/mps_to_host_tester.cpp
  qir/NVQIRTester.cpp  
  qis/QubitQISTester.cpp
  integration/kernels_tester.cpp
  common/MeasureCountsTester.cpp
  common/NoiseModelTester.cpp
  common/CompiledModuleTester.cpp
  ptsbe/PTSBESampleTester.cpp
  ptsbe/PTSBEMultiBackendTester.cpp
  integration/tracer_tester.cpp
  integration/gate_library_tester.cpp
  integration/analysis_scope_tester.cpp
)
if (CUDAQ_ENABLE_NLOPT)
  list(APPEND CUDAQ_RUNTIME_TEST_SOURCES integration/nlopt_tester.cpp)
endif()

# Make it so we can get function symbols
set (CMAKE_ENABLE_EXPORTS TRUE)

## This Macro allows us to create a test_runtime executable for
## the sources in CUDAQ_RUNTIME_TEST_SOURCE for a specific backend simulator
macro (create_tests_with_backend NVQIR_BACKEND EXTRA_BACKEND_TESTER)
  set(TEST_EXE_NAME "test_runtime_${NVQIR_BACKEND}")
  string(REPLACE "-" "_" NVQIR_BACKEND_OUT ${NVQIR_BACKEND})

  set(TEST_LABELS "")
  set(TEST_PROCESSORS 1)

  # Map the NVQIR_BACKEND name (used for gtest suite naming and, before
  # nvq++ assembled the link line, for the explicit nvqir-${NVQIR_BACKEND}
  # link) to the nvq++ --target (and --target-option, for targets that use a
  # configuration-matrix) that resolves to the same nvqir-* library.
  set(NVQPP_TEST_TARGET_OPTION "")
  if (${NVQIR_BACKEND} STREQUAL "qpp")
    set(NVQPP_TEST_TARGET "qpp-cpu")
    set(NVQPP_TEST_TARGET_OPTION "--target-option" "fp64")
  elseif (${NVQIR_BACKEND} STREQUAL "dm")
    set(NVQPP_TEST_TARGET "density-matrix-cpu")
    set(NVQPP_TEST_TARGET_OPTION "-DCUDAQ_BACKEND_DM")
  elseif (${NVQIR_BACKEND} STREQUAL "stim")
    set(NVQPP_TEST_TARGET "stim")
    set(NVQPP_TEST_TARGET_OPTION "-DCUDAQ_BACKEND_STIM")
  elseif (${NVQIR_BACKEND} STREQUAL "tensornet")
    set(NVQPP_TEST_TARGET "tensornet")
    set(NVQPP_TEST_TARGET_OPTION "--target-option" "fp64" "-DCUDAQ_BACKEND_TENSORNET")
    set(TEST_LABELS "gpu_required")
  elseif (${NVQIR_BACKEND} STREQUAL "tensornet-mps")
    set(NVQPP_TEST_TARGET "tensornet-mps")
    set(NVQPP_TEST_TARGET_OPTION "--target-option" "fp64" "-DCUDAQ_BACKEND_TENSORNET" "-DCUDAQ_BACKEND_TENSORNET_MPS") 
    set(TEST_LABELS "gpu_required")
  elseif (${NVQIR_BACKEND} STREQUAL "tensornet-fp32")
    set(NVQPP_TEST_TARGET "tensornet")
    set(NVQPP_TEST_TARGET_OPTION "--target-option" "fp32" "-DCUDAQ_BACKEND_TENSORNET") 
    set(TEST_LABELS "gpu_required")
  elseif (${NVQIR_BACKEND} STREQUAL "tensornet-mps-fp32")
    set(NVQPP_TEST_TARGET "tensornet-mps")
    set(NVQPP_TEST_TARGET_OPTION "--target-option" "fp32" "-DCUDAQ_BACKEND_TENSORNET" "-DCUDAQ_BACKEND_TENSORNET_MPS")
    set(TEST_LABELS "gpu_required")
  elseif (${NVQIR_BACKEND} STREQUAL "custatevec-fp32")
    # nvidia-legacy always resolves to nvqir-custatevec-fp32 (no cusvsim
    # substitution), matching the explicit nvqir-custatevec-fp32 link this
    # macro used before nvq++ assembled the link line.
    set(NVQPP_TEST_TARGET "nvidia-legacy")
    set(NVQPP_TEST_TARGET_OPTION "--target-option" "fp32" "-DCUDAQ_BACKEND_CUSTATEVEC_FP3") 
    set(TEST_LABELS "gpu_required")
  else()
    message(FATAL_ERROR "create_tests_with_backend: no nvq++ --target mapping for backend '${NVQIR_BACKEND}'")
  endif()

  add_nvqpp_executable(${TEST_EXE_NAME}
    SOURCES ${CUDAQ_GTEST_MAIN} ${CUDAQ_RUNTIME_TEST_SOURCES} ${EXTRA_BACKEND_TESTER}
    NVQPP_FLAGS --target=${NVQPP_TEST_TARGET} ${NVQPP_TEST_TARGET_OPTION}
    EXTRA_LINK_LIBS gtest_main
    EXTRA_DEPS nvqir-${NVQIR_BACKEND}
  )
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DNVQIR_BACKEND_NAME=${NVQIR_BACKEND_OUT})
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE __MATH_LONG_DOUBLE_CONSTANTS)
  target_include_directories(${TEST_EXE_NAME} PRIVATE . ..)

  if (${NVQIR_BACKEND} STREQUAL "qpp" OR ${NVQIR_BACKEND} STREQUAL "dm")
    set(TEST_PROCESSORS ${CUDAQ_TEST_OMP_SLOTS})
  endif()
  if (${NVQIR_BACKEND} STREQUAL "stim")
    target_include_directories(${TEST_EXE_NAME} PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime/nvqir/stim
      ${CMAKE_SOURCE_DIR}/tpls/Stim/src)
  endif()
  if ("${TEST_LABELS}" STREQUAL "")
    cudaq_gtest_discover_tests(${TEST_EXE_NAME} PROPERTIES PROCESSORS ${TEST_PROCESSORS} DISCOVERY_TIMEOUT 120)
  else()
    cudaq_gtest_discover_tests(${TEST_EXE_NAME} PROPERTIES LABELS "${TEST_LABELS}" PROCESSORS ${TEST_PROCESSORS} RESOURCE_LOCK "gpu" DISCOVERY_TIMEOUT 120)
  endif()
endmacro()

# We will always have the QPP backend, create a tester for it
create_tests_with_backend(qpp "")
create_tests_with_backend(dm "")
create_tests_with_backend(stim "")

if (cuStateVec_FOUND)
  find_program(NVIDIA_SMI "nvidia-smi")
  if(${NVIDIA_SMI} STREQUAL "NVIDIA_SMI-NOTFOUND")
    # libcustatevec.so has linkage to libnvdia-ml.so.1, which is part of NVIDIA driver.
    # On a build system without NVIDIA GPUs, this lib cannot be resolved;
    # hence, linking these test executables will fail.
    # On these CPU-only build systems, we directly link the nvidia-ml-dev library, which provides
    # stub symbols for the actual driver library, to these test executables.
    # Running these tests will ultimately requires a system with GPUs, i.e., the proper nvidia-ml lib
    # will be loaded by the runtime linker/loader.
    find_package(CUDAToolkit REQUIRED)
    # make sure CUDA_TOOLKIT_ROOT_DIR is set
    if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
      set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_BIN_DIR}/..)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      set(NVIDIA_ML_PATH ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(NVIDIA_ML_PATH ${CUDA_TOOLKIT_ROOT_DIR}/targets/sbsa-linux/lib/stubs)
    else()
      message(FATAL_ERROR "Neither x86_64 nor aarch64 was detected." )
    endif()
    # Find nvidia-ml manually
    find_library(NVIDIA_ML NAMES nvidia-ml PATHS ${NVIDIA_ML_PATH})
    message(STATUS "NVIDIA ML lib: ${NVIDIA_ML}")
    # Inject the file during build and remove it after build (install)
    file(CREATE_LINK ${NVIDIA_ML} ${CMAKE_CURRENT_BINARY_DIR}/libnvidia-ml.so.1 SYMBOLIC)
    link_directories(${CMAKE_CURRENT_BINARY_DIR})
    install(CODE "file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/libnvidia-ml.so.1)")
  endif()

  create_tests_with_backend(custatevec-fp32 "")
  # Given that the fp32 and fp64 difference is largely inherited
  # from a dependency, we omit fp64 tests here and rely on the
  # dependency to to validate that functionality.
  # create_tests_with_backend(custatevec-fp64 "")

  add_nvqpp_executable(test_mqpu
    SOURCES ${CUDAQ_GTEST_MAIN} mqpu/mqpu_tester.cpp
    NVQPP_FLAGS --target=nvidia-legacy --target-option fp64,mqpu
    EXTRA_LINK_LIBS gtest_main
    EXTRA_DEPS cudaq-platform-mqpu nvqir-custatevec-fp64
  )
  cudaq_gtest_discover_tests(test_mqpu PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")

  # Test CUDAQ_OBSERVE_FROM_SAMPLING=ON mode.
  # (term-by-term expectation value calculation by applying change-of-basis gates then reverting them)
  set(CUSTATEVEC_OFS_SOURCES
    integration/builder_tester.cpp
    integration/deuteron_variational_tester.cpp
    integration/gradient_tester.cpp
  )
  if (CUDAQ_ENABLE_NLOPT)
    list(APPEND CUSTATEVEC_OFS_SOURCES integration/nlopt_tester.cpp)
  endif()
  add_nvqpp_executable(test_custatevec_observe_from_sampling
    SOURCES ${CUSTATEVEC_OFS_SOURCES}
    NVQPP_FLAGS --target=nvidia-legacy --target-option fp32
    EXTRA_LINK_LIBS gtest_main
    EXTRA_DEPS nvqir-custatevec-fp32
  )
  # ".." for CUDAQTestUtils.h (builder_tester.cpp/deuteron_variational_tester.cpp
  # include it), now that INCLUDE_DIRECTORIES is cleared above.
  target_include_directories(test_custatevec_observe_from_sampling PRIVATE . ..)
  # CUDAQ_SIMULATION_SCALAR_FP32 is already supplied automatically by nvq++
  # from the nvidia-legacy target's own preprocessor-defines.
  target_compile_definitions(test_custatevec_observe_from_sampling
    PRIVATE -DNVQIR_BACKEND_NAME=custatevec_fp32)
  # Run this test with "CUDAQ_OBSERVE_FROM_SAMPLING=1"
  cudaq_gtest_discover_tests(test_custatevec_observe_from_sampling TEST_SUFFIX _DirectObserve PROPERTIES ENVIRONMENT "CUDAQ_OBSERVE_FROM_SAMPLING=1" PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")

  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPU
      if (${NGPUS} GREATER_EQUAL 2)
        add_nvqpp_executable(test_mpi
          SOURCES ${CUDAQ_GTEST_MAIN} mqpu/mpi_mqpu_tester.cpp
          NVQPP_FLAGS --target=nvidia-legacy --target-option fp32,mqpu
          EXTRA_LINK_LIBS gtest_main
          EXTRA_DEPS cudaq-platform-mqpu nvqir-custatevec-fp32
        )
        configure_file("mqpu/run_mpi.sh.in"
                    "${CMAKE_BINARY_DIR}/unittests/nvqpp/run_mpi.sh" @ONLY)
        add_test(NAME MPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 bash ${CMAKE_BINARY_DIR}/unittests/nvqpp/run_mpi.sh)
        set_tests_properties(MPITest PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
      endif()
    endif()

    # NOTE: test_gpu_get_state mixes a CUDA (.cu) source with the shared
    # CXX gtest main. CMake picks CUDA as this target's LINKER_LANGUAGE
    # whenever any CUDA source is present, so nvcc (not nvq++) drives the
    # final link here regardless of CMAKE_CXX_COMPILER — nvq++'s automatic
    # link-line assembly never applies, hence this target keeps explicit
    # target_link_libraries() rather than add_nvqpp_executable(). The .cu
    # file's __qpu__ kernel is never seen by cudaq-quake (nvcc compiles it,
    # not nvq++), so it still needs the CUDAQ_LIBRARY_MODE inline
    # qubit_qis.h path removed at the top of this file for every other
    # target in this subtree.
    add_executable(test_gpu_get_state ${CUDAQ_GTEST_MAIN} gpu/get_state_tester.cu)
    # main.cpp in this target is still compiled by nvq++ (CMAKE_CXX_COMPILER
    # for this directory), so it still needs the nvq++ toolchain built first.
    add_dependencies(test_gpu_get_state nvq++)
    target_compile_definitions(test_gpu_get_state PRIVATE CUDAQ_LIBRARY_MODE)
    # Need to force the link to nvqir-qpp here if gcc.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(test_gpu_get_state PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
    endif()
    target_compile_options(test_gpu_get_state PRIVATE -Wno-suggest-override -Wno-attributes)
    # ".." for CUDAQTestUtils.h (gpu/get_state_tester.cu includes it), now
    # that INCLUDE_DIRECTORIES is cleared above.
    target_include_directories(test_gpu_get_state PRIVATE . .. ${CMAKE_SOURCE_DIR}/tpls/json/single_include_fwd)
    target_link_libraries(test_gpu_get_state
      PRIVATE
      cudaq
      cudaq-builder
      cudaq-platform-mqpu
      cudaq-mlir-runtime
      nvqir-custatevec-fp64
      gtest_main)
      cudaq_gtest_discover_tests(test_gpu_get_state PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
  endif()
endif()

if(TARGET nvqir-tensornet)
  message(STATUS "Building cutensornet backend tests.")
  create_tests_with_backend(tensornet "")
  create_tests_with_backend(tensornet-mps "")
  create_tests_with_backend(tensornet-fp32 "")
  create_tests_with_backend(tensornet-mps-fp32 "")
  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPUs
      if (${NGPUS} GREATER_EQUAL 2)
        message(STATUS "Building cutensornet MPI tests.")
        add_nvqpp_executable(test_tensornet_mpi
          SOURCES mpi/tensornet_mpi_tester.cpp
          NVQPP_FLAGS --target=tensornet
          EXTRA_LINK_LIBS gtest
          EXTRA_DEPS nvqir-tensornet
        )
        add_test(NAME TensornetMPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/nvqpp/test_tensornet_mpi)
        set_tests_properties(TensornetMPITest PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
      endif() # NGPUS
    endif() # NVIDIA_SMI
  endif() # MPI_CXX_FOUND

  # Test CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON mode (on a few test cases that have cudaq::observe)
  add_nvqpp_executable(test_tensornet_observe_path_reuse
    SOURCES
      integration/builder_tester.cpp
      integration/deuteron_variational_tester.cpp
      integration/observe_result_tester.cpp
      integration/noise_tester.cpp # This test contains noisy observe test cases.
    NVQPP_FLAGS --target=tensornet
    EXTRA_LINK_LIBS gtest_main
    EXTRA_DEPS nvqir-tensornet
  )
  # ".." for CUDAQTestUtils.h (builder_tester.cpp/deuteron_variational_tester.cpp/
  # observe_result_tester.cpp/noise_tester.cpp include it), now that
  # INCLUDE_DIRECTORIES is cleared above.
  target_include_directories(test_tensornet_observe_path_reuse PRIVATE . ..)
  target_compile_definitions(test_tensornet_observe_path_reuse
                             PRIVATE -DNVQIR_BACKEND_NAME=tensornet)
  target_compile_definitions(test_tensornet_observe_path_reuse PRIVATE -DCUDAQ_BACKEND_TENSORNET)
  # Run this test with "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE"
  cudaq_gtest_discover_tests(test_tensornet_observe_path_reuse TEST_SUFFIX _PathReuse PROPERTIES ENVIRONMENT "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON" PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
endif()

# build the test qudit execution manager
add_subdirectory(qudit)
add_nvqpp_executable(test_qudit
  SOURCES ${CUDAQ_GTEST_MAIN} qudit/SimpleQuditTester.cpp
  # --target=qpp-cpu pins the qpp simulator explicitly: nvq++ defaults to the
  # "nvidia" target when a GPU + cuStateVec are present, which would silently
  # change the backend this test runs against.
  NVQPP_FLAGS --target=qpp-cpu --execution-manager=qudit
  EXTRA_LINK_LIBS gtest_main
  EXTRA_DEPS cudaq-em-qudit
)
cudaq_gtest_discover_tests(test_qudit PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)

# build the test photonics execution manager
add_nvqpp_executable(test_photonics
  SOURCES ${CUDAQ_GTEST_MAIN} photonics/PhotonicsTester.cpp
  # See test_qudit above for why --target=qpp-cpu is explicit here.
  NVQPP_FLAGS --target=orca-photonics
  EXTRA_LINK_LIBS gtest_main
  EXTRA_DEPS cudaq-em-photonics
)
cudaq_gtest_discover_tests(test_photonics PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)

# Create an executable for MPI UnitTests
# (only if MPI was found, i.e., the builtin plugin is available)
if (MPI_CXX_FOUND)
  set(CUDAQ_MPI_TEST_SOURCES
    mpi/mpi_tester.cpp
  )
  add_nvqpp_executable(test_mpi_plugin
    SOURCES ${CUDAQ_MPI_TEST_SOURCES}
    NVQPP_FLAGS --target=qpp-cpu
    EXTRA_LINK_LIBS gtest
  )
  set(NUM_PROCS 4)
  target_compile_definitions(test_mpi_plugin PRIVATE -DNUM_PROCS=${NUM_PROCS})
  target_link_options(test_mpi_plugin PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  # Check if `--allow-run-as-root` is supported (OpenMPI)
  # Note: MPICH doesn't need `--allow-run-as-root`.
  execute_process(COMMAND ${MPIEXEC} --allow-run-as-root -np ${NUM_PROCS} hostname ERROR_VARIABLE CHECK_ALLOW_RUN_AS_ROOT_RESULTS ERROR_STRIP_TRAILING_WHITESPACE OUTPUT_QUIET)
  if ("${CHECK_ALLOW_RUN_AS_ROOT_RESULTS}" STREQUAL "")
    set(MPI_EXEC_CMD_ARGS "--allow-run-as-root")
  endif()

  add_test(NAME MPIApiTest COMMAND ${MPIEXEC} ${MPI_EXEC_CMD_ARGS} -np ${NUM_PROCS} ${CMAKE_BINARY_DIR}/unittests/nvqpp/test_mpi_plugin)
  math(EXPR MPI_API_TEST_SLOTS "${NUM_PROCS} * ${CUDAQ_TEST_OMP_SLOTS}")
  set_tests_properties(MPIApiTest PROPERTIES PROCESSORS ${MPI_API_TEST_SLOTS})
endif()

if (CUDA_FOUND AND TARGET nvqir-dynamics)

  # Multi-QPU evolve_async test
  add_nvqpp_executable(test_evolve_async
    SOURCES mqpu/dynamics_async_tester.cpp
    NVQPP_FLAGS --target=dynamics
    EXTRA_LINK_LIBS
      cuDensityMat::cuDensityMat
      CUDA::cudart
      CUDA::cublas
      cuTensor::cuTensor
      CUDA::cusparse
      cuTensorNet::cuTensorNet
      gtest_main
    EXTRA_DEPS nvqir-dynamics cudaq-platform-mqpu
  )
  # CUDAQ_ANALOG_TARGET is already supplied automatically by nvq++ from the
  # dynamics target's own preprocessor-defines.
  # Count the number of GPUs
  find_program(NVIDIA_SMI "nvidia-smi")
  if(NVIDIA_SMI)
    execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
    # Only add this test if we have more than 1 GPU
    if (${NGPUS} GREATER_EQUAL 2)
      cudaq_gtest_discover_tests(test_evolve_async PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")

      # Multi-GPU batching test: this test needs MPI as well
      if (MPI_CXX_FOUND)
        add_nvqpp_executable(test_dynamics_mpi_batching
          SOURCES mqpu/dynamics_mpi_batching_tester.cpp
          NVQPP_FLAGS --target=dynamics
          EXTRA_LINK_LIBS
            cuDensityMat::cuDensityMat
            CUDA::cudart
            CUDA::cublas
            cuTensor::cuTensor
            CUDA::cusparse
            cuTensorNet::cuTensorNet
            gtest_main
          EXTRA_DEPS nvqir-dynamics cudaq-platform-mqpu
        )
        # CUDAQ_ANALOG_TARGET is already supplied automatically by nvq++
        # from the dynamics target's own preprocessor-defines.

        add_test(NAME DynamicsMPIBatchingTest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/nvqpp/test_dynamics_mpi_batching)
        set_tests_properties(DynamicsMPIBatchingTest PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
      endif()
    endif()
  endif()
endif()

#add_subdirectory(backends)

if (CUDAQ_ENABLE_PYTHON)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "find_package(Python) not run?")
  endif()

  execute_process(COMMAND ${Python_EXECUTABLE} -c "import openfermionpyscf"
    OUTPUT_VARIABLE PYSCF_output
    ERROR_VARIABLE  PYSCF_error
    RESULT_VARIABLE PYSCF_result)

  if(NOT ${PYSCF_result} EQUAL 0)
    message(STATUS "OpenFermion PySCF not available for chemistry tests.")
    return()
  endif()

  if(NOT TARGET cudaq-pyscf)
    message(STATUS "cudaq-pyscf plugin not enabled; skipping chemistry tests.")
    return()
  endif()

  message(STATUS "OpenFermion PySCF found, enabling chemistry tests.")
  add_nvqpp_executable(test_domains
    SOURCES ${CUDAQ_GTEST_MAIN} domains/ChemistryTester.cpp
    # --target=qpp-cpu pins the qpp simulator explicitly: nvq++ defaults to
    # the "nvidia" target when a GPU + cuStateVec are present, which would
    # silently change the backend this test runs against.
    NVQPP_FLAGS --target=qpp-cpu
    EXTRA_LINK_LIBS cudaq-chemistry cudaq-pyscf gtest_main
  )
  # CUDAQ_SIMULATION_SCALAR_FP64 is already supplied automatically by nvq++
  # from the qpp-cpu target's own preprocessor-defines.
  target_compile_definitions(test_domains PRIVATE -DNVQIR_BACKEND_NAME=qpp)
  target_include_directories(test_domains PRIVATE . ..)
  # Embedded Python links to system framework, not venv, so needs explicit sitelib.
  set_property(TARGET test_domains PROPERTY ENVIRONMENT
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${Python_SITELIB}")
  cudaq_gtest_discover_tests(test_domains
    TEST_SUFFIX _Sampling PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${Python_SITELIB}" PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)
endif()
