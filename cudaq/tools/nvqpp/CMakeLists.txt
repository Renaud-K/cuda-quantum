# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_IS_APPLE "false")
if (APPLE) 
  set(CUDAQ_IS_APPLE "true")
endif()

set(CUDAQ_DEVICE_CALL_RUNTIME_LINK_LIBS "")
if (CUDAQ_ENABLE_REALTIME)
  set(CUDAQ_DEVICE_CALL_RUNTIME_LINK_LIBS "-lcudaq-device-call-runtime")
endif()

add_custom_target(nvq++)
add_dependencies(nvq++
  # Toolchain binaries invoked by the nvq++ driver script.
  cudaq-quake
  cudaq-opt
  cudaq-translate
  cudaq-target-conf
  fixup-linkage
  # Runtime shared libraries that nvq++ always links into the final
  # executable (see LINKLIBS/NVQIR_LIBS in nvq++.in). CMAKE_CXX_COMPILER=nvq++
  # alone creates no build edge for these -l flags, so callers that depend on
  # the nvq++ custom target (e.g. add_nvqpp_executable()) transitively get
  # correct build ordering without repeating this list per target.
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-operator
  cudaq-logger
  cudaq-common
  cudaq-ensmallen
  cudaq-nlopt
  cudaq-platform-default
  cudaq-em-default
  nvqir
)
configure_file("nvq++.in"
               "${CUDAQ_BINARY_DIR}/bin/nvq++" 
               FILE_PERMISSIONS
	              OWNER_READ OWNER_WRITE OWNER_EXECUTE
	              GROUP_READ GROUP_EXECUTE
	              WORLD_READ WORLD_EXECUTE
               @ONLY)

install(FILES ${CUDAQ_BINARY_DIR}/cudaq/include/nvqpp_config.h
        DESTINATION include/nvqpp
        COMPONENT Development)
install(PROGRAMS ${CUDAQ_BINARY_DIR}/bin/nvq++
        DESTINATION bin
        COMPONENT Nvqpp)
