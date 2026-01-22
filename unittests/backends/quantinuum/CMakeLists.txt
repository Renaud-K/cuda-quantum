# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_quantinuum 
  SOURCES QuantinuumTester.cpp QuantinuumNGTester.cpp
  BACKEND quantinuum
  )

add_backend_unittest_executable(test_quantinuum_emulation 
  SOURCES QuantinuumEmulationTester.cpp 
  BACKEND quantinuum
  )

configure_file("QuantinuumStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/quantinuum/QuantinuumStartServerAndTest.sh" @ONLY)
add_test(NAME quantinuum-tests COMMAND bash QuantinuumStartServerAndTest.sh test_quantinuum WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/quantinuum/)
add_test(NAME quantinuum-emulation-tests COMMAND bash QuantinuumStartServerAndTest.sh test_quantinuum_emulation WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/quantinuum/)

# Test for Nexus helper functions
add_backend_unittest_executable(test_quantinuum_util 
  SOURCES QuantinuumUtilTester.cpp
  
  INCLUDES
  ../.. 
  ${CMAKE_SOURCE_DIR}/runtime/cudaq/platform/default/rest/helpers/quantinuum

  LINK_LIBS
  cudaq
  cudaq-common 
  cudaq-operator
  gtest_main
)
gtest_discover_tests(test_quantinuum_util)
