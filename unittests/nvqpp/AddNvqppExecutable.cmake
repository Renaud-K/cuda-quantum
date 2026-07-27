# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# add_nvqpp_executable(<target>
#   SOURCES <src1> [<src2> ...]
#   [NVQPP_FLAGS <flag1> [<flag2> ...]]
#   [EXTRA_LINK_LIBS <lib1> [<lib2> ...]]
#   [EXTRA_DEPS <dep1> [<dep2> ...]]
# )
#
# Thin wrapper around add_executable() for targets compiled and linked by
# nvq++ (this subdirectory sets CMAKE_CXX_COMPILER=nvq++). Handles the parts
# that are specific to nvq++:
#
# - NVQPP_FLAGS (e.g. --target=qpp-cpu, --anyon-url=...) are applied to BOTH
#   target_compile_options() and target_link_options(). CMake invokes nvq++
#   as two separate, independent processes for a multi-source target -- once
#   per translation unit to compile, once to link -- and nvq++.in is a
#   stateless script that re-derives its target/backend selection from its
#   own argv on every invocation (defaulting to qpp-cpu, or silently
#   "nvidia" if a GPU + cuStateVec happen to be present). Passing NVQPP_FLAGS
#   only at compile time would leave the link invocation to fall back to
#   that default and link the wrong nvqir-*/cudaq-platform-*/cudaq-em-*
#   libraries -- a silent behavior regression, not a build failure. At
#   compile time, NVQPP_FLAGS also embeds NVQPP_TARGET_BACKEND_CONFIG (see
#   runtime/cudaq/host_config.h) in the TU that instantiates
#   cudaq::TargetSetter.
# - add_dependencies(<target> nvq++) so ninja builds the whole nvq++
#   toolchain (and the runtime libs it silently links via -l, see
#   cudaq/tools/nvqpp/CMakeLists.txt) before this target. Callers only need
#   EXTRA_DEPS for target-specific shared libs nvq++ doesn't cover by default
#   (e.g. nvqir-custatevec-fp32, nvqir-tensornet, libstim).
#
# This is not a gtest helper: callers pass SOURCES explicitly (including the
# shared gtest main where needed) and still call target_include_directories(),
# target_compile_definitions(), gtest_discover_tests(), etc. at the call site,
# same as before this helper existed.
#
# gtest/gtest_main get special-cased in EXTRA_LINK_LIBS below: they carry
# target_compile_options(gtest PUBLIC ...) GCC-only workarounds (-Wno-restrict,
# --param=evrp-mode=legacy, -Werror; see top-level CMakeLists.txt) that CMake
# re-applies to every consumer as a usage requirement via target_link_libraries()
# -- including targets compiled by nvq++'s clang++, where clang treats several
# of them as fatal (unknown/unused-argument errors, escalated by that trailing
# -Werror). This is a *target* property, not something inherited from an
# ancestor directory, so directory-scoped fixes (see the COMPILE_OPTIONS /
# COMPILE_DEFINITIONS handling at the top of CMakeLists.txt) can't remove it.
# Stripping it from gtest's target property directly would fix nvq++ targets
# but silently regress the (real) GCC-12 workarounds it exists for on every
# other, host g++-compiled test that still links gtest/gtest_main normally.
# Instead, link gtest's/gtest_main's raw output files and copy over just their
# include dirs, bypassing target-level usage-requirement propagation (and
# hence its COMPILE_OPTIONS) entirely.
function(add_nvqpp_executable target)
  set(multiValues SOURCES NVQPP_FLAGS EXTRA_LINK_LIBS EXTRA_DEPS)
  cmake_parse_arguments(ARG "" "" "${multiValues}" ${ARGN})

  add_executable(${target} ${ARG_SOURCES})

  if (ARG_NVQPP_FLAGS)
    target_compile_options(${target} PRIVATE ${ARG_NVQPP_FLAGS})
    target_link_options(${target} PRIVATE ${ARG_NVQPP_FLAGS})
  endif()

  add_dependencies(${target} nvq++)
  if (ARG_EXTRA_DEPS)
    add_dependencies(${target} ${ARG_EXTRA_DEPS})
  endif()

  if (ARG_EXTRA_LINK_LIBS)
    set(link_libs "")
    set(needs_gtest_files FALSE)
    foreach(lib IN LISTS ARG_EXTRA_LINK_LIBS)
      if (lib STREQUAL "gtest_main")
        set(needs_gtest_files TRUE)
        list(APPEND link_libs "$<TARGET_FILE:gtest_main>")
      elseif (lib STREQUAL "gtest")
        set(needs_gtest_files TRUE)
      else()
        list(APPEND link_libs ${lib})
      endif()
    endforeach()
    if (needs_gtest_files)
      target_include_directories(${target} SYSTEM PRIVATE
        $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>)
      # gtest_main.a (if requested above) depends on symbols in gtest.a, so it
      # must precede it on the link line.
      list(APPEND link_libs "$<TARGET_FILE:gtest>" Threads::Threads)
    endif()
    target_link_libraries(${target} PRIVATE ${link_libs})
  endif()
endfunction()
