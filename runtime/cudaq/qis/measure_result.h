/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include "cudaq/host_config.h"
#include <utility>
#include <vector>

namespace cudaq {
using SpinMeasureResult = std::pair<double, sample_result>;

extern "C" {
bool __nvqpp__MeasureResultBoolConversion(int);
}

#ifdef CUDAQ_LIBRARY_MODE

/// In library mode, we model the return type of a qubit measurement result via
/// the measure_result type. This allows us to keep track of when the result is
/// implicitly cast to a boolean (likely in the case of conditional feedback),
/// and affect the simulation accordingly.
class measure_result {
private:
  /// The intrinsic measurement result
  int result = 0;

  /// Unique integer for measure result identification
  [[maybe_unused]] std::size_t uniqueId = 0;

public:
  measure_result(int res, std::size_t id) : result(res), uniqueId(id) {}
  measure_result(int res) : result(res) {}

  operator int() const { return result; }
  operator bool() const { return __nvqpp__MeasureResultBoolConversion(result); }

  static std::vector<bool>
  to_bool_vector(const std::vector<measure_result> &results) {
    std::vector<bool> boolResults;
    boolResults.reserve(results.size());
    for (const auto &res : results)
      boolResults.push_back(static_cast<bool>(res));
    return boolResults;
  }
};
#else
/// When compiling with MLIR, we default to a boolean.
using measure_result = bool;
#endif

} // namespace cudaq
