/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace cudaq {
// Split a string on the given delimiter
template <class Op>
void split(const std::string_view s, char delim, Op op) {
  std::stringstream ss(s.data());
  for (std::string item; std::getline(ss, item, delim);) {
    *op++ = item;
  }
}

// Split a string on the given delimiter
inline std::vector<std::string> split(const std::string_view s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

// Trim a string on the left
inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
  ltrim(s);
  rtrim(s);
}
} // namespace cudaq
