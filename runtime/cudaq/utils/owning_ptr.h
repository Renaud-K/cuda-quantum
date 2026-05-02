/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <memory>

namespace cudaq {
/// @brief A deleter whose `operator()` is declared by the primary template
/// and defined out-of-line per type in a translation unit where @c T is
/// complete.  Use together with @c owning_ptr to hold a @c std::unique_ptr<T>
/// in a header that only forward-declares @c T.
///
/// For each @c T to be wrapped, declare the explicit specialization in a
/// header visible to every TU that destroys an @c owning_ptr<T>, and define
/// its @c operator() in a TU where @c T is complete:
/// @code
/// // some_fwd_header.h, alongside `class MyType;`
/// template <> struct opaque_deleter<MyType> {
///   void operator()(MyType *) const;
/// };
///
/// // MyType.cpp
/// void opaque_deleter<MyType>::operator()(MyType *p) const { delete p; }
/// @endcode
///
/// The specialization declaration must be visible at every point of
/// instantiation; otherwise the deleter type is incomplete and
/// @c std::unique_ptr's SFINAE-constrained constructors are silently
/// dropped from the overload set.
template <typename T>
struct opaque_deleter {
  void operator()(T *p) const;
};

/// @brief A @c std::unique_ptr<T> whose destruction is performed by an
/// out-of-line @c opaque_deleter<T> specialization.
template <typename T>
using owning_ptr = std::unique_ptr<T, opaque_deleter<T>>;

} // namespace cudaq
