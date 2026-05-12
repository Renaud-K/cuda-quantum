#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace cudaq {
namespace details {

/// @brief The `kernel_builder_base` provides a base type for the templated
/// kernel builder so that we can get a single handle on an instance within the
/// runtime.
class kernel_builder_base {
public:
  virtual std::string to_quake() const = 0;
  virtual void jitCode(std::vector<std::string> extraLibPaths = {}) = 0;
  virtual ~kernel_builder_base() = default;

  /// @brief Write the kernel_builder to the given output stream. This outputs
  /// the Quake representation.
  friend std::ostream &operator<<(std::ostream &stream,
                                  const kernel_builder_base &builder);
};

} // namespace details
} // namespace cudaq
