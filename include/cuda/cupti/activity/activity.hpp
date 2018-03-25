#ifndef CUDA_CUPTI_ACTIVITY_ACTIVITY_HPP
#define CUDA_CUPTI_ACTIVITY_ACTIVITY_HPP

#include <cassert>
#include <chrono>
#include <map>
#include <string>

#include <cupti.h>

#include "nlohmann/json.hpp"

namespace cuda {
namespace cupti {
namespace activity {

class Activity {
  using json = nlohmann::json;

public:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  typedef std::chrono::nanoseconds duration_t;

  uint64_t start_ns() const;
  uint64_t dur_ns() const;

  virtual json to_json() const = 0;
  std::string to_json_string() const;
  virtual std::string hprof_kind() const { return "cupti_activity"; }

protected:
  duration_t duration_;
  time_point_t start_;
};

} // namespace activity
} // namespace cupti
} // namespace cuda

#endif