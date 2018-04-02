#include <nlohmann/json.hpp>

#include "cuda/cupti/activity/activity.hpp"

using json = nlohmann::json;

namespace cuda {
namespace cupti {
namespace activity {

uint64_t Activity::start_ns() const {
  auto startNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(start_);
  auto startEpoch = startNs.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(startEpoch);
  return value.count();
}

uint64_t Activity::dur_ns() const {
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_);
  return value.count();
}

json Activity::to_json() const {
  json j;
  j["hprof_kind"] = hprof_kind();
  j["start"] = start_ns();
  j["duration"] = dur_ns();
  return j;
}

std::string Activity::to_json_string() const { return to_json().dump(); }

} // namespace activity
} // namespace cupti
} // namespace cuda
