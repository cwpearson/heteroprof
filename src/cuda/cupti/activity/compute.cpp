#include <nlohmann/json.hpp>

#include "cuda/cupti/activity/compute.hpp"

using json = nlohmann::json;

namespace cuda {
namespace cupti {
namespace activity {

std::string to_string(const Compute::Kind &kind) {
  switch (kind) {
  case Compute::Kind::CUPTI_KERNEL3:
    return "cupti_kernel3";
  case Compute::Kind::INVALID:
    return "invalid";
  default:
    assert(0 && "unxpected Compute::Kind");
  }
}

Compute::Compute() : kind_(Compute::Kind::INVALID) {}

Compute::Compute(const CUpti_ActivityKernel3 *record) : Compute() {
  assert(record && "invalid record");

  kind_ = Kind::CUPTI_KERNEL3;

  if (record->completed == CUPTI_TIMESTAMP_UNKNOWN) {
    completed_ = time_point_t(std::chrono::nanoseconds(0));
  } else {
    completed_ = time_point_t(std::chrono::nanoseconds(record->completed));
  }
  start_ = time_point_t(std::chrono::nanoseconds(record->start));
  duration_ = std::chrono::nanoseconds(record->end) -
              std::chrono::nanoseconds(record->start);
  cudaDeviceId_ = record->deviceId;
  contextId_ = record->contextId;
  correlationId_ = record->correlationId;
  streamId_ = record->streamId;
  name_ = record->name;
}

uint64_t Compute::completed_ns() const {
  auto cNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(completed_);
  auto cEpoch = cNs.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(cEpoch);
  return value.count();
}

json Compute::to_json() const {
  json j = Activity::to_json();
  j["cuda_device_id"] = cudaDeviceId_;
  j["kind"] = to_string(kind_);
  j["name"] = name_;
  j["completed"] = completed_ns();
  j["stream_id"] = streamId_;
  j["correlation_id"] = correlationId_;
  return j;
}

} // namespace activity
} // namespace cupti
} // namespace cuda
