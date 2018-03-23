#include "nlohmann/json.hpp"

#include "model/cuda/cupti/activity/transfer.hpp"
#include "util_cupti.hpp"

using json = nlohmann::json;

namespace model {
namespace cuda {
namespace cupti {
namespace activity {

std::string to_string(const Transfer::Kind &kind) {
  switch (kind) {
  case Transfer::Kind::CUPTI_MEMCPY:
    return "cupti_memcpy";
  case Transfer::Kind::INVALID:
    return "invalid";
  default:
    assert(0 && "unxpected Transfer::Kind");
  }
}

Transfer::Transfer() : kind_(Transfer::Kind::INVALID) {}

Transfer::Transfer(const CUpti_ActivityMemcpy *record) : Transfer() {

  static_assert(sizeof(uint8_t) == sizeof(record->dstKind),
                "unexpected data type for dstKind");
  static_assert(sizeof(uint8_t) == sizeof(record->srcKind),
                "unexpected data type for srcKind");
  static_assert(sizeof(uint8_t) == sizeof(record->copyKind),
                "unexpected data type for copyKind");

  assert(record && "invalid record");

  // unused record fields
  // void * reserved0

  bytes_ = record->bytes;
  duration_ = std::chrono::nanoseconds(record->end - record->start);
  start_ = time_point_t(std::chrono::nanoseconds(record->start));

  cudaDeviceId_ = record->deviceId;
  kind_ = Kind::CUPTI_MEMCPY;
  cudaMemcpyKind_ = from_cupti_activity_memcpy_kind(record->copyKind);
  srcKind_ = from_cupti_activity_memory_kind(record->srcKind);
  dstKind_ = from_cupti_activity_memory_kind(record->dstKind);
  contextId_ = record->contextId;
  correlationId_ = record->correlationId;
  flags_ = record->flags;
  runtimeCorrelationId_ = record->runtimeCorrelationId;
  streamId_ = record->streamId;
}

uint64_t Transfer::start_ns() const {
  auto startNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(start_);
  auto startEpoch = startNs.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(startEpoch);
  return value.count();
}

uint64_t Transfer::dur_ns() const {
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_);
  return value.count();
}

json Transfer::to_json() const {
  json j;
  j["transfer"]["cuda_device_id"] = cudaDeviceId_;
  j["transfer"]["kind"] = to_string(kind_);
  j["transfer"]["cuda_memcpy_kind"] = to_string(cudaMemcpyKind_);
  j["transfer"]["src_kind"] = to_string(srcKind_);
  j["transfer"]["dst_kind"] = to_string(dstKind_);
  j["transfer"]["start"] = start_ns();
  j["transfer"]["dur"] = dur_ns();
  j["transfer"]["stream_id"] = streamId_;
  j["transfer"]["correlation_id"] = correlationId_;
  j["transfer"]["runtime_correlation_id"] = runtimeCorrelationId_;
  j["transfer"]["kv"] = json(kv_);
  return j;
}

std::string Transfer::to_json_string() const { return to_json().dump(); }

} // namespace activity
} // namespace cupti
} // namespace cuda
} // namespace model
