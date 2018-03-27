#include "nlohmann/json.hpp"

#include "cuda/cupti/activity/transfer.hpp"

using json = nlohmann::json;

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

  start_ = time_point_t(std::chrono::nanoseconds(record->start));
  duration_ = std::chrono::nanoseconds(record->end) -
              std::chrono::nanoseconds(record->start);
}

json Transfer::to_json() const {
  json j = Activity::to_json();
  j["cuda_device_id"] = cudaDeviceId_;
  j["kind"] = to_string(kind_);
  j["cuda_memcpy_kind"] = to_string(cudaMemcpyKind_);
  j["src_kind"] = to_string(srcKind_);
  j["dst_kind"] = to_string(dstKind_);
  j["stream_id"] = streamId_;
  j["correlation_id"] = correlationId_;
  j["runtime_correlation_id"] = runtimeCorrelationId_;
  return j;
}

} // namespace activity
} // namespace cupti
} // namespace cuda
