#ifndef CPROF_ACTIVITY_TRANSFER_HPP
#define CPROF_ACTIVITY_TRANSFER_HPP

#include <cassert>
#include <chrono>
#include <map>
#include <string>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "util_cupti.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace activity {

class Transfer {
  using json = nlohmann::json;

public:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  typedef std::chrono::nanoseconds duration_t;

  enum class Kind { CUPTI_MEMCPY, INVALID };

  Transfer();
  explicit Transfer(const CUpti_ActivityMemcpy *record);

  size_t start_ns() const;
  size_t dur_ns() const;

  json to_json() const;
  std::string to_json_string() const;

private:
  // General fields
  size_t bytes_;
  duration_t duration_;
  time_point_t start_;
  std::map<std::string, std::string> kv_;

  // CUDA-specific fields (FIXME: move to derived class)
  uint32_t cudaDeviceId_;
  Kind kind_;
  CuptiActivityMemcpyKind cudaMemcpyKind_;
  CuptiActivityMemoryKind srcKind_;
  CuptiActivityMemoryKind dstKind_;
  uint32_t contextId_;
  uint32_t correlationId_;
  uint8_t flags_;
  uint32_t runtimeCorrelationId_;
  uint32_t streamId_;
};

} // namespace activity
} // namespace cupti
} // namespace cuda
} // namespace model

#endif