#ifndef CPROF_ACTIVITY_TRANSFER_HPP
#define CPROF_ACTIVITY_TRANSFER_HPP

#include <cassert>
#include <chrono>
#include <map>
#include <string>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/activity/activity.hpp"
#include "cuda/cupti/util.hpp"

namespace cuda {
namespace cupti {
namespace activity {

class Transfer : public Activity {
  using json = nlohmann::json;

public:
  enum class Kind { CUPTI_MEMCPY, INVALID };

  Transfer();
  explicit Transfer(const CUpti_ActivityMemcpy *record);

  virtual json to_json() const override;

private:
  // General fields
  size_t bytes_;

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

#endif