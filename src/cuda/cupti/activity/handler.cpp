#include <string>

#include "cuda/cupti/activity/activity.hpp"
#include "cuda/cupti/activity/compute.hpp"
#include "cuda/cupti/activity/handler.hpp"
#include "cuda/cupti/activity/transfer.hpp"

using namespace cuda::cupti::activity;

typedef std::shared_ptr<Activity> ActivityRef;

namespace cuda {
namespace cupti {
namespace activity {

void handler(const CUpti_Activity *record, Profiler &profiler) {

  ActivityRef activity(nullptr);

  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL: {
    auto activityCast = reinterpret_cast<const CUpti_ActivityKernel3 *>(record);

    activity = std::make_shared<Compute>(activityCast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    auto activityCast = reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
    activity = std::make_shared<Transfer>(activityCast);
    break;
  }
  default: { return; }
  };

  assert(activity && "Expected activity to be set");
  profiler.record(activity->to_json());
}

} // namespace activity
} // namespace cupti
} // namespace cuda
