#include <string>

#include "cupti_activity_handler.hpp"

#include "model/cuda/cupti/activity/compute.hpp"
#include "model/cuda/cupti/activity/transfer.hpp"

#include "profiler.hpp"

using namespace model::cuda::cupti::activity;

//
// KERNEL
//
static void handleKernel(const CUpti_ActivityKernel3 *record) {
  assert(record);
  auto compute = Compute(record);
  profiler::record(compute.to_json());
}

//
// MEMCPY
//
static void handleMemcpy(const CUpti_ActivityMemcpy *record) {
  assert(record);
  auto transfer = Transfer(record);
  profiler::record(transfer.to_json());
}

void activityHander(const CUpti_Activity *record) {

  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL: {
    auto activity_cast =
        reinterpret_cast<const CUpti_ActivityKernel3 *>(record);
    handleKernel(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    auto activity_cast = reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
    handleMemcpy(activity_cast);
    break;
  }
  default: { break; }
  };
}