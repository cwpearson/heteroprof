#include <string>

#include "model/cupti_activity/compute.hpp"
#include "model/cupti_activity/transfer.hpp"

#include "cupti_activity_handler.hpp"
#include "profiler.hpp"

//
// KERNEL
//
static void handleKernel(const CUpti_ActivityKernel3 *record) {
  assert(record);
  auto compute = model::activity::Compute(record);
  profiler::record_json(compute.to_json());
}

//
// MEMCPY
//
static void handleMemcpy(const CUpti_ActivityMemcpy *record) {
  assert(record);
  auto transfer = model::activity::Transfer(record);
  profiler::record_json(transfer.to_json());
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
  default: {
    //   auto activity_cast = record;
    break;
  }
  };
}