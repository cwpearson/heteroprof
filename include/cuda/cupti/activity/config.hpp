#ifndef CUDA_CUPTI_ACTIVITY_CONFIG_HPP
#define CUDA_CUPTI_ACTIVITY_CONFIG_HPP

class Profiler;

#include <cupti.h>

#include "cuda/cupti/activity/handler.hpp"
#include "profiler.hpp"

namespace cuda {
namespace cupti {
namespace activity {
namespace config {

void set_profiler(Profiler &p);

void set_buffer_size(const size_t n);
void set_device_buffer_size(const size_t bytes);
size_t buffer_size();
size_t *attr_device_buffer_size();
size_t *attr_device_buffer_pool_limit();
size_t *attr_value_size(const CUpti_ActivityAttribute &attr);

} // namespace config
} // namespace activity
} // namespace cupti
} // namespace cuda

typedef void (*BufReqFun)(uint8_t **buffer, size_t *size,
                          size_t *maxNumRecords);

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize);
void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords);

#endif