#ifndef CUDA_CUPTI_CALLBACK_CONFIG_HPP
#define CUDA_CUPTI_CALLBACK_CONFIG_HPP

class Profiler;

#include <cupti.h>

#include "profiler.hpp"

namespace cuda {
namespace cupti {
namespace callback {
namespace config {

/* !brief set Profiler instance used by CUPTI callbacks
 */
void set_profiler(Profiler &p);
Profiler &profiler();

void set_record_nested_runtime(const bool enable);

} // namespace config
} // namespace callback
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