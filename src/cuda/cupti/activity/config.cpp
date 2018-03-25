#include "cuda/cupti/activity/config.hpp"
#include "cuda/cupti/activity/handler.hpp"
#include "cuda/cupti/util.hpp"

namespace cuda {
namespace cupti {
namespace activity {
namespace config {

Profiler *profiler_ = nullptr;
void set_profiler(Profiler &p) { profiler_ = &p; }
Profiler &profiler() {
  assert(profiler_);
  return *profiler_;
}

size_t localBufferSize = 1 * 1024 * 1024;
size_t attrDeviceBufferSize = 8 * 1024 * 1024;
size_t attrDeviceBufferPoolLimit = 2;
size_t attrValueSize_size_t = sizeof(size_t);

size_t *attr_value_size(const CUpti_ActivityAttribute &attr) {
  return &attrValueSize_size_t; // all attributes are size_t as of CUDA 9.1
}

void set_local_buffer_size(const size_t bytes) { localBufferSize = bytes; }
void set_device_buffer_size(const size_t bytes) {
  attrDeviceBufferSize = bytes;
}
void set_device_buffer_pool_limit(const size_t npools) {
  attrDeviceBufferPoolLimit = npools;
}
size_t local_buffer_size() { return localBufferSize; }
size_t align_size() { return 8; }
size_t *attr_device_buffer_size() { return &attrDeviceBufferSize; }
size_t *attr_device_buffer_pool_limit() { return &attrDeviceBufferPoolLimit; }

} // namespace config
} // namespace activity
} // namespace cupti
} // namespace cuda

#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

using namespace cuda::cupti::activity::config;
using namespace cuda::cupti::activity;

void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {

  uint8_t *rawBuffer;

  *size = local_buffer_size();
  rawBuffer = (uint8_t *)malloc(*size + align_size());

  *buffer = ALIGN_BUFFER(rawBuffer, align_size());
  *maxNumRecords = 0; // as many records as possible

  if (*buffer == NULL) {
    profiler().log() << "ERROR: out of memory" << std::endl;
    exit(-1);
  }
}

static void threadFunc(uint8_t *localBuffer, size_t validSize) {

  auto start = std::chrono::high_resolution_clock::now();

  CUpti_Activity *record = NULL;
  if (validSize > 0) {
    do {
      auto err = cuptiActivityGetNextRecord(localBuffer, validSize, &record);
      if (err == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }

      CUPTI_CHECK(err, profiler().log());
      handler(record, profiler());
    } while (true);
  }

  auto end = std::chrono::high_resolution_clock::now();
}

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {

  threadFunc(buffer, validSize);
}