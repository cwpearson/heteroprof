#include "util_cudnn.hpp"

#include <cassert>

#include "profiler.hpp"

size_t tensorSize(const cudnnTensorDescriptor_t tensorDesc) {
  size_t size;
  CUDNN_CHECK(cudnnGetTensorSizeInBytes(tensorDesc, &size), profiler::log());
  return size;
}