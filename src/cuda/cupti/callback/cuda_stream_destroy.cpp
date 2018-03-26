#include "cuda/cupti/callback/cuda_stream_destroy.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

CudaStreamDestroy::CudaStreamDestroy(const Api &api, const cudaStream_t stream)
    : Api(api), stream_(reinterpret_cast<uintptr_t>(stream)) {
  static_assert(sizeof(uintptr_t) == sizeof(stream), "uh oh");
}

json CudaStreamDestroy::to_json() const {
  json j = Api::to_json();
  j["stream"] = stream_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
