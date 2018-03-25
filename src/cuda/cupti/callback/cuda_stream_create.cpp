#include "cuda/cupti/callback/cuda_stream_create.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CudaStreamCreate::CudaStreamCreate(const tid_t callingThread,
                                   const CUpti_CallbackData *cbdata)
    : Api(callingThread, cbdata) {}

void CudaStreamCreate::set_stream(const cudaStream_t stream) {
  stream_ = stream;
}

json CudaStreamCreate::to_json() const {
  json j = Api::to_json();
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
