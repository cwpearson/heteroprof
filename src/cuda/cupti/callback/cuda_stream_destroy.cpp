#include "cuda/cupti/callback/cuda_stream_destroy.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CudaStreamDestory::CudaStreamDestory(const tid_t callingThread,
                                     const CUpti_CallbackData *cbdata,
                                     const cudaStream_t stream);
    : Api(callingThread, cbdata), stream_(stream) {
}

    json CudaStreamDestory::to_json() const {
      json j = Api::to_json();
      j["stream"] = stream_;
      return j;
    }

    } // namespace callback
    } // namespace cupti
    } // namespace cuda
