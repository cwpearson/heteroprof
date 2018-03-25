#include "cuda/cupti/callback/cuda_set_device.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CudaSetDevice::CudaSetDevice(const tid_t callingThread,
                             const CUpti_CallbackData *cbdata, const int device)
    : Api(callingThread, cbdata), device_(device) {}

json CudaSetDevice::to_json() const {
  json j = Api::to_json();
  j["device"] = device_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda