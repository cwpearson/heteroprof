#include "model/cuda/cupti/callback/cuda_malloc.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = model::sys::tid_t;

CudaMalloc::CudaMalloc(const tid_t callingThread,
                       const CUpti_CallbackData *cbdata, const size_t size)
    : Api(callingThread, cbdata), devPtr_(0), size_(size) {}

void CudaMalloc::set_devptr(const void *const *devPtr) {
  devPtr_ = reinterpret_cast<uintptr_t>(*devPtr);
}

json CudaMalloc::to_json() const {
  json j = Api::to_json();
  j["dev_ptr"] = devPtr_;
  j["size"] = size_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model