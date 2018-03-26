#include "cuda/cupti/callback/device_alloc.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

DeviceAlloc::DeviceAlloc(const tid_t callingThread,
                         const CUpti_CallbackData *cbdata, const size_t size)
    : Api(callingThread, cbdata), ptr_(0), size_(size) {}

void DeviceAlloc::set_ptr(const void *ptr) {
  ptr_ = reinterpret_cast<uintptr_t>(ptr);
}

json DeviceAlloc::to_json() const {
  json j = Api::to_json();
  j["ptr"] = devPtr_;
  j["size"] = size_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
