#include "cuda/cupti/callback/host_alloc.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

HostAlloc::HostAlloc(const tid_t callingThread,
                     const CUpti_CallbackData *cbdata, const size_t bytesize,
                     const unsigned int rtFlags, const unsigned int drFlags)
    : Api(callingThread, cbdata), size_(bytesize), ptr_(0) {
  portable_ = drFlags & CU_MEMHOSTALLOC_PORTABLE;
  devicemap_ = drFlags & CU_MEMHOSTALLOC_DEVICEMAP;
  writecombined_ = drFlags & CU_MEMHOSTALLOC_WRITECOMBINED;

  portable_ = rtFlags & CU_MEMHOSTALLOC_PORTABLE;
  devicemap_ = rtFlags & CU_MEMHOSTALLOC_DEVICEMAP;
  writecombined_ = rtFlags & CU_MEMHOSTALLOC_WRITECOMBINED;
}

void HostAlloc::set_ptr(const void *ptr) {
  ptr_ = reinterpret_cast<uintptr_t>(ptr);
}

json HostAlloc::to_json() const {
  json j = Api::to_json();
  j["portable"] = portable_;
  j["devicemap"] = devicemap_;
  j["writecombined"] = writecombined_;
  j["ptr"] = ptr_;
  j["size"] = size_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
