#include "cuda/cupti/callback/host_free.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

HostFree::HostFree(const tid_t callingThread, const CUpti_CallbackData *cbdata,
                   const void *ptr)
    : Api(callingThread, cbdata), ptr_(reinterpret_cast<const uintptr_t>(ptr)) {
}

json HostFree::to_json() const {
  json j = Api::to_json();
  j["ptr"] = ptr_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
