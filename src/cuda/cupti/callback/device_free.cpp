#include "cuda/cupti/callback/device_free.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

DeviceFree::DeviceFree(const Api &api, const void *ptr)
    : Api(api), ptr_(reinterpret_cast<const uintptr_t>(ptr)) {}

json DeviceFree::to_json() const {
  json j = Api::to_json();
  j["ptr"] = ptr_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
