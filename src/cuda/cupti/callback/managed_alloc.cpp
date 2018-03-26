#include "cuda/cupti/callback/managed_alloc.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

ManagedAlloc::ManagedAlloc(const Api &api, const size_t size,
                           const unsigned int flags)
    : Api(api), size_(size), memAttachGlobal_(flags & cudaMemAttachGlobal),
      memAttachHost_(flags & cudaMemAttachHost) {}

void ManagedAlloc::set_ptr(const void *ptr) {
  ptr_ = reinterpret_cast<uintptr_t>(ptr);
}

json ManagedAlloc::to_json() const {
  json j = Api::to_json();
  j["ptr"] = ptr_;
  j["size"] = size_;
  j["mem_attach_global"] = memAttachGlobal_;
  j["mem_attach_host"] = memAttachHost_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
