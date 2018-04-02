#include "cuda/cupti/callback/cuda_setup_argument.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CudaSetupArgument::CudaSetupArgument(const Api &api, const void *arg,
                                     const size_t size, const size_t offset)
    : Api(api), size_(size), offset_(offset) {
  if (size <= sizeof(arg_)) { // arg could be a pointer
    arg_ = *reinterpret_cast<const uintptr_t *>(arg);
    is_arg_deref_ = true;
  } else {
    arg_ = reinterpret_cast<uintptr_t>(arg);
    is_arg_deref_ = false;
  }
}

json CudaSetupArgument::to_json() const {
  json j = Api::to_json();
  j["is_arg_deref"] = is_arg_deref_;
  j["arg"] = arg_;
  j["size"] = size_;
  j["offset"] = offset_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda