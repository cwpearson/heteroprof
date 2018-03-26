#include "cuda/cupti/callback/cuda_setup_argument.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CudaSetupArgument::CudaSetupArgument(const Api &api, const void *arg,
                                     const size_t size, const size_t offset)
    : Api(api), arg_(reinterpret_cast<uintptr_t>(arg)), size_(size),
      offset_(offset) {}

json CudaSetupArgument::to_json() const {
  json j = Api::to_json();
  j["arg"] = arg_;
  j["size"] = size_;
  j["offset"] = offset_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda