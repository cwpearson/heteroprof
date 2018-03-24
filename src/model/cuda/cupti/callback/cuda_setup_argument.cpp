#include "model/cuda/cupti/callback/cuda_setup_argument.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = model::sys::tid_t;

CudaSetupArgument::CudaSetupArgument(const tid_t callingThread,
                                     const CUpti_CallbackData *cbdata,
                                     const void *arg, const size_t size,
                                     const size_t offset)
    : Api(callingThread, cbdata), arg_(reinterpret_cast<uintptr_t>(arg)),
      size_(size), offset_(offset) {}

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
} // namespace model