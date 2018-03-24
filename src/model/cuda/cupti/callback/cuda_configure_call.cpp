#include "model/cuda/cupti/callback/cuda_configure_call.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

json CudaConfigureCall::to_json() const {
  auto j = Api::to_json();
  auto &v = j[profiler_type()];
  v["gridDim.x"] = gridDim_.x;
  v["gridDim.y"] = gridDim_.y;
  v["gridDim.z"] = gridDim_.z;
  v["blockDim.x"] = blockDim_.x;
  v["blockDim.y"] = blockDim_.y;
  v["blockDim.z"] = blockDim_.z;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model
