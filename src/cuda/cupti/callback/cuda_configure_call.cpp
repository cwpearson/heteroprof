#include "cuda/cupti/callback/cuda_configure_call.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

json to_json(const dim3 &d) { return json{{"x", d.x}, {"y", d.y}, {"z", d.z}}; }

json CudaConfigureCall::to_json() const {
  auto j = Api::to_json();
  j["grid_dim"] = cuda::cupti::callback::to_json(gridDim_);
  j["block_dim"] = cuda::cupti::callback::to_json(blockDim_);
  j["stream"] = stream_;
  j["shared_mem"] = sharedMem_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
