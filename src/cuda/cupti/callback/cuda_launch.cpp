#include "cuda/cupti/callback/cuda_launch.hpp"

using json = nlohmann::json;

void to_json(json &j, const dim3 &d) {
  j = json{{"x", d.x}, {"y", d.y}, {"z", d.z}};
}

void to_json(json &j, const cudaStream_t &s) {
  static_assert(sizeof(s) == sizeof(uintptr_t), "uh oh");
  j = reinterpret_cast<uintptr_t>(s);
}

namespace cuda {
namespace cupti {
namespace callback {

void to_json(json &j, const CudaLaunchParams &clp) { j = clp.to_json(); }

json CudaLaunchParams::to_json() const {
  json j;
  j["grid_dim"] = gridDim_;
  j["block_dim"] = blockDim_;
  j["args"] = args_;
  j["shared_mem"] = sharedMem_;
  j["stream"] = stream_;
  return j;
}

json CudaLaunch::to_json() const {
  json j = Api::to_json();
  j["func"] = func_;
  j["params"] = params_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
