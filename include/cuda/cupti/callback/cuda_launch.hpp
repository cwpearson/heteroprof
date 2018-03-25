#ifndef CUDA_CUPTI_CALLBACK_CUDA_LAUNCH_HPP
#define CUDA_CUPTI_CALLBACK_CUDA_LAUNCH_HPP

#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaLaunchParams {
  using json = nlohmann::json;

private:
  dim3 gridDim;
  dim3 blockDim;
  std::vector<uintptr_t> args_;
  size_t sharedMem_;
  cudaStream_t stream_;

public:
  json to_json() {
    json j;
    j["grid_dim"] = json(gridDim);
    j["block_dim"] = json(blockDim);
    j["args"] = json(args_);
    j["shared_mem"] = sharedMem_;
    j["stream"] = stream_;
  }
};

// cudaLaunch
// cudaLaunchKernel
// cudaLaunchCooperativeKernel
// cudaLaunchCOoperativeKernelMultiDevice
class CudaLaunch : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;

private:
  std::string cudaLaunchKind_;
  std::vector<CudaLaunchParams> params_;

public:
  CudaLaunch(const tid_t callingThread, const CUpti_CallbackData *cbdata,
             const std::string &cudaLaunchKind,
             const std::vector<CudaLaunchParams> &params)
      : Api(callingThread, cbdata), cudaLaunchKind_(cudaLaunchKind),
        params_(params) {}

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif