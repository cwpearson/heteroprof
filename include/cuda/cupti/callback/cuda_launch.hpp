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
  dim3 gridDim_;
  dim3 blockDim_;
  std::vector<uintptr_t> args_;
  size_t sharedMem_;
  cudaStream_t stream_;

public:
  CudaLaunchParams(const dim3 gridDim, const dim3 blockDim,
                   std::vector<uintptr_t> args, const size_t sharedMem,
                   const cudaStream_t stream)
      : gridDim_(gridDim), blockDim_(blockDim), args_(args),
        sharedMem_(sharedMem), stream_(stream) {}

  CudaLaunchParams(const dim3 gridDim, const dim3 blockDim,
                   const size_t sharedMem, const cudaStream_t stream)
      : gridDim_(gridDim), blockDim_(blockDim), sharedMem_(sharedMem),
        stream_(stream) {}

  CudaLaunchParams() : CudaLaunchParams(0, 0, 0, nullptr) {}

  json to_json() {
    json j;
    j["grid_dim"] = json(gridDim_);
    j["block_dim"] = json(blockDim_);
    j["args"] = json(args_);
    j["shared_mem"] = sharedMem_;
    j["stream"] = stream_;
    return j;
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
  uintptr_t func_;
  std::vector<CudaLaunchParams> params_;

public:
  CudaLaunch(const tid_t callingThread, const CUpti_CallbackData *cbdata,
             const void *func, const std::vector<CudaLaunchParams> &params)
      : Api(callingThread, cbdata),
        func_(reinterpret_cast<const uintptr_t>(func)), params_(params) {}

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif