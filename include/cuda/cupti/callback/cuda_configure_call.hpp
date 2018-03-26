#ifndef CUDA_CUPTI_CALLBACK_CUDA_CONFIGURE_CALL_HPP
#define CUDA_CUPTI_CALLBACK_CUDA_CONFIGURE_CALL_HPP

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaConfigureCall : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;

private:
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  cudaStream_t stream_;

public:
  CudaConfigureCall(const Api &api, const dim3 gridDim, const dim3 blockDim,
                    const size_t sharedMem, const cudaStream_t stream)
      : Api(api), gridDim_(gridDim), blockDim_(blockDim), sharedMem_(sharedMem),
        stream_(stream) {}

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif