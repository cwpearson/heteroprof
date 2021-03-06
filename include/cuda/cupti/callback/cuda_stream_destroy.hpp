#ifndef CUDA_CUPTI_CALLBACK_CUDASTREAMDESTROY_HPP
#define CUDA_CUPTI_CALLBACK_CUDASTREAMDESTROY_HPP

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaStreamDestroy : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  const uintptr_t stream_;

public:
  CudaStreamDestroy(const Api &api, const cudaStream_t stream);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif