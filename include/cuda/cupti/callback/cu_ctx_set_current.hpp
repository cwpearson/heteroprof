#ifndef CUDA_CUPTI_CALLBACK_CUCTXSETCURRENT_HPP
#define CUDA_CUPTI_CALLBACK_CUCTXSETCURRENT_HPP

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CuCtxSetCurrent : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  const uintptr_t ctx_;

public:
  CuCtxSetCurrent(const Api &api, const CUcontext ctx);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif