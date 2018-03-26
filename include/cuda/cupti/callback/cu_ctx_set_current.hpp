#ifndef CUDA_CUPTI_CALLBACK_CUCTXSETCURRENT_HPP
#define CUDA_CUPTI_CALLBACK_CUCTXSETCURRENT_HPP

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CuCtxSetCurrent : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  const CUcontext ctx_;

public:
  CuCtxSetCurrent(const tid_t callingThread, const CUpti_CallbackData *cbdata,
                  const CUcontext ctx);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif