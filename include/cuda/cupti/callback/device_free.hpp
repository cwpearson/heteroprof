#ifndef CUDA_CUPTI_CALLBACK_DEVICEFREE_HPP
#define CUDA_CUPTI_CALLBACK_DEVICEFREE_HPP

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class DeviceFree : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t ptr_;

public:
  /*!
    \param rtFlags Flags from cudaMallocHost.
    \param drFlags Flags from cuMemHostAlloc.
  */
  DeviceFree(const Api &api, const void *ptr);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif