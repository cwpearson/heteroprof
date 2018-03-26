#ifndef CUDA_CUPTI_CALLBACK_HOSTALLOC_HPP
#define CUDA_CUPTI_CALLBACK_HOSTALLOC_HPP

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class HostAlloc : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t ptr_;
  const size_t size_;

  bool portable_;
  bool devicemap_;
  bool writecombined_;

public:
  /*!
    \param rtFlags Flags from cudaMallocHost.
    \param drFlags Flags from cuMemHostAlloc.
  */
  HostAlloc(const Api &api, const size_t size, const unsigned int rtFlags,
            const unsigned int drFlags);

  void set_ptr(const void *ptr);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif