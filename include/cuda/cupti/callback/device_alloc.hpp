#ifndef CUDA_CUPTI_CALLBACK_CUDAMALLOC_HPP
#define CUDA_CUPTI_CALLBACK_CUDAMALLOC_HPP

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class DeviceAlloc : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t ptr_;
  const size_t size_;

public:
  DeviceAlloc(const tid_t callingThread, const CUpti_CallbackData *cbdata,
              const size_t size);

  void set_ptr(const void *ptr);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif