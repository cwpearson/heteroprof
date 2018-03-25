#ifndef CUDA_MALLOC_HPP
#define CUDA_MALLOC_HPP

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaMalloc : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t devPtr_;
  const size_t size_;

public:
  CudaMalloc(const tid_t callingThread, const CUpti_CallbackData *cbdata,
             const size_t size);

  void set_devptr(const void *const *devPtr);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif