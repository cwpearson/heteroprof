#ifndef CUDA_MALLOC_HPP
#define CUDA_MALLOC_HPP

#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "model/cuda/cupti/callback/api.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

class CudaMalloc : public model::cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = model::cuda::cupti::callback::Api;
  using tid_t = model::sys::tid_t;

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
} // namespace model

#endif