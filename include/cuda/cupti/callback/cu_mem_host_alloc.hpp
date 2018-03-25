#ifndef CUDA_CUPTI_CALLBACK_CUMEMHALLOCHOST_HPP
#define CUDA_CUPTI_CALLBACK_CUMEMHALLOCHOST_HPP

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CuMemHostAlloc : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t pp_;
  const size_t size_;

  const bool portable_;
  const bool devicemap_;
  const bool writecombined_;

public:
  CuMemHostAlloc(const tid_t callingThread, const CUpti_CallbackData *cbdata,
                 const size_t bytesize, const unsigned int flags);

  void set_pp(const void *const *pp);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif