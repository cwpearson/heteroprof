#ifndef CUDA_CUPTI_CALLBACK_CUDASETUPARGUMENT_HPP
#define CUDA_CUPTI_CALLBACK_CUDASETUPARGUMENT_HPP

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaSetupArgument : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t arg_;
  bool is_arg_deref_;
  const size_t size_;   ///< size of argument
  const size_t offset_; ///< offset in argument stack to push new argument

public:
  CudaSetupArgument(const Api &api, const void *arg, const size_t size,
                    const size_t offset);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif