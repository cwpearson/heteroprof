#ifndef CUDA_SETUP_ARGUMENT_HPP
#define CUDA_SETUP_ARGUMENT_HPP

#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaSetupArgument : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  const uintptr_t arg_;
  const size_t size_;   ///< size of argument
  const size_t offset_; ///< offset in argument stack to push new argument

public:
  CudaSetupArgument(const tid_t callingThread, const CUpti_CallbackData *cbdata,
                    const void *arg, const size_t size, const size_t offset);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif