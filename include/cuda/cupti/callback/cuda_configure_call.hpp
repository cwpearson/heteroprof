#ifndef CUDA_CONFIGURE_CALL_HPP
#define CUDA_CONFIGURE_CALL_HPP

#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaConfigureCall : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;

private:
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  cudaStream_t stream_;

public:
  CudaConfigureCall(const tid_t callingThread, const CUpti_CallbackData *cbdata,
                    const dim3 gridDim, const dim3 blockDim,
                    const size_t sharedMem, const cudaStream_t stream)
      : Api(callingThread, cbdata), gridDim_(gridDim), blockDim_(blockDim),
        sharedMem_(sharedMem), stream_(stream) {}

  virtual std::string hprof_kind() const override { return "cupti_callback"; }
  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif