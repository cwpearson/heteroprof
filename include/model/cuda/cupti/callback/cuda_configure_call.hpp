#ifndef CUDA_CONFIGURE_CALL_HPP
#define CUDA_CONFIGURE_CALL_HPP

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

class CudaConfigureCall : public model::cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = model::cuda::cupti::callback::Api;

private:
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  cudaStream_t stream_;

public:
  CudaConfigureCall(const Api &api, const dim3 gridDim, const dim3 blockDim,
                    const size_t sharedMem, const cudaStream_t stream)
      : Api(api), gridDim_(gridDim), blockDim_(blockDim), sharedMem_(sharedMem),
        stream_(stream) {}

  virtual std::string profiler_type() const { return "callback_api"; }
  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model

#endif