#ifndef CUDA_STREAM_CREATE_HPP
#define CUDA_STREAM_CREATE_HPP

#include "nlohmann/json.hpp"

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class CudaStreamCreate : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  cudaStream_t stream_;

public:
  CudaStreamCreate(const Api &api);

  void set_stream(const cudaStream_t stream);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif