#ifndef CUDA_CUDNN_DESTROY
#define CUDA_CUDNN_DESTROY

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnDestroy : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  cudnnHandle_t cudnnHandle_;

public:
  CudnnDestroy(const Api &api, const cudnnHandle_t cudnnHandle);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif