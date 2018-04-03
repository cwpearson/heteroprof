#ifndef CUDA_CUDNN_CREATE
#define CUDA_CUDNN_CREATE

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnCreate : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t *cudnnHandle_;

public:
  CudnnCreate(const Api &api, const cudnnHandle_t *cudnnHandle);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif