#ifndef CUDA_CUDNN_CONVOLUTION_BACKWARD_BIAS
#define CUDA_CUDNN_CONVOLUTION_BACKWARD_BIAS

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnConvolutionBackwardBias : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  const void *alpha_;
  const cudnnTensorDescriptor_t dyDesc_;
  const void *dy_;
  const void *beta_;
  const cudnnTensorDescriptor_t dbDesc_;
  void *db_;

public:
  CudnnConvolutionBackwardBias(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                               const cudnnTensorDescriptor_t dyDesc,
                               const void *dy, const void *beta,
                               const cudnnTensorDescriptor_t dbDesc, void *db);

  virtual json to_json() const override;
};

} // namespace cudnn


#endif