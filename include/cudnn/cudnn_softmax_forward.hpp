#ifndef CUDA_CUDNN_SOFTMAX_FORWARD
#define CUDA_CUDNN_SOFTMAX_FORWARD

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnSoftmaxForward : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  cudnnSoftmaxAlgorithm_t algo_;
  cudnnSoftmaxMode_t mode_;
  const void *alpha_;
  const cudnnTensorDescriptor_t xDesc_;
  const void *x_;
  const void *beta_;
  const cudnnTensorDescriptor_t yDesc_;
  void *y_;

public:
  CudnnSoftmaxForward(const Api &api, cudnnHandle_t cudnnHandle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                          const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                          const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);

 //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif