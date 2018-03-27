#ifndef CUDA_CUDNN_ACTIVATION_FORWARD
#define CUDA_CUDNN_ACTIVATION_FORWARD

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnActivationForward : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  cudnnActivationDescriptor_t activationDesc_;
  const void *alpha_;
  const cudnnTensorDescriptor_t xDesc_;
  const void *x_;
  const void *beta_;
  const cudnnTensorDescriptor_t yDesc_;
  void *y_;

public:
  CudnnActivationForward(const Api &api, const cudnnHandle_t cudnnHandle,
                         cudnnActivationDescriptor_t activationDesc, const void* alpha,
                         const cudnnTensorDescriptor_t xDesc, const void *x, 
                         const void *beta, const cudnnTensorDescriptor_t yDesc,
                         void *y);

  virtual json to_json() const override;
};

} // namespace cudnn


#endif