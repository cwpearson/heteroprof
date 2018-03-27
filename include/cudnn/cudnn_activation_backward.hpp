#ifndef CUDA_CUDNN_ACTIVATION_BACKWARD
#define CUDA_CUDNN_ACTIVATION_BACKWARD

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnActivationBackward : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  cudnnActivationDescriptor_t activationDesc_;
  const void *alpha_;
  const cudnnTensorDescriptor_t yDesc_;
  const void *y_;
  const cudnnTensorDescriptor_t dyDesc_;
  const void *dy_;
  const cudnnTensorDescriptor_t xDesc_;
  const void *x_;
  const void *beta_;
  const cudnnTensorDescriptor_t dxDesc_;
  void *dx_;

public:
  CudnnActivationBackward(const Api &api, cudnnHandle_t cudnnHandle, cudnnActivationDescriptor_t activationDesc,
                          const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
                          const cudnnTensorDescriptor_t dyDesc, const void *dy,
                          const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                          const cudnnTensorDescriptor_t dxDesc, void *dx);

  virtual json to_json() const override;
};

} // namespace cudnn


#endif