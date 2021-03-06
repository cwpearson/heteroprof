#ifndef CUDA_CUDNN_CONVOLUTION_FORWARD
#define CUDA_CUDNN_CONVOLUTION_FORWARD

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnConvolutionForward : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  const void *alpha_;
    const cudnnTensorDescriptor_t xDesc_;
    const void *x_;
    const cudnnFilterDescriptor_t wDesc_;
    const void *w_;
    const cudnnConvolutionDescriptor_t convDesc_;
    cudnnConvolutionFwdAlgo_t algo_;
    void *workspace_;
    size_t workSpaceSizeInBytes_;
    const void *beta_;
    const cudnnTensorDescriptor_t yDesc_;
    void *y_;

public:
  CudnnConvolutionForward(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                          const cudnnTensorDescriptor_t xDesc, const void *x,
                          const cudnnFilterDescriptor_t wDesc, const void *w,
                          const cudnnConvolutionDescriptor_t convDesc,
                          cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                          size_t workSpaceSizeInBytes, const void *beta,
                          const cudnnTensorDescriptor_t yDesc, void *y);

 //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif