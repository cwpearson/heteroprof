#ifndef CUDA_CUDNN_CONVOLUTION_BACKWARD_FILTER
#define CUDA_CUDNN_CONVOLUTION_BACKWARD_FILTER  

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnConvolutionBackwardFilter : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  const void *alpha_;
  const cudnnTensorDescriptor_t xDesc_;
  const void *x_;
  const cudnnTensorDescriptor_t dyDesc_;
  const void *dy_;
  const cudnnConvolutionDescriptor_t convDesc_;
  cudnnConvolutionBwdFilterAlgo_t algo_;
  void *workspace_;
  size_t workSpaceSizeInBytes_;
  const void *beta_;
  const cudnnFilterDescriptor_t dwDesc_;
  void *dw_;

public:
  CudnnConvolutionBackwardFilter(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                                 const cudnnTensorDescriptor_t xDesc, const void *x,
                                 const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                 const cudnnConvolutionDescriptor_t convDesc,
                                 cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
                                 size_t workSpaceSizeInBytes, const void *beta,
                                 const cudnnFilterDescriptor_t dwDesc, void *dw);

  virtual json to_json() const override;
};

} // namespace cudnn


#endif