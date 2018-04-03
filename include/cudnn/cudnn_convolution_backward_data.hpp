#ifndef CUDA_CUDNN_CONVOLUTION_BACKWARD_DATA
#define CUDA_CUDNN_CONVOLUTION_BACKWARD_DATA

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnConvolutionBackwardData : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  const void *alpha_;
  const cudnnFilterDescriptor_t wDesc_;
  const void *w_;
  const cudnnTensorDescriptor_t dyDesc_;
  const void *dy_;
  const cudnnConvolutionDescriptor_t convDesc_;
  cudnnConvolutionBwdDataAlgo_t algo_;
  void *workspace_;
  size_t workSpaceSizeInBytes_;
  const void *beta_;
  const cudnnTensorDescriptor_t dxDesc_;
  void *dx_;

public:
  CudnnConvolutionBackwardData(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                               const cudnnFilterDescriptor_t wDesc, const void *w,
                               const cudnnTensorDescriptor_t dyDesc, const void *dy,
                               const cudnnConvolutionDescriptor_t convDesc,
                               cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
                               size_t workSpaceSizeInBytes, const void *beta,
                               const cudnnTensorDescriptor_t dxDesc, void *dx);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif