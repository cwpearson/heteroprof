
#include "cudnn/cudnn_convolution_forward.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnConvolutionForward::CudnnConvolutionForward(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                                                 const cudnnTensorDescriptor_t xDesc, const void *x,
                                                 const cudnnFilterDescriptor_t wDesc, const void *w,
                                                 const cudnnConvolutionDescriptor_t convDesc,
                                                 cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                 size_t workSpaceSizeInBytes, const void *beta,
                                                 const cudnnTensorDescriptor_t yDesc, void *y)
    : Api(api), cudnnHandle_(cudnnHandle), alpha_(alpha), xDesc_(xDesc), x_(x), wDesc_(wDesc), w_(w),
      convDesc_(convDesc), algo_(algo), workspace_(workSpace), workSpaceSizeInBytes_(workSpaceSizeInBytes),
      beta_(beta), yDesc_(yDesc), y_(y) {}

json CudnnConvolutionForward::to_json() const {
  json j = Api::to_json();
  j["cudnn_handle"] = (uint64_t)cudnnHandle_;
  j["alpha"] = (uint64_t)alpha_;
  j["x_desc"] = (uint64_t)xDesc_;
  j["x"] = (uint64_t)x_;
  j["w_desc"] = (uint64_t)wDesc_;
  j["w"] = (uint64_t)w_;
  j["conv_desc"] = (uint64_t)convDesc_;
  j["algo"] = (uint64_t)algo_;
  j["workspace"] = (uint64_t)workspace_;
  j["workspace_size_in_bytes"] = (uint64_t)workSpaceSizeInBytes_;
  j["beta"] = (uint64_t)beta_;
  j["y_desc"] = (uint64_t)yDesc_;
  j["y"] = (uint64_t)y_;
  return j;
}

}  // namespace cudnn
