
#include "cudnn/cudnn_convolution_backward_filter.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnConvolutionBackwardFilter::CudnnConvolutionBackwardFilter(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                                                           const cudnnTensorDescriptor_t xDesc, const void *x,
                                                           const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                           const cudnnConvolutionDescriptor_t convDesc,
                                                           cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
                                                           size_t workSpaceSizeInBytes, const void *beta,
                                                           const cudnnFilterDescriptor_t dwDesc, void *dw)
    : Api(api), cudnnHandle_(cudnnHandle), alpha_(alpha), xDesc_(xDesc), x_(x), dyDesc_(dyDesc), dy_(dy),
      convDesc_(convDesc), algo_(algo), workspace_(workSpace), workSpaceSizeInBytes_(workSpaceSizeInBytes),
      beta_(beta), dwDesc_(dwDesc), dw_(dw) {
        cudnn_handle_ = (uintptr_t)cudnnHandle_;
        std::vector<uint64_t> input_vector {
                                            (uint64_t)alpha_, (uint64_t)xDesc_, (uint64_t)x_, (uint64_t)dyDesc_, (uint64_t)dy_,
                                            (uint64_t)convDesc_, (uint64_t)algo_, (uint64_t)workspace_, (uint64_t)workSpaceSizeInBytes_, (uint64_t)beta_, 
                                            (uint64_t)dwDesc_, (uint64_t)dw_
                                           };
        std::vector<uint64_t> output_vector {
                                              (uint64_t)dw_
                                            };
        set_cudnn_inputs(input_vector);
        set_cudnn_outputs(output_vector);
      }

//Disable for now
// json CudnnConvolutionBackwardFilter::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   j["alpha"] = (uint64_t)alpha_;
//   j["x_desc"] = (uint64_t)xDesc_;
//   j["x"] = (uint64_t)x_;
//   j["dy_desc"] = (uint64_t)dyDesc_;
//   j["dy"] = (uint64_t)dy_;
//   j["conv_desc"] = (uint64_t)convDesc_;
//   j["algo"] = (uint64_t)algo_;
//   j["workspace"] = (uint64_t)workspace_;
//   j["workspace_size_in_bytes"] = (uint64_t)workSpaceSizeInBytes_;
//   j["beta"] = (uint64_t)beta_;
//   j["dw_desc"] = (uint64_t)dwDesc_;
//   j["dw"] = (uint64_t)dw_;
//   return j;
// }

}  // namespace cudnn
