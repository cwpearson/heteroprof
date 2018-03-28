
#include "cudnn/cudnn_convolution_backward_data.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnConvolutionBackwardData::CudnnConvolutionBackwardData(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                                                           const cudnnFilterDescriptor_t wDesc, const void *w,
                                                           const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                           const cudnnConvolutionDescriptor_t convDesc,
                                                           cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
                                                           size_t workSpaceSizeInBytes, const void *beta,
                                                           const cudnnTensorDescriptor_t dxDesc, void *dx)
    : Api(api), cudnnHandle_(cudnnHandle), alpha_(alpha), wDesc_(wDesc), w_(w), dyDesc_(dyDesc), dy_(dy),
      convDesc_(convDesc), algo_(algo), workspace_(workSpace), workSpaceSizeInBytes_(workSpaceSizeInBytes), 
      beta_(beta), dxDesc_(dxDesc), dx_(dx) {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)cudnnHandle_, (uint64_t)alpha_, (uint64_t)wDesc_, (uint64_t)w_, (uint64_t)dyDesc_, (uint64_t)dy_,
                                            (uint64_t)convDesc_, (uint64_t)algo_, (uint64_t)workspace_, (uint64_t)workSpaceSizeInBytes_, (uint64_t)beta_, 
                                            (uint64_t)dxDesc_, (uint64_t)dx_
                                           };
        set_cudnn_inputs(input_vector);
      }


//Disable this for now
// json CudnnConvolutionBackwardData::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   j["alpha"] = (uint64_t)alpha_;
//   j["w_desc"] = (uint64_t)wDesc_;
//   j["w"] = (uint64_t)w_;
//   j["dy_desc"] = (uint64_t)dyDesc_;
//   j["dy"] = (uint64_t)dy_;
//   j["conv_desc"] = (uint64_t)convDesc_;
//   j["algo"] = (uint64_t)algo_;
//   j["workspace"] = (uint64_t)workspace_;
//   j["workspace_size_in_bytes"] = (uint64_t)workSpaceSizeInBytes_;
//   j["beta"] = (uint64_t)beta_;
//   j["dx_desc"] = (uint64_t)dxDesc_;
//   j["dx"] = (uint64_t)dx_;
//   return j;
// }

}  // namespace cudnn
