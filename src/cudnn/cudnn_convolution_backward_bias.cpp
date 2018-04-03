
#include "cudnn/cudnn_convolution_backward_bias.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnConvolutionBackwardBias::CudnnConvolutionBackwardBias(const Api &api, cudnnHandle_t cudnnHandle, const void *alpha,
                                                           const cudnnTensorDescriptor_t dyDesc,
                                                           const void *dy, const void *beta,
                                                           const cudnnTensorDescriptor_t dbDesc, void *db)
    : Api(api), cudnnHandle_(cudnnHandle), alpha_(alpha), dyDesc_(dyDesc), dy_(dy), beta_(beta),
      dbDesc_(dbDesc), db_(db) {
        cudnn_handle_ = (uintptr_t)cudnnHandle_;
        std::vector<uint64_t> input_vector {
                                            (uint64_t)alpha_, (uint64_t)dyDesc_, (uint64_t)dy_, (uint64_t)beta_,
                                            (uint64_t)dbDesc_
                                           };

        std::vector<uint64_t> output_vector {
                                              (uint64_t)db_
                                            };
        set_cudnn_inputs(input_vector);
        set_cudnn_outputs(output_vector);
      }


//Disable this for now
// json CudnnConvolutionBackwardBias::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   j["alpha"] = (uint64_t)alpha_;
//   j["dy_desc"] = (uint64_t)dyDesc_;
//   j["dy"] = (uint64_t)dy_;
//   j["beta"] = (uint64_t)beta_;
//   j["db_desc"] = (uint64_t)dbDesc_;
//   j["db"] = (uint64_t)db_;
//   return j;
// }

}  // namespace cudnn
