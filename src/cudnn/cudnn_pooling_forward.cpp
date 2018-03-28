
#include "cudnn/cudnn_pooling_forward.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnPoolingForward::CudnnPoolingForward(const Api &api, cudnnHandle_t cudnnHandle, const cudnnPoolingDescriptor_t poolingDesc,
                                         const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                         const void *beta, const cudnnTensorDescriptor_t yDesc, void *y)
    : Api(api), cudnnHandle_(cudnnHandle), poolingDesc_(poolingDesc), alpha_(alpha), xDesc_(xDesc), x_(x),
      beta_(beta), yDesc_(yDesc), y_(y) {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)cudnnHandle_, (uint64_t)poolingDesc_, (uint64_t)alpha_, (uint64_t)xDesc_, (uint64_t)x_,
                                            (uint64_t)beta_, (uint64_t)yDesc_, (uint64_t)y_
                                           };
        set_cudnn_inputs(input_vector);
      }

//Disable for now
// json CudnnPoolingForward::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   j["pooling_desc"] =  (uint64_t)poolingDesc_;
//   j["alpha"] =  (uint64_t)alpha_;
//   j["x_desc"] =  (uint64_t)xDesc_;
//   j["x"] =  (uint64_t)x_;
//   j["beta"] =  (uint64_t)beta_;
//   j["y_desc"] = (uint64_t)yDesc_;
//   j["y"] = (uint64_t)y_;
//   return j;
// }

}  // namespace cudnn
