
#include "cudnn/cudnn_activation_backward.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnActivationBackward::CudnnActivationBackward(const Api &api, cudnnHandle_t cudnnHandle, cudnnActivationDescriptor_t activationDesc,
                                               const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
                                               const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                               const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                               const cudnnTensorDescriptor_t dxDesc, void *dx)
    : Api(api), cudnnHandle_(cudnnHandle), activationDesc_(activationDesc), alpha_(alpha),
      yDesc_(yDesc), y_(y), dyDesc_(dyDesc), dy_(dy), xDesc_(xDesc), x_(x), beta_(beta),
      dxDesc_(dxDesc), dx_(dx) {
        std::vector<uint64_t> input_vector {
                                            (uint64_t) cudnnHandle_, (uint64_t)activationDesc_, (uint64_t)alpha_, (uint64_t)yDesc_, (uint64_t)y_, 
                                            (uint64_t) dyDesc_, (uint64_t)dy_, (uint64_t)xDesc_, (uint64_t)x_, (uint64_t)beta_, (uint64_t)dxDesc_, (uint64_t)dx_
                                           };
        set_cudnn_inputs(input_vector);
      }

//Not necessary for now
// json CudnnActivationBackward::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   j["activation_desc"] = (uint64_t)activationDesc_;
//   j["alpha"] = (uint64_t)alpha_;
//   j["y_desc"] = (uint64_t)yDesc_;
//   j["y"] = (uint64_t)y_;
//   j["dy_desc"] = (uint64_t)dyDesc_;
//   j["dy"] = (uint64_t)dy_;
//   j["x_desc"] = (uint64_t)xDesc_;
//   j["x"] = (uint64_t)x_;
//   j["beta"] = (uint64_t)beta_;
//   j["dx_desc"] = (uint64_t)dxDesc_;
//   j["dx"] = (uint64_t)dx_;
//   return j;
// }

}  // namespace cudnn
