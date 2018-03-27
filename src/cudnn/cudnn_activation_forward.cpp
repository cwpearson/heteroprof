
#include "cudnn/cudnn_activation_forward.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnActivationForward::CudnnActivationForward(const Api &api, const cudnnHandle_t cudnnHandle,
                                                cudnnActivationDescriptor_t activationDesc, const void* alpha,
                                                const cudnnTensorDescriptor_t xDesc, const void *x, 
                                                const void *beta, const cudnnTensorDescriptor_t yDesc,
                                                void *y)
    : Api(api), cudnnHandle_(cudnnHandle), activationDesc_(activationDesc), 
      activationDesc_(activationDesc), alpha_(alpha), xDesc_(xDesc), x_(x),
      beta_(beta), yDesc_(yDesc), y_(y) {}

json CudnnCreate::to_json() const {
  json j = Api::to_json();
  j["cudnn_handle"] = (uint64_t)cudnnHandle_;
  j["activation_desc"] = (uint64_t)activationDesc_;
  j["alpha"] = (uint64_t)alpha_;
  j["x_desc"] = (uint64_t)xDesc_;
  j["x"] = (uint64_t)x_;
  j["beta"] = (uint64_t)beta_;
  j["y_desc"] = (uint64_t)yDesc_;
  j["y"] = (uint64_t)y_;
  return j;
}

}  // namespace cudnn
