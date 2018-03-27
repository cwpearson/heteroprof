
#include "cudnn/cudnn_softmax_forward.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnSoftmaxForward::CudnnSoftmaxForward(const Api &api, cudnnHandle_t cudnnHandle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                                                const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                                const void *beta, const cudnnTensorDescriptor_t yDesc, void *y)
    : Api(api), cudnnHandle_(cudnnHandle), algo_(algo), mode_(mode),
      alpha_(alpha), xDesc_(xDesc), x_(x), beta_(beta), yDesc_(yDesc), y_(y) {}

json CudnnSoftmaxForward::to_json() const {
  json j = Api::to_json();
  j["cudnn_handle"] = (uint64_t)cudnnHandle_;
  j["algo"] = (uint64_t)algo_;
  j["mode"] = (uint64_t)mode_;
  j["alpha"] = (uint64_t)alpha_;
  j["x_desc"] = (uint64_t)xDesc_;
  j["x"] = (uint64_t)x_;
  j["beta"] = (uint64_t)beta_;
  j["y_desc"] = (uint64_t)yDesc_;
  j["y"] = (uint64_t)y_;
  return j;
}

}  // namespace cudnn
