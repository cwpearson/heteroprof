#ifndef CUDA_CUDNN_ADD_TENSOR
#define CUDA_CUDNN_ADD_TENSOR

#include "cudnn/api.hpp"
#include "cudnn/util.hpp"


namespace cudnn {

class CudnnAddTensor : public cudnn::Cudnn {
  using json = nlohmann::json;
  using Api = cudnn::Cudnn;
  using tid_t = sys::tid_t;

protected:
  const cudnnHandle_t cudnnHandle_;
  const void *alpha_;
  const cudnnTensorDescriptor_t aDesc_;
  const void *A_;
  const void *beta_;
  const cudnnTensorDescriptor_t cDesc_;
  void *C_;

public:
  CudnnAddTensor(const Api &api, const cudnnHandle_t cudnnHandle,
                 const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A,
                 const void *beta, const cudnnTensorDescriptor_t cDesc, void *C);

  virtual json to_json() const override;
};

} // namespace cudnn


#endif