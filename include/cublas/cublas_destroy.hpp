#ifndef CUDA_CUBLAS_DESTROY
#define CUDA_CUBLAS_DESTROY

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasDestroy : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
  cublasHandle_t cublasHandle_;

public:
  CublasDestroy(const Api &api, cublasHandle_t handle);

};

} // namespace cudnn


#endif