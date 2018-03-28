#ifndef CUDA_CUBLAS_SASUM
#define CUDA_CUBLAS_SASUM

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasSasum : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    const float *x_;
    float *result_;


public:
  CublasSasum(const Api &api, cublasHandle_t handle, int n,
              const float *x, int incx, float *result);

};

} // namespace cudnn


#endif