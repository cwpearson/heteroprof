#ifndef CUDA_CUBLAS_SGEMV
#define CUDA_CUBLAS_SGEMV

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasSgemv : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    cublasOperation_t trans_;
    const float *alpha_;
    const float *A_;
    const float *x_;
    const float *beta_;
    float *y_;


public:
  CublasSgemv(const Api &api, cublasHandle_t handle,
              cublasOperation_t trans, int m, int n,
              const float *alpha, const float *A,
              int lda, const float *x, int incx,
              const float *beta, float *y, int incy);

};

} // namespace cudnn


#endif