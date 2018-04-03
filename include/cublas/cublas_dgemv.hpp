#ifndef CUDA_CUBLAS_DGEMV
#define CUDA_CUBLAS_DGEMV

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasDgemv : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    cublasOperation_t trans_;
    const double *alpha_;
    const double *A_;
    const double *x_;
    const double *beta_;
    double *y_;


public:
  CublasDgemv(const Api &api, cublasHandle_t handle,
              cublasOperation_t trans, int m, int n,
              const double *alpha, const double *A,
              int lda, const double *x, int incx,
              const double *beta, double *y, int incy);

};

} // namespace cudnn


#endif