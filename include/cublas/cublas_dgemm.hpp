#ifndef CUDA_CUBLAS_DGEMM
#define CUDA_CUBLAS_DGEMM

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasDGemm : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    cublasOperation_t transa_;
    cublasOperation_t transb_;
    const double *alpha_;
    const double *A_;
    const double *B_;
    const double *beta_;
    //Input/Outputs
    double *C_;


public:
  CublasDGemm(const Api &api, cublasHandle_t handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k, const double *alpha,
              const double *A, int lda, const double *B, int ldb,
              const double *beta, double *C, int ldc);

};

} // namespace cudnn


#endif