#ifndef CUDA_CUBLAS_SGEMM
#define CUDA_CUBLAS_SGEMM

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasSgemm : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    cublasOperation_t transa_;
    cublasOperation_t transb_;
    const float *alpha_;
    const float *A_;
    const float *B_;
    const float *beta_;
    float *C_;


public:
  CublasSgemm(const Api &api, cublasHandle_t handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k,
              const float *alpha, /* host or device pointer */
              const float *A, int lda, const float *B, int ldb,
              const float *beta, /* host or device pointer */
              float *C, int ldc);

};

} // namespace cudnn


#endif