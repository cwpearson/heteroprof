#ifndef CUDA_CUBLAS_SAXPY
#define CUDA_CUBLAS_SAXPY

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasSaxpy : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    const float *alpha_;
    const float *x_;
    float *y_;


public:
  CublasSaxpy(const Api &api, cublasHandle_t handle, int n,
              const float *alpha, /* host or device pointer */
              const float *x, int incx, float *y, int incy);

};

} // namespace cublas


#endif