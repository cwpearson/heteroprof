#ifndef CUDA_CUBLAS_SDOT
#define CUDA_CUBLAS_SDOT

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasSdot : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    const float *x_;
    const float *y_;
    float *result_;


public:
  CublasSdot(const Api &api, cublasHandle_t handle, int n,
             const float *x, int incx, const float *y,
             int incy, float *result);

};

} // namespace cudnn


#endif