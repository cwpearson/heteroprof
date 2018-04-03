#ifndef CUDA_CUBLAS_SSCAL
#define CUDA_CUBLAS_SSCAL

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasSscal : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
    //Inputs
    cublasHandle_t cublasHandle_;
    const float *alpha_;
    float *x_;


public:
  CublasSscal(const Api &api, cublasHandle_t handle, int n,
            const float *alpha, /* host or device pointer */
            float *x, int incx);

};

} // namespace cudnn


#endif