
#include "cublas/cublas_sgemv.hpp"


namespace cublas {

using json = nlohmann::json;

CublasSgemv::CublasSgemv(const Api &api, cublasHandle_t handle,
                         cublasOperation_t trans, int m, int n,
                         const float *alpha, const float *A,
                         int lda, const float *x, int incx,
                         const float *beta, float *y, int incy)
    : Api(api), cublasHandle_(handle), trans_(trans), alpha_(alpha), A_(A), 
      x_(x), beta_(beta), y_(y)
    {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)trans_, (uint64_t)alpha_, (uint64_t)A_, (uint64_t)x_,
                                            (uint64_t)beta_, (uint64_t)y_
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)y_
                                            };
        handle_ = (uintptr_t)cublasHandle_;

        set_cublas_inputs(input_vector);
        set_cublas_outputs(output_vector);
    }


}  // namespace cublas
