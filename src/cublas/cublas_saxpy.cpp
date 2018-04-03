
#include "cublas/cublas_saxpy.hpp"


namespace cublas {

using json = nlohmann::json;

CublasSaxpy::CublasSaxpy(const Api &api, cublasHandle_t handle, int n,
                         const float *alpha, /* host or device pointer */
                         const float *x, int incx, float *y, int incy)
    : Api(api), cublasHandle_(handle), alpha_(alpha), x_(x), y_(y)
    {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)alpha_, (uint64_t)x_, (uint64_t)y_
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)y_
                                            };
        handle_ = (uintptr_t)cublasHandle_;

        set_cublas_inputs(input_vector);
        set_cublas_outputs(output_vector);
    }


}  // namespace cublas
