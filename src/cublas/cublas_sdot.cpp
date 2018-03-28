
#include "cublas/cublas_sdot.hpp"


namespace cublas {

using json = nlohmann::json;

CublasSdot::CublasSdot(const Api &api, cublasHandle_t handle, int n,
                       const float *x, int incx, const float *y,
                       int incy, float *result)
    : Api(api), cublasHandle_(handle), x_(x), y_(y), result_(result)
    {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)x_, (uint64_t)y_
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)result
                                            };
        handle_ = (uintptr_t)cublasHandle_;

        set_cublas_inputs(input_vector);
        set_cublas_outputs(output_vector);
    }


}  // namespace cublas
