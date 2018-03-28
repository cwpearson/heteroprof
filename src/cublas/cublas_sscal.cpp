
#include "cublas/cublas_sscal.hpp"


namespace cublas {

using json = nlohmann::json;

CublasSscal::CublasSscal(const Api &api, cublasHandle_t handle, int n,
                         const float *alpha, /* host or device pointer */
                         float *x, int incx)
    : Api(api), cublasHandle_(handle), alpha_(alpha), x_(x)
    {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)alpha_, (uint64_t)x_
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)x_
                                            };
        handle_ = (uintptr_t)cublasHandle_;

        set_cublas_inputs(input_vector);
        set_cublas_outputs(output_vector);
    }


}  // namespace cublas
