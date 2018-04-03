
#include "cublas/cublas_sasum.hpp"


namespace cublas {

using json = nlohmann::json;

CublasSasum::CublasSasum(const Api &api, cublasHandle_t handle, int n,
                         const float *x, int incx, float *result)  
    : Api(api), cublasHandle_(handle), x_(x), result_(result)
    {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)x_
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)result_
                                            };
        handle_ = (uintptr_t)cublasHandle_;

        set_cublas_inputs(input_vector);
        set_cublas_outputs(output_vector);
    }


}  // namespace cublas
