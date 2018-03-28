
#include "cublas/cublas_sgemm.hpp"


namespace cublas {

using json = nlohmann::json;

CublasSgemm::CublasSgemm(const Api &api, cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, int m, int n, int k,
                         const float *alpha, /* host or device pointer */
                         const float *A, int lda, const float *B, int ldb,
                         const float *beta, /* host or device pointer */
                         float *C, int ldc)
    : Api(api), cublasHandle_(handle), transa_(transa), transb_(transb), alpha_(alpha),
      A_(A), B_(B), beta_(beta), C_(C)
    {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)transa_, (uint64_t)transb_, (uint64_t)alpha_, (uint64_t)A_, 
                                            (uint64_t)B_, (uint64_t)beta_, (uint64_t)C_
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)C_
                                            };
        handle_ = (uintptr_t)cublasHandle_;

        set_cublas_inputs(input_vector);
        set_cublas_outputs(output_vector);
    }


}  // namespace cublas
