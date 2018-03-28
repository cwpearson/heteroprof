
#include "cublas/cublas_destroy.hpp"


namespace cublas {

using json = nlohmann::json;

CublasDestroy::CublasDestroy(const Cublas &api, cublasHandle_t handle)
    : Api(api), cublasHandle_(handle) {
        handle_ = (uintptr_t)cublasHandle_;
    }


}  // namespace cublas
