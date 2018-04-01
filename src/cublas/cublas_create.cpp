
#include "cublas/cublas_create.hpp"


namespace cublas {

using json = nlohmann::json;

CublasCreate::CublasCreate(const Cublas &api, const cublasHandle_t *handle)
    : Api(api), cublasHandle_(handle) {
        handle_ = (uintptr_t)cublasHandle_;
    }

json CublasCreate::to_json() const {
    //Call parent class to_json
    json j = cuda::Api::to_json();
    j["handle"] = handle_;
    return j;
}


}  // namespace cublas
