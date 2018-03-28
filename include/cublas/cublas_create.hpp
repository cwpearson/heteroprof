#ifndef CUDA_CUBLAS_CREATE
#define CUDA_CUBLAS_CREATE

#include "cublas/api.hpp"
// #include "cudnn/util.hpp"


namespace cublas {

class CublasCreate : public cublas::Cublas {
  using json = nlohmann::json;
  using Api = cublas::Cublas;
  using tid_t = sys::tid_t;

protected:
  const cublasHandle_t *cublasHandle_;

public:
  CublasCreate(const Api &api, const cublasHandle_t *handle);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif