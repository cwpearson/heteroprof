
#include "cudnn/cudnn_add_tensor.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnAddTensor::CudnnAddTensor(const Api &api, const cudnnHandle_t cudnnHandle,
                                               const void *alpha, const cudnnTensorDescriptor_t aDesc, 
                                               const void *A, const void *beta, 
                                               const cudnnTensorDescriptor_t cDesc, void *C)
    : Api(api), cudnnHandle_(cudnnHandle), alpha_(alpha), 
      aDesc_(aDesc), A_(A), beta_(beta), cDesc_(cDesc),
      C_(C) {
        cudnn_handle_ = (uintptr_t)cudnnHandle_;
        std::vector<uint64_t> input_vector {
                                            (uint64_t)alpha_, (uint64_t)aDesc_, (uint64_t)A_, (uint64_t)beta_, (uint64_t)cDesc_, (uint64_t)C_
                                           };
        std::vector<uint64_t> output_vector {
                                              (uint64_t)C_
                                            };


        set_cudnn_inputs(input_vector);
        set_cudnn_outputs(output_vector);
      }

// json CudnnAddTensor::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   j["alpha"] = (uint64_t)alpha_;
//   j["a_desc"] = (uint64_t)aDesc_;
//   j["A"] = (uint64_t)A_;
//   j["beta"] = (uint64_t)beta_;
//   j["c_desc"] = (uint64_t)cDesc_;
//   j["C"] = (uint64_t)C_;
//   return j;
// }

}  // namespace cudnn
