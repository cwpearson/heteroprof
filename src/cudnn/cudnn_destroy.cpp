
#include "cudnn/cudnn_destroy.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnDestroy::CudnnDestroy(const Cudnn &api, const cudnnHandle_t cudnnHandle)
    : Api(api), cudnnHandle_(cudnnHandle) {
      
      cudnn_handle_ = (uintptr_t)cudnnHandle_;
    }


//Disable this for now
// json CudnnDestroy::to_json() const {
//   json j = Api::to_json();
//   j["cudnn_handle"] = (uint64_t)cudnnHandle_;
//   return j;
// }

}  // namespace cudnn
