
#include "cudnn/cudnn_create.hpp"


namespace cudnn {

using json = nlohmann::json;

CudnnCreate::CudnnCreate(const Cudnn &api, const cudnnHandle_t *cudnnHandle)
    : Api(api), cudnnHandle_(cudnnHandle) {}

json CudnnCreate::to_json() const {
  json j = Api::to_json();
  j["cudnn_handle*"] = (uint64_t)cudnnHandle_;
  return j;
}

}  // namespace cudnn
