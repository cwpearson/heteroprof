#ifndef CUDA_SET_DEVICE_HPP
#define CUDA_SET_DEVICE_HPP

#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "model/cuda/cupti/callback/api.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

class CudaSetDevice : public model::cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = model::cuda::cupti::callback::Api;
  using tid_t = model::sys::tid_t;

private:
  const int device_;

public:
  CudaSetDevice(const tid_t callingThread, const CUpti_CallbackData *cbdata,
                const int device);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model

#endif