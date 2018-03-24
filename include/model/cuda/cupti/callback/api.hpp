#ifndef API_HPP
#define API_HPP

#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "model/cuda/api.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

class Api : public model::cuda::Api {
  using json = nlohmann::json;

private:
  uint32_t contextUid_;
  uint32_t correlationId_;
  std::string symbolName_;

public:
  using tid_t = model::sys::tid_t;
  Api(const tid_t callingThread, const CUpti_CallbackData *cbdata);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model

#endif