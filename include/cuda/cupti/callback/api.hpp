#ifndef API_HPP
#define API_HPP

#include <string>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "cuda/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class Api : public cuda::Api {
  using json = nlohmann::json;

public:
  typedef unsigned short domain_t;
  using tid_t = sys::tid_t;

private:
  const uint32_t contextUid_;
  const uint32_t correlationId_;
  std::string symbolName_;
  const domain_t domain_;

public:
  Api(const tid_t callingThread, const CUpti_CallbackData *cbdata,
      const CUpti_CallbackDomain domain);

  virtual json to_json() const override;
  virtual std::string hprof_kind() const override { return "cupti_callback"; }

  domain_t domain() const noexcept { return domain_; }
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif