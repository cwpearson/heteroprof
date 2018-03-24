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
  using tid_t = model::sys::tid_t;

private:
  std::vector<uintptr_t> args_;
  uint32_t contextUid_;
  uint32_t correlationId_;
  std::string symbolName_;

public:
  Api(const tid_t callingThread, const CUpti_CallbackData *cbdata);

  void add_arg(const void *);
  void add_kv(const std::string &key, const std::string &val);
  void add_kv(const std::string &key, const size_t &val);

  virtual std::string profiler_type() const { return "callback_api"; }
  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model

#endif