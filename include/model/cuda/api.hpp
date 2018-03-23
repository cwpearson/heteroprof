#ifndef MODEL_CUDA_API_HPP
#define MODEL_CUDA_API_HPP

#include <string>

#include "nlohmann/json.hpp"

#include "model/sys/thread.hpp"

namespace model {
namespace cuda {

class Api {
  using tid_t = model::sys::tid_t;
  using json = nlohmann::json;

public:
  Api(const tid_t callingThread, const std::string &name);

  json to_json() const;
  const std::string &name() const;

protected:
  std::string name_;
  tid_t callingThread_;
};

} // namespace cuda
} // namespace model
#endif