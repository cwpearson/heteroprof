#include "model/cuda/api.hpp"

namespace model {
namespace cuda {

using tid_t = model::sys::tid_t;
using json = nlohmann::json;

Api::Api(const tid_t callingThread, const std::string &name)
    : name_(name), callingThread_(callingThread) {}

json Api::to_json() const {
  json j;
  j["name"] = name_;
  j["calling_tid"] = callingThread_;
  return j;
}

const std::string &Api::name() const { return name_; }

} // namespace cuda
} // namespace model
