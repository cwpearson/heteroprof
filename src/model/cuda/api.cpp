#include "model/cuda/api.hpp"

#include "util/chrono.hpp"

namespace model {
namespace cuda {

using tid_t = model::sys::tid_t;
using json = nlohmann::json;

Api::Api(const tid_t callingThread, const std::string &name)
    : name_(name), callingThread_(callingThread) {}

json Api::to_json() const {
  json j;
  auto &v = j[profiler_type()];
  v["name"] = name_;
  v["calling_tid"] = callingThread_;
  v["wall_start"] = wall_start_ns();
  v["wall_end"] = wall_end_ns();
  return j;
}

const std::string &Api::name() const { return name_; }

uint64_t Api::wall_start_ns() const { return epoch_nanos(wallStart_); }
uint64_t Api::wall_end_ns() const { return epoch_nanos(wallEnd_); }
} // namespace cuda
} // namespace model
