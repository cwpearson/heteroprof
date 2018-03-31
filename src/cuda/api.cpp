#include "cuda/api.hpp"

#include "util/chrono.hpp"

namespace cuda {

using tid_t = sys::tid_t;
using json = nlohmann::json;

std::atomic<size_t> Api::count_(1);

Api::Api(const tid_t callingThread, const std::string &name)
    : id_(count_++), name_(name), callingThread_(callingThread) {}

json Api::to_json() const {
  json j;
  j["hprof_kind"] = hprof_kind();
  j["id"] = id_;
  j["name"] = name_;
  j["calling_tid"] = callingThread_;
  j["wall_start"] = wall_start_ns();
  j["wall_end"] = wall_end_ns();
  return j;
}

json Api::to_json_vector() const {
  assert(0 && "to_json_vector not implemented here");
}

const std::string &Api::name() const { return name_; }

uint64_t Api::wall_start_ns() const { return epoch_nanos(wallStart_); }
uint64_t Api::wall_end_ns() const { return epoch_nanos(wallEnd_); }
} // namespace cuda
