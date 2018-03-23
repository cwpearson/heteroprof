#include <cassert>

#include <nlohmann/json.hpp>

#include "model/cuda/api.hpp"
#include "model/cuda/cupti/callback/api.hpp"
#include "model/sys/thread.hpp"

using json = nlohmann::json;
using namespace model::cuda::cupti::callback;
using tid_t = model::sys::tid_t;

Api::Api(const tid_t callingThread, const CUpti_CallbackData *cbdata)
    : model::cuda::Api(callingThread, cbdata->functionName) {
  name_ = cbdata->functionName;
}

void Api::add_kv(const std::string &k, const std::string &v) { kv_[k] = v; }
void Api::add_kv(const std::string &k, const size_t &v) {
  add_kv(k, std::to_string(v));
}

void Api::set_wall_start(const time_point_t &start) { wallStart_ = start; }

void Api::set_wall_end(const time_point_t &end) { wallEnd_ = end; }

void Api::set_wall_time(const time_point_t &start, const time_point_t &end) {
  set_wall_start(start);
  set_wall_end(end);
}

uint64_t Api::wall_start_ns() const {}
uint64_t Api::wall_end_ns() const {}

json Api::to_json() const {

  json j = model::cuda::Api::to_json();

  j["api"]["device"] = device_;
  j["api"]["symbolname"] = kernelName_;
  j["api"]["args"] = json(args_);
  j["api"]["wall_start"] = wall_start_ns();
  j["api"]["wall_end"] = wall_end_ns();
  j["api"]["correlation_id"] = correlationId_;
  j["api"]["kv"] = json(kv_);
  return j;
}
