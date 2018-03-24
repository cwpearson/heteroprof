#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <atomic>
#include <memory>
#include <ostream>

#include "nlohmann/json.hpp"

#include "model/cuda/driver.hpp"
#include "model/cuda/hardware.hpp"
#include "util/environment_variable.hpp"
#include "util/logging.hpp"

#include "cupti_activity.hpp"

namespace profiler {
using model::cuda::Hardware;
Hardware &hardware();

std::ostream &log();
void record(const std::string &s);
void record(const nlohmann::json &j);
} // namespace profiler

class Profiler {
  using Driver = model::cuda::Driver;
  using Hardware = model::cuda::Hardware;

  friend Hardware &profiler::hardware();

public:
  ~Profiler();

  static Profiler &instance();

  std::ostream &log();
  void record(const std::string &s);
  void record(const nlohmann::json &j);

  Driver &driver();

private:
  Profiler();
  Profiler(const Profiler &) = delete;
  void operator=(const Profiler &) = delete;

  // CUDA model
  Hardware hardware_;
  Driver driver_;

  CUpti_SubscriberHandle cuptiCallbackSubscriber_;
};

class ProfilerInitializer {
public:
  ProfilerInitializer();
};

#endif