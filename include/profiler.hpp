#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <atomic>
#include <memory>
#include <ostream>

#include "nlohmann/json.hpp"

#include "cuda/cupti/activity/config.hpp"
#include "cuda/driver.hpp"
#include "cuda/hardware.hpp"
#include "util/environment_variable.hpp"
#include "util/logging.hpp"

class Profiler {
  using Driver = cuda::Driver;
  using Hardware = cuda::Hardware;

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