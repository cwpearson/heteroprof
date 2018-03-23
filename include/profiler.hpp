#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <atomic>
#include <memory>
#include <ostream>

#include "model/cuda/driver.hpp"
#include "model/cuda/hardware.hpp"
#include "util/environment_variable.hpp"
#include "util/logging.hpp"
#include "util/tracer.hpp"

#include "cupti_activity.hpp"

#include <nlohmann/json.hpp>

namespace profiler {
cprof::model::Driver &driver();
cprof::model::Hardware &hardware();
Timer &timer();

std::ostream &log();
void record(const std::string &s);
void record(const nlohmann::json &j);
} // namespace profiler

class Profiler {
  friend cprof::model::Driver &profiler::driver();
  friend cprof::model::Hardware &profiler::hardware();
  friend Timer &profiler::timer();

public:
  ~Profiler();

  static Profiler &instance();

  std::ostream &log();
  void record(const std::string &s);
  void record(const nlohmann::json &j);

private:
  Profiler();
  Profiler(const Profiler &) = delete;
  void operator=(const Profiler &) = delete;
  enum class Mode { DetailTimeline };

  // CUDA model
  cprof::model::Hardware hardware_;
  cprof::model::Driver driver_;
  cprof::Allocations allocations_;

  // from environment variables
  Mode mode_;

  CUpti_SubscriberHandle cuptiCallbackSubscriber_;
};

class ProfilerInitializer {
public:
  ProfilerInitializer();
};

#endif