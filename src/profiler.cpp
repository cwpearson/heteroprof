#include <cassert>
#include <fstream>
#include <iostream>

#include "util/environment_variable.hpp"
#include "util_cupti.hpp"

#include "cupti_activity_handler.hpp"
#include "cupti_callback.hpp"
#include "preload_cublas.hpp"
#include "preload_cudnn.hpp"
#include "preload_nccl.hpp"
#include "profiler.hpp"

using model::cuda::Driver;
using model::cuda::Hardware;
using nlohmann::json;

namespace profiler {
Driver &driver() { return Profiler::instance().driver_; }
Hardware &hardware() { return Profiler::instance().hardware_; }

void record(const std::string &s) { Profiler::instance().record(s); }
void record(const json &j) { Profiler::instance().record(j); }
std::ostream &log() { return Profiler::instance().log(); }
} // namespace profiler

/*! \brief Profiler() create a profiler
 *
 * Should not handle any initialization. Defer that to the init() method.
 */
Profiler::Profiler() {

  // Configure logging
  auto quiet = EnvironmentVariable<bool>("CPROF_QUIET", false).get();
  if (quiet) {
    logging::disable_err();
  }
  auto outPath = EnvironmentVariable<std::string>("CPROF_OUT", "-").get();
  if (outPath != "-") {
    logging::set_out_path(outPath.c_str());
  }
  auto errPath = EnvironmentVariable<std::string>("CPROF_ERR", "-").get();
  if (errPath != "-") {
    logging::set_err_path(errPath.c_str());
  }

  {
    auto n = EnvironmentVariable<uint32_t>("CPROF_CUPTI_DEVICE_BUFFER_SIZE", 0)
                 .get();
    if (n != 0) {
      cupti_activity_config::set_device_buffer_size(n);
    }
    log() << "INFO: CUpti activity device buffer size: "
          << *cupti_activity_config::attr_device_buffer_size() << std::endl;
  }

  // Set CUPTI parameters
  CUPTI_CHECK(cuptiActivitySetAttribute(
                  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                  cupti_activity_config::attr_value_size(
                      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE),
                  cupti_activity_config::attr_device_buffer_size()),
              log());
  CUPTI_CHECK(cuptiActivitySetAttribute(
                  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                  cupti_activity_config::attr_value_size(
                      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT),
                  cupti_activity_config::attr_device_buffer_pool_limit()),
              log());

  // Set handler functions
  cupti_activity_config::add_activity_handler(activityHander);

  // disable preloads
  preload_nccl::set_passthrough(true);
  preload_cublas::set_passthrough(true);
  preload_cudnn::set_passthrough(true);

  // Enable CUPTI Activity API
  auto cuptiActivityKinds = std::vector<CUpti_ActivityKind>{
      CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_ENVIRONMENT, // not compatible on minsky2
      CUPTI_ACTIVITY_KIND_CUDA_EVENT,  // FIXME:available before cuda9?
      CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME,
      CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
      // CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS, // causes a hang in nccl on
      // minsky2
      CUPTI_ACTIVITY_KIND_OVERHEAD};
  log() << "INFO: Profiler enabling activity API" << std::endl;
  for (const auto &kind : cuptiActivityKinds) {
    log() << "DEBU: Enabling cuptiActivityKind " << kind << std::endl;
    CUptiResult code = cuptiActivityEnable(kind);
    if (code == CUPTI_ERROR_NOT_COMPATIBLE) {
      log() << "WARN: CUPTI_ERROR_NOT_COMPATIBLE when enabling " << kind
            << std::endl;
    } else if (code == CUPTI_ERROR_INVALID_KIND) {
      log() << "WARN: CUPTI_ERROR_INVALID_KIND when enabling " << kind
            << std::endl;
    } else {
      CUPTI_CHECK(code, log());
    }
  }
  CUPTI_CHECK(cuptiActivityRegisterCallbacks(cuptiActivityBufferRequested,
                                             cuptiActivityBufferCompleted),
              log());

  // Enable CUPTI Callback API
  log() << "INFO: CuptiSubscriber enabling callback API" << std::endl;
  CUPTI_CHECK(cuptiSubscribe(&cuptiCallbackSubscriber_,
                             (CUpti_CallbackFunc)cuptiCallbackFunction,
                             nullptr),
              log());
  CUPTI_CHECK(cuptiEnableDomain(1, cuptiCallbackSubscriber_,
                                CUPTI_CB_DOMAIN_RUNTIME_API),
              log());
  CUPTI_CHECK(cuptiEnableDomain(1, cuptiCallbackSubscriber_,
                                CUPTI_CB_DOMAIN_DRIVER_API),
              log());
  log() << "INFO: done enabling callback API domains" << std::endl;

  log() << "INFO: scanning devices" << std::endl;
  hardware_.get_device_properties();
  log() << "INFO: done" << std::endl;
}

Profiler::~Profiler() {
  logging::err() << "Profiler dtor\n";
  log() << "INFO: CuptiSubscriber Deactivating callback API!" << std::endl;
  CUPTI_CHECK(cuptiUnsubscribe(cuptiCallbackSubscriber_), log());
  log() << "INFO: done deactivating callbacks!" << std::endl;
  log() << "INFO: CuptiSubscriber cleaning up activity API" << std::endl;
  cuptiActivityFlushAll(0);
  log() << "INFO: done cuptiActivityFlushAll" << std::endl;
  logging::err() << "Profiler dtor done.\n";
}

std::ostream &Profiler::log() { return logging::err(); }
void Profiler::record(const std::string &s) { return logging::atomic_out(s); }
void Profiler::record(const json &j) {
  return logging::atomic_out(j.dump() + "\n");
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}

bool isInitialized = false;
std::mutex initOnce;

ProfilerInitializer::ProfilerInitializer() {
  std::lock_guard<std::mutex> lock(initOnce);
  if (!isInitialized) {
    Profiler::instance();
    isInitialized = true;
  }
}

static ProfilerInitializer pi;
