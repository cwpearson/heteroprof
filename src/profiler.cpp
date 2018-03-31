#include <cassert>
#include <fstream>
#include <iostream>

#include "cuda/cupti/util.hpp"
#include "util/environment_variable.hpp"

#include "cuda/cupti/activity/config.hpp"
#include "cuda/cupti/callback/callback.hpp"
#include "cuda/cupti/callback/config.hpp"
#include "cudnn/preload.hpp"
#include "cublas/preload.hpp"
#include "preload_cublas.hpp"
#include "preload_nccl.hpp"
#include "profiler.hpp"
#include "version.hpp"

using cuda::Driver;
using cuda::Hardware;
using nlohmann::json;

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

  cuda::cupti::callback::config::set_profiler(*this);
  cuda::cupti::activity::config::set_profiler(*this);

  {
    auto n = EnvironmentVariable<uint32_t>("CPROF_CUPTI_DEVICE_BUFFER_SIZE", 0)
                 .get();
    if (n != 0) {
      cuda::cupti::activity::config::set_device_buffer_size(n);
    }
    log() << "INFO: CUpti activity device buffer size: "
          << *cuda::cupti::activity::config::attr_device_buffer_size()
          << std::endl;
  }
  // Set CUPTI parameters
  CUPTI_CHECK(cuptiActivitySetAttribute(
                  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                  cuda::cupti::activity::config::attr_value_size(
                      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE),
                  cuda::cupti::activity::config::attr_device_buffer_size()),
              log());
  CUPTI_CHECK(
      cuptiActivitySetAttribute(
          CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
          cuda::cupti::activity::config::attr_value_size(
              CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT),
          cuda::cupti::activity::config::attr_device_buffer_pool_limit()),
      log());

  // Enable CUPTI Activity API
  // CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS causes nccl hang on minsky2
  auto cuptiActivityKinds = std::vector<CUpti_ActivityKind>{
      CUPTI_ACTIVITY_KIND_KERNEL,          CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_ENVIRONMENT,     CUPTI_ACTIVITY_KIND_CUDA_EVENT,
      CUPTI_ACTIVITY_KIND_DRIVER,          CUPTI_ACTIVITY_KIND_RUNTIME,
      CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, CUPTI_ACTIVITY_KIND_OVERHEAD};
  log() << "INFO: enabling activity API" << std::endl;
  for (const auto &kind : cuptiActivityKinds) {

    CUptiResult code = cuptiActivityEnable(kind);
    if (code == CUPTI_ERROR_NOT_COMPATIBLE) {
      log() << "WARN: CUPTI_ERROR_NOT_COMPATIBLE when enabling " << kind
            << std::endl;
    } else if (code == CUPTI_ERROR_INVALID_KIND) {
      log() << "WARN: CUPTI_ERROR_INVALID_KIND when enabling " << kind
            << std::endl;
    } else {
      CUPTI_CHECK(code, log());
      log() << "INFO: enabled cuptiActivityKind " << kind << std::endl;
    }
  }

  log() << "INFO: done enabling activity API" << std::endl;
  log() << "INFO: registering activity callbacks" << std::endl;
  CUPTI_CHECK(cuptiActivityRegisterCallbacks(cuptiActivityBufferRequested,
                                             cuptiActivityBufferCompleted),
              log());
  log() << "INFO: done registering activity callbacks" << std::endl;

  // Enable CUPTI Callback API
  log() << "INFO: CuptiSubscriber enabling callback API" << std::endl;
  CUPTI_CHECK(cuptiSubscribe(&cuptiCallbackSubscriber_,
                             (CUpti_CallbackFunc)cuptiCallbackFunction,
                             nullptr),
              log());

  log() << "INFO: scanning devices" << std::endl;
  hardware_.get_device_properties();
  log() << "INFO: done" << std::endl;

  CUPTI_CHECK(cuptiEnableDomain(1, cuptiCallbackSubscriber_,
                                CUPTI_CB_DOMAIN_RUNTIME_API),
              log());
  CUPTI_CHECK(cuptiEnableDomain(1, cuptiCallbackSubscriber_,
                                CUPTI_CB_DOMAIN_DRIVER_API),
              log());
  log() << "INFO: done enabling callback API domains" << std::endl;

  if (WITH_CUDNN) {
    cudnn::set_profiler(*this);
  }

  //Need to add if statement
  cublas::set_profiler(*this);

  log() << "INFO: dumping version" << std::endl;
  log() << version() << std::endl;
  log() << version_git() << std::endl;
  log() << version_build() << std::endl;
  json j;
  j["version"] = version();
  j["git"] = version_git();
  j["build"] = version_build();
  record(j);
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
void Profiler::record(const std::vector<nlohmann::json> &j){
  for (nlohmann::json cur_json : j){
    logging::atomic_out(cur_json.dump() + "\n");
  }
}


Driver &Profiler::driver() { return driver_; }

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
