#ifndef API_HPP
#define API_HPP

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

#include "model/cuda/api.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

class Api : public model::cuda::Api {
  using json = nlohmann::json;
  using tid_t = model::sys::tid_t;

public:
  typedef uint64_t id_type;

private:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  std::vector<uintptr_t> args_;
  std::string kernelName_;
  int device_;
  std::map<std::string, std::string> kv_;

  CUcontext context;
  uint32_t contextUid;
  uint64_t *correlationData;
  uint32_t correlationId;
  const char *functionName;
  const void *functionParams;
  void *functionReturnValue;
  const char *symbolName;

  int64_t correlationId_;

  /// Wall time of start and end of API
  time_point_t wallStart_;
  time_point_t wallEnd_;

public:
  Api(const tid_t callingThread, const CUpti_CallbackData *cbdata);

  void add_arg(const void *);
  void add_kv(const std::string &key, const std::string &val);
  void add_kv(const std::string &key, const size_t &val);
  void set_wall_start(const time_point_t &start);
  void set_wall_end(const time_point_t &end);
  void set_wall_time(const time_point_t &start, const time_point_t &end);

  int device() const { return device_; }

  json to_json() const;
  std::string to_json_string() const;

  const time_point_t &wall_end() const { return wallEnd_; }
  const time_point_t &wall_start() const { return wallStart_; }

  uint64_t wall_start_ns() const;
  uint64_t wall_end_ns() const;
};

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model

#endif