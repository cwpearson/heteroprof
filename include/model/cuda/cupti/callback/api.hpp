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

public:
  typedef uint64_t id_type;

private:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  std::vector<uintptr_t> args_;
  std::string apiName_;
  std::string kernelName_;
  int device_;
  std::map<std::string, std::string> kv_;

  CUpti_CallbackDomain domain_;
  CUpti_CallbackId cbid_;
  int64_t correlationId_;

  /// Wall time of start and end of API
  time_point_t wallStart_;
  time_point_t wallEnd_;

public:
  Api(const std::string &apiName, const int device,
      const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
      const int64_t correlationId, const CUpti_CallbackData *cbInfo)
      : apiName_(apiName), device_(device), domain_(domain), cbid_(cbid),
        correlationId_(correlationId), wallStart_(std::chrono::nanoseconds(0)),
        wallEnd_(std::chrono::nanoseconds(0)) {}
  Api(const int device, const CUpti_CallbackDomain domain,
      const CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
      : Api(cbInfo->functionName, device, domain, cbid, cbInfo->correlationId,
            cbInfo) {}

  void add_arg(const void *);
  void add_kv(const std::string &key, const std::string &val);
  void add_kv(const std::string &key, const size_t &val);
  void set_wall_start(const time_point_t &start);
  void set_wall_end(const time_point_t &end);
  void set_wall_time(const time_point_t &start, const time_point_t &end);

  int device() const { return device_; }
  const std::string &name() const { return apiName_; }

  json to_json() const;
  std::string to_json_string() const;

  bool is_runtime() const { return domain_ == CUPTI_CB_DOMAIN_RUNTIME_API; }
  CUpti_CallbackDomain domain() const { return domain_; }
  CUpti_CallbackId cbid() const { return cbid_; }

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