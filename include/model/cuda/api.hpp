#ifndef MODEL_CUDA_API_HPP
#define MODEL_CUDA_API_HPP

#include <chrono>
#include <string>

#include "nlohmann/json.hpp"

#include "model/sys/thread.hpp"

namespace model {
namespace cuda {

class Api {
  using tid_t = model::sys::tid_t;
  using json = nlohmann::json;

protected:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;

public:
  Api(const tid_t callingThread, const std::string &name);

  virtual std::string profiler_type() const { return "api"; }
  virtual json to_json() const;
  const std::string &name() const;

  const time_point_t &wall_end() const { return wallEnd_; }
  const time_point_t &wall_start() const { return wallStart_; }
  void set_wall_start(const time_point_t &start) { wallStart_ = start; }
  void set_wall_end(const time_point_t &end) { wallEnd_ = end; }

  uint64_t wall_start_ns() const;
  uint64_t wall_end_ns() const;

protected:
  std::string name_;
  tid_t callingThread_;

  time_point_t wallStart_; ///< wall time API call started
  time_point_t wallEnd_;   ///< wall time API call ended
};

} // namespace cuda
} // namespace model
#endif