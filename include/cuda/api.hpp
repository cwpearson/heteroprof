#ifndef CUDA_API_HPP
#define CUDA_API_HPP

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "sys/thread.hpp"

namespace cuda {

class Api {
  using tid_t = sys::tid_t;
  using json = nlohmann::json;

protected:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;

public:
  Api(const tid_t callingThread, const std::string &name);

  virtual size_t id() const { return id_; }
  virtual json to_json() const;
  virtual std::vector<json> to_json_vector() const;
  virtual std::string hprof_kind() const { return "cuda_api"; }
  const std::string &name() const;

  const time_point_t &wall_end() const { return wallEnd_; }
  const time_point_t &wall_start() const { return wallStart_; }
  void set_wall_start(const time_point_t &start) { wallStart_ = start; }
  void set_wall_end(const time_point_t &end) { wallEnd_ = end; }

  uint64_t wall_start_ns() const;
  uint64_t wall_end_ns() const;

protected:
  static std::atomic<size_t> count_;
  size_t id_;

  std::string name_;
  tid_t callingThread_;

  time_point_t wallStart_; ///< wall time API call started
  time_point_t wallEnd_;   ///< wall time API call ended
};

} // namespace cuda
#endif