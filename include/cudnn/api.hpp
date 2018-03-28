#ifndef CUDNN_API_HPP
#define CUDNN_API_HPP

#include "cuda/api.hpp"
#include "cudnn/api.hpp"
#include "sys/thread.hpp"
#include <vector>

namespace cudnn {

class Cudnn : public cuda::Api {
  using tid_t = sys::tid_t;
  using json = nlohmann::json;


public:
  Cudnn(const tid_t callingThread, const std::string &name);

  virtual std::string hprof_kind() const override { return "cudnn"; }

  virtual json to_json() const override;

protected:
  void set_cudnn_inputs(std::vector<uint64_t> input_vector);
  void set_cudnn_outputs(std::vector<uint64_t> output_vector);
  uintptr_t cudnn_handle_;

private:
  std::vector<uint64_t> input_vector_;
  std::vector<uint64_t> output_vector_;
};

} // namespace cudnn
#endif