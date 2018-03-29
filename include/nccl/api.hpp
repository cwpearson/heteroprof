#ifndef NCCL_API_HPP
#define NCCL_API_HPP

#include "cuda/api.hpp"
#include "nccl/api.hpp"
#include "sys/thread.hpp"
#include <vector>

namespace nccl {

class Nccl : public cuda::Api {
  using tid_t = sys::tid_t;
  using json = nlohmann::json;


public:
  Nccl(const tid_t callingThread, const std::string &name);

  virtual std::string hprof_kind() const override { return "nccl"; }

  virtual json to_json() const override;

protected:
  void set_nccl_inputs(std::vector<uint64_t> input_vector);
  void set_nccl_outputs(std::vector<uint64_t> output_vector);
  uintptr_t nccl_handle_;

private:
  std::vector<uint64_t> input_vector_;
  std::vector<uint64_t> output_vector_;
};

} // namespace cudnn
#endif