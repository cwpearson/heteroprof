#ifndef CUBLAS_API_HPP
#define CUBLAS_API_HPP

#include <cublas_v2.h>
#include "cuda/api.hpp"
#include "cudnn/api.hpp"
#include "sys/thread.hpp"
#include <vector>



namespace cublas {

class Cublas : public cuda::Api {
  using tid_t = sys::tid_t;
  using json = nlohmann::json;


public:
  Cublas(const tid_t callingThread, const std::string &name);

  virtual std::string hprof_kind() const override { return "cublas"; }

  virtual json to_json() const override;

protected:
  void set_cublas_inputs(std::vector<uint64_t> input_vector);
  void set_cublas_outputs(std::vector<uint64_t> output_vector);
  uintptr_t handle_;

private:
  std::vector<uint64_t> input_vector_;
  std::vector<uint64_t> output_vector_;
};

} // namespace cublas
#endif