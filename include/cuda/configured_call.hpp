#ifndef MODEL_CUDA_CONFIGURED_CALL_HPP
#define MODEL_CUDA_CONFIGURED_CALL_HPP

#include <vector>

#include <cuda_runtime.h>

namespace cuda {

class ConfiguredCall {
public:
  ConfiguredCall() : numArgs_(0), valid_(false) {}
  void add_arg() { numArgs_ += 1; }
  size_t num_args() const noexcept {
    assert(valid_);
    return numArgs_;
  }

  void start() noexcept {
    assert(!valid_);
    numArgs_ = 0;
    valid_ = true;
  }

  void finish() noexcept {
    assert(valid_);
    numArgs_ = 0;
    valid_ = false;
  }

private:
  size_t numArgs_;
  bool valid_;
};

} // namespace cuda

#endif