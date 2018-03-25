#ifndef CUDNN_API_HPP
#define CUDNN_API_HPP

#include "cuda/api.hpp"
#include "cudnn/api.hpp"
#include "sys/thread.hpp"

namespace cudnn {

class Cudnn : public cuda::Api {

  using tid_t = sys::tid_t;

public:
  Cudnn(const tid_t callingThread, const std::string &name);

  virtual std::string hprof_kind() const override { return "cudnn"; }
};

} // namespace cudnn
#endif