#ifndef CUDA_CUPTI_CALLBACK_MEMCPY_HPP
#define CUDA_CUPTI_CALLBACK_MEMCPY_HPP

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class Memcpy : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

protected:
  const uintptr_t dst_;
  const uintptr_t src_;
  const size_t count_;
  cudaMemcpyKind kind_;

public:
  Memcpy(const Api &api, const void *dst, const void *src, const size_t count,
         const cudaMemcpyKind kind_);

  virtual json to_json() const override;
};

class MemcpyAsync : public Memcpy {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uintptr_t stream_;

public:
  MemcpyAsync(const Memcpy &m, const cudaStream_t stream);

  virtual json to_json() const override;
};

class MemcpyPeerAsync : public MemcpyAsync {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;
  using tid_t = sys::tid_t;

private:
  uint64_t dstDevice_;
  uint64_t srcDevice_;

public:
  MemcpyPeerAsync(const MemcpyAsync &m, const uint64_t dstDevice,
                  const uint64_t srcDevice);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif