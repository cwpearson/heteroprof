#ifndef CUDA_CUPTI_CALLBACK_MANAGEDALLOC_HPP
#define CUDA_CUPTI_CALLBACK_MANAGEDALLOC_HPP

#include "cuda/cupti/callback/api.hpp"

namespace cuda {
namespace cupti {
namespace callback {

class ManagedAlloc : public cuda::cupti::callback::Api {
  using json = nlohmann::json;
  using Api = cuda::cupti::callback::Api;

private:
  uintptr_t ptr_;
  const size_t size_;
  const bool memAttachGlobal_;
  const bool memAttachHost_;

public:
  ManagedAlloc(const Api &a, const size_t size, const unsigned int flags);

  void set_ptr(const void *ptr);

  virtual json to_json() const override;
};

} // namespace callback
} // namespace cupti
} // namespace cuda

#endif