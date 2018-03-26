#include "cuda/cupti/callback/config.hpp"

namespace cuda {
namespace cupti {
namespace callback {
namespace config {

Profiler *profiler_ = nullptr;
bool recordNestedRuntime_ = false;

void set_profiler(Profiler &p) { profiler_ = &p; }
Profiler &profiler() {
  assert(profiler_);
  return *profiler_;
}

void set_record_nested_runtime(const bool enable) {
  recordNestedRuntime_ = false;
}

} // namespace config
} // namespace callback
} // namespace cupti
} // namespace cuda
