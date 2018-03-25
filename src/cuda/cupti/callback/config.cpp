#include "cuda/cupti/callback/config.hpp"

namespace cuda {
namespace cupti {
namespace callback {
namespace config {

Profiler *profiler_ = nullptr;
void set_profiler(Profiler &p) { profiler_ = &p; }
Profiler &profiler() {
  assert(profiler_);
  return *profiler_;
}

} // namespace config
} // namespace callback
} // namespace cupti
} // namespace cuda
