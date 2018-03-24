#include "model/cuda/cupti/callback/arg.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

Arg::Arg(const void *arg, const size_t size, const size_t offset)
    : arg_(reinterpret_cast<uintptr_t>(arg)), size_(size), offset_(offset) {}

json Arg::to_json() const {
  json j;
  j["arg"] = arg_;
  j["size"] = size_;
  j["offset"] = offset_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model