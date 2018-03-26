#include "cuda/cupti/callback/cu_ctx_set_current.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

CuCtxSetCurrent::CuCtxSetCurrent(const Api &api, const CUcontext ctx)
    : Api(api), ctx_(reinterpret_cast<uintptr_t>(ctx)) {
  static_assert(sizeof(uintptr_t) == sizeof(ctx), "uh oh");
}

json CuCtxSetCurrent::to_json() const {
  json j = Api::to_json();
  j["ctx"] = ctx_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda