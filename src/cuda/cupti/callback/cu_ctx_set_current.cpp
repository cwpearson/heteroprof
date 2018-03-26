#include "cuda/cupti/callback/cu_ctx_set_current.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CuCtxSetCurrent::CuCtxSetCurrent(const tid_t callingThread,
                                 const CUpti_CallbackData *cbdata,
                                 const CUcontext ctx)
    : Api(callingThread, cbdata), ctx_(ctx) {}

json CuCtxSetCurrent::to_json() const {
  json j = Api::to_json();
  j["ctx"] = ctx_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda