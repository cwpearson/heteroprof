#include "cuda/cupti/callback/cu_mem_host_alloc.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;
using tid_t = sys::tid_t;

CuMemHostAlloc::CuMemHostAlloc(const tid_t callingThread,
                               const CUpti_CallbackData *cbdata,
                               const size_t bytesize, const unsigned int flags)
    : Api(callingThread, cbdata), size_(bytesize), pp_(0),
      portable_(flags & CU_MEMHOSTALLOC_PORTABLE),
      devicemap_(flags & CU_MEMHOSTALLOC_DEVICEMAP),
      writecombined_(flags & CU_MEMHOSTALLOC_WRITECOMBINED) {}

void CuMemHostAlloc::set_pp(const void *const *pp) {
  pp_ = reinterpret_cast<uintptr_t>(*pp);
}

json CuMemHostAlloc::to_json() const {
  json j = Api::to_json();
  j["portable"] = portable_;
  j["devicemap"] = devicemap_;
  j["writecombined"] = writecombined_;
  j["pp"] = pp_;
  j["size"] = size_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
