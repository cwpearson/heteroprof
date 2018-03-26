#include "cuda/cupti/callback/memcpy.hpp"

namespace cuda {
namespace cupti {
namespace callback {

using json = nlohmann::json;

Memcpy::Memcpy(const Api &api, const void *dst, const void *src,
               const size_t count, const cudaMemcpyKind kind)
    : Api(api), dst_(reinterpret_cast<uintptr_t>(dst)),
      src_(reinterpret_cast<uintptr_t>(src)), count_(count), kind_(kind) {}

json Memcpy::to_json() const {
  json j = Api::to_json();
  j["dst"] = dst_;
  j["src"] = src_;
  j["count"] = count_;
  j["cuda_memcpy_kind"] = kind_;
  return j;
}

MemcpyAsync::MemcpyAsync(const Memcpy &m, const cudaStream_t stream)
    : Memcpy(m), stream_(reinterpret_cast<uintptr_t>(stream)) {
  static_assert(sizeof(uintptr_t) == sizeof(cudaStream_t), "uh oh");
}

json MemcpyAsync::to_json() const {
  json j = Memcpy::to_json();
  j["stream"] = stream_;
  return j;
}

MemcpyPeerAsync::MemcpyPeerAsync(const MemcpyAsync &m, const uint64_t dstDevice,
                                 const uint64_t srcDevice)
    : MemcpyAsync(m), dstDevice_(dstDevice), srcDevice_(srcDevice) {}

json MemcpyPeerAsync::to_json() const {
  json j = Api::to_json();
  j["dst_device"] = dstDevice_;
  j["src_device"] = srcDevice_;
  return j;
}

} // namespace callback
} // namespace cupti
} // namespace cuda
