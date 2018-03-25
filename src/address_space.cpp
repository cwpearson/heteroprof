#include "cuda/address_space.hpp"

using json = nlohmann::json;
using AddressSpace = cuda::AddressSpace;

static std::string to_string(const AddressSpace::Kind &t) {
  switch (t) {
  case AddressSpace::Kind::Host:
    return "host";
  case AddressSpace::Kind::CudaDevice:
    return "cuda";
  case AddressSpace::Kind::CudaUVA:
    return "uva";
  case AddressSpace::Kind::Unknown:
    return "unknown";
  default:
    assert(0 && "Unhandled AddressSpace::Type");
  }
}

json AddressSpace::to_json() const {
  json j;
  j["type"] = to_string(kind_);
  j["id"] = id_;
  return j;
}

std::string AddressSpace::to_json_string() const { return to_json().dump(); }
