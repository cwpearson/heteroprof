#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <string>

#include <nlohmann/json.hpp>

namespace cuda {

class AddressSpace {

public:
  enum class Kind {
    Unknown,
    Host,       ///< CUDA <4.0 host address space
    CudaDevice, ///< CUDA <4.0 address space for a single device
    CudaUVA,    ///< CUDA >4.0 unified virtual addressing
    Invalid
  };

private:
  uint64_t id_;
  Kind kind_;

  using json = nlohmann::json;

  json to_json() const;
  std::string to_json_string() const;

public:
  static AddressSpace CudaDevice(uint64_t device_id) {
    AddressSpace a;
    a.kind_ = Kind::CudaDevice;
    a.id_ = device_id;
    return a;
  }

  static AddressSpace CudaUVA() {
    AddressSpace a;
    a.kind_ = Kind::CudaUVA;
    return a;
  }
};

} // namespace cuda
#endif
