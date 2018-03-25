#ifndef CUDA_HARDWARE_HPP
#define CUDA_HARDWARE_HPP

#include <vector>

#include "cuda/address_space.hpp"
#include "cuda/device.hpp"

namespace cuda {

/*! \brief Represents information about the physical system
 *
 * We need to know some information about the profiled system in order
 * to fully understand the semantics of various APIs.
 */
class Hardware {

  using Device = cuda::Device;
  using AddressSpace = cuda::AddressSpace;

  std::vector<Device> cudaDevices_;

public:
  const Device &cuda_device(size_t i) { return cudaDevices_[i]; }
  void get_device_properties();

  /*! \brief Address space a device participates in
   */
  const AddressSpace address_space(const int dev) {
    if (cudaDevices_[dev].unifiedAddressing_) {
      return AddressSpace::CudaUVA();
    }
    return AddressSpace::CudaDevice(dev);
  }
};

} // namespace cuda

#endif