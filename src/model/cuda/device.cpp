#include "model/cuda/device.hpp"

using namespace model::cuda;

Device::Device(const cudaDeviceProp &prop, int id)
    : id_(id), unifiedAddressing_(prop.unifiedAddressing),
      canMapHostMemory_(prop.canMapHostMemory),
      pageableMemoryAccess_(prop.pageableMemoryAccess),
      concurrentManagedAccess_(prop.concurrentManagedAccess),
      major_(prop.major), minor_(prop.minor) {}