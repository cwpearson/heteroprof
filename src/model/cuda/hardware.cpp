#include <cuda_runtime.h>

#include "model/cuda/hardware.hpp"
#include "util/logging.hpp"
#include "util_cuda.hpp"

using namespace model::cuda;

void Hardware::get_device_properties() {
  static bool done = false;
  if (done)
    return;

  done = true;

  int numDevices;
  CUDA_CHECK(cudaGetDeviceCount(&numDevices), logging::err());
  logging::err() << "INFO: scanning " << numDevices << " cuda devices\n";
  for (int i = 0; i < numDevices; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i), logging::err());
    cudaDevices_.push_back(cuda::Device(prop, i));
  }
}