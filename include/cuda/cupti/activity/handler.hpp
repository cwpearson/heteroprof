#ifndef CUDA_CUPTI_ACTIVITY_HANDLER_HPP
#define CUDA_CUPTI_ACTIVITY_HANDLER_HPP

#include <cupti.h>

#include "profiler.hpp"

namespace cuda {
namespace cupti {
namespace activity {
void handler(const CUpti_Activity *record, Profiler &profiler);
}
} // namespace cupti
} // namespace cuda

#endif