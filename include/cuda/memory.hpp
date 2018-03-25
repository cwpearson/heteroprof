#ifndef MODEL_CUDA_MEMORY_HPP
#define MODEL_CUDA_MEMORY_HPP

#include <nlohmann/json.hpp>

namespace cuda {

enum class Memory {
  Unknown,    ///< an unknown type of memory
  Pageable,   ///< CUDA pageable memory
  Pagelocked, ///< CUDA Page-locked memory
  Unified3,   ///< CUDA unified memory >sm_30
  Unified6    ///< CUDA unified memory >sm_60
};

nlohmann::json to_json(const Memory &m);

} // namespace cuda

#endif