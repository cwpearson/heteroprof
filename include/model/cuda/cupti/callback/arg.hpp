#ifndef ARG_HPP
#define ARG_HPP

#include <map>
#include <string>
#include <vector>

#include <cupti.h>

#include "nlohmann/json.hpp"

namespace model {
namespace cuda {
namespace cupti {
namespace callback {

class Arg {
  using json = nlohmann::json;

private:
  uintptr_t arg_;
  size_t size_;
  size_t offset_;

public:
  Arg(const void *arg, const size_t size, const size_t offset);
  json to_json() const;
};

} // namespace callback
} // namespace cupti
} // namespace cuda
} // namespace model

#endif