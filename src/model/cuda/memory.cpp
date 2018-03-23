#include "model/cuda/memory.hpp"

#include <cassert>

using json = nlohmann::json;
using namespace model::cuda;

json to_json(const Memory &m) {
  json j;
  switch (m) {
  case Memory::Unknown:
    j["type"] = "unknown";
    break;
  case Memory::Pageable:
    j["type"] = "pageable";
    break;
  case Memory::Pagelocked:
    j["type"] = "pagelocked";
    break;
  case Memory::Unified3:
    j["type"] = "unified3";
    break;
  case Memory::Unified6:
    j["type"] = "unified6";
    break;
  default:
    assert(0 && "Unexpected memory type");
  }
  return j;
}