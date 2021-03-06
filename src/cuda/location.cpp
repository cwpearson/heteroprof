#include "cuda/location.hpp"

using json = nlohmann::json;
using cuda::Location;

json Location::to_json() const {
  json j;
  j["type"] = to_string(type_);
  j["id"] = id_;
  return j;
}

std::string Location::to_json_string() const { return to_json().dump(); }