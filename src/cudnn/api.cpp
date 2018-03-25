
#include "cudnn/api.hpp"

using namespace cudnn;

using json = nlohmann::json;
using tid_t = sys::tid_t;

Cudnn::Cudnn(const tid_t callingThread, const std::string &name)
    : Api(callingThread, name) {}