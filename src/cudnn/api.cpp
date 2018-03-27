
#include "cudnn/api.hpp"
#include "profiler.hpp"

using namespace cudnn;
using sys::get_thread_id;
using json = nlohmann::json;
using tid_t = sys::tid_t;

Cudnn::Cudnn(const tid_t callingThread, const std::string &name)
    : Api(callingThread, name) {}


    /*
    Make cudnn its own namespace
    Commit after the first one is right so Carl can check it out
    Try to keep the JSON field names snakecase to easily deserialize into rust
    */