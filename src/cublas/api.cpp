
#include "cublas/api.hpp"
#include "profiler.hpp"

using namespace cublas;
using sys::get_thread_id;
using json = nlohmann::json;
using tid_t = sys::tid_t;

Cublas::Cublas(const tid_t callingThread, const std::string &name)
    : Api(callingThread, name) {}

json Cublas::to_json() const {
    //Call parent class to_json
    json j = cuda::Api::to_json();
    j["handle"] = handle_;
    j["input_vector"] = input_vector_;
    j["output_vector"] = output_vector_;
    return j;
}

void Cublas::set_cublas_inputs(std::vector<uint64_t> input_vector){
    input_vector_ = input_vector;
}

void Cublas::set_cublas_outputs(std::vector<uint64_t> output_vector){
    output_vector_ = output_vector;
}

    /*
    Make cudnn its own namespace
    Commit after the first one is right so Carl can check it out
    Try to keep the JSON field names snakecase to easily deserialize into rust
    */