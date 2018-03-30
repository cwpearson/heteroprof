
#include "nccl/api.hpp"
#include "profiler.hpp"

using namespace nccl;
using sys::get_thread_id;
using json = nlohmann::json;
using tid_t = sys::tid_t;

Nccl::Nccl(const tid_t callingThread, const std::string &name)
    : Api(callingThread, name) {}

json Nccl::to_json() const {
    //Call parent class to_json
    json j = cuda::Api::to_json();
    j["handle"] = comm_;
    j["input_vector"] = input_vector_;
    j["output_vector"] = output_vector_;
    return j;
}

void Nccl::set_nccl_inputs(std::vector<uint64_t> input_vector){
    input_vector_ = input_vector;
}

void Nccl::set_nccl_outputs(std::vector<uint64_t> output_vector){
    output_vector_ = output_vector;
}