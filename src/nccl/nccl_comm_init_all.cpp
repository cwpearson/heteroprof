
#include "nccl/nccl_comm_init_all.hpp"


namespace nccl {

using json = nlohmann::json;

NcclCommInitAll::NcclCommInitAll(const Nccl &api, ncclComm_t *comms, int nGPUs,
                                 const int *devList)
    : Api(api), comm_(comms), nGPUs_(nGPUs), devList_(devList) {
        fill_in_handles();
    }

//Have specialized json function due to the fact that it contains vector to comms
json NcclCommInitAll::to_json() const {
    json j = cuda::Api::to_json();
    j["handle"] = (uint64_t)comm_;
    // j["input_vector"] = input_vector_;
    // j["output_vector"] = output_vector_;
    return j;
}

std::vector<json> NcclCommInitAll::to_json_vector(){
    return handle_json_;
}

void NcclCommInitAll::fill_in_handles(){
    for (int i=0; i<nGPUs_; i++){
        const int dev = devList_ ? devList_[i] : i;
        handle_json_.push_back(make_handle_json(comm_[i], dev));
    }
}

json NcclCommInitAll::make_handle_json(ncclComm_t comm, int cur_gpu){
    //Investigate the device list
    json j = cuda::Api::to_json();
    j["ncclComm_t"] = (uint64_t)comm;
    j["gpu"] = (uint64_t)cur_gpu;
    return j;
}

}  // namespace nccl
