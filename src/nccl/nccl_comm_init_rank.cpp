
#include "nccl/nccl_comm_init_rank.hpp"


namespace nccl {

using json = nlohmann::json;

NcclCommInitRank::NcclCommInitRank(const Nccl &api, ncclComm_t *comm, int ndev,
                                   ncclUniqueId cliqueId, int rank)
    : Api(api), comm_(comm), ndev_(ndev), cliqueId_(cliqueId), rank_(rank) {
        // device_ = profiler().driver().this_thread().device(comm_);
        // handle_ = (uintptr_t)cublasHandle_;
        fill_in_handles();
    }

    std::vector<json> NcclCommInitRank::to_json_vector() const {
        return handle_json_;
    }   

    void NcclCommInitRank::fill_in_handles(){
        // for (int i=0; i<nGPUs_; i++){
        //     const int dev = devList_ ? devList_[i] : i;
        //     handle_json_.push_back(make_handle_json(comm_[i], dev));
        // }
    }

json NcclCommInitRank::make_handle_json(ncclComm_t comm, int cur_gpu){
    //Investigate the device list
    json j = cuda::Api::to_json();
    j["ncclComm_t"] = (uint64_t)comm;
    j["gpu"] = (uint64_t)cur_gpu;
    return j;
}

}  // namespace nccl
