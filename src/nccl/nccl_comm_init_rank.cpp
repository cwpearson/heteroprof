
#include "nccl/nccl_comm_init_rank.hpp"


namespace nccl {

using json = nlohmann::json;

NcclCommInitRank::NcclCommInitRank(const Nccl &api, ncclComm_t *comm, int ndev,
                                   ncclUniqueId cliqueId, int rank)
    : Api(api), comm_(comm) {
        // device_ = profiler().driver().this_thread().device(comm_);
        // handle_ = (uintptr_t)cublasHandle_;
    }

    std::vector<json> NcclCommInitRank::to_json_vector() const {
        return handle_json_;
    }   

}  // namespace nccl
