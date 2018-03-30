
#include "nccl/nccl_comm_init_rank.hpp"


namespace nccl {

using json = nlohmann::json;

NcclCommInitRank::NcclCommInitRank(const Nccl &api, ncclComm_t comm)
    : Api(api), comm_(comm) {
        // device_ = profiler().driver().this_thread().device(comm_);
        // handle_ = (uintptr_t)cublasHandle_;
    }


}  // namespace nccl
