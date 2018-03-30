
#include "nccl/nccl_comm_init_all.hpp"


namespace nccl {

using json = nlohmann::json;

NcclCommInitAll::NcclCommInitAll(const Nccl &api, ncclComm_t *comms, int nGPUs,
                                 const int *devList)
    : Api(api), comm_(comms), nGPUs_(nGPUs) {
        // device_ = profiler().driver().this_thread().device(comm_);
        // handle_ = (uintptr_t)cublasHandle_;
    }


}  // namespace nccl
