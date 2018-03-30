
#include "nccl/nccl_comm_destroy.hpp"


namespace nccl {

using json = nlohmann::json;

NcclCommDestroy::NcclCommDestroy(const Nccl &api, ncclComm_t comm)
    : Api(api), comm_(comm) {
        // device_ = profiler().driver().this_thread().device(comm_);
        // handle_ = (uintptr_t)cublasHandle_;
    }


}  // namespace nccl
