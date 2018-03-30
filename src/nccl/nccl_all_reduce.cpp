
#include "nccl/nccl_all_reduce.hpp"


namespace nccl {

using json = nlohmann::json;

NcclAllReduce::NcclAllReduce(const Nccl &api, ncclComm_t comm)
    : Api(api), comm_(comm) {
        // device_ = profiler().driver().this_thread().device(comm_);
    }


}  // namespace nccl
