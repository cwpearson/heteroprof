
#include "nccl/nccl_all_reduce.hpp"


namespace nccl {

using json = nlohmann::json;

NcclAllReduce::NcclAllReduce(const Api &api, const void *sendbuff, void *recvbuff,
                             int count, ncclDataType_t datatype, ncclRedOp_t op, 
                             ncclComm_t comm, cudaStream_t stream)
    : Api(api), comm_(comm), sendbuff_(sendbuff), recvbuff_(recvbuff) {
        std::vector<uint64_t> input_vector {
                                            (uint64_t)sendbuff_ 
                                           };
        std::vector<uint64_t> output_vector {
                                             (uint64_t)recvbuff_
                                            };
        set_nccl_inputs(input_vector);
        set_nccl_outputs(output_vector);
    }


}  // namespace nccl
