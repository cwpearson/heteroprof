
#include "nccl/nccl_bcast.hpp"


namespace nccl {

using json = nlohmann::json;

NcclBcast::NcclBcast(const Nccl &api, void *buff, int count,
                     ncclDataType_t datatype, int root,
                     ncclComm_t comm, cudaStream_t stream)
    : Api(api), comm_(comm), buff_(buff) {


        int currentRank;
        auto result = ncclCommUserRank(comm, &currentRank);

        if (currentRank == root){
            std::vector<uint64_t> input_vector {
                                            (uint64_t)buff_
                                           };
            std::vector<uint64_t> output_vector;
            set_nccl_inputs(input_vector);
            set_nccl_outputs(output_vector);
        } else {
            std::vector<uint64_t> input_vector;
            std::vector<uint64_t> output_vector {
                                                 (uint64_t)buff_
                                                };
            set_nccl_inputs(input_vector);
            set_nccl_outputs(output_vector);
        }
    }
}  // namespace nccl
