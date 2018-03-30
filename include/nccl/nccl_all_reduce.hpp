#ifndef CUDA_NCCL_ALL_REDUCE
#define CUDA_NCCL_ALL_REDUCE

#include "nccl/api.hpp"
#include <vector>


namespace nccl {

class NcclAllReduce : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t comm_;
  const void *sendbuff_;
  const void *recvbuff_;

public:
  NcclAllReduce(const Api &api, const void *sendbuff, void *recvbuff,
                int count, ncclDataType_t datatype, ncclRedOp_t op, 
                ncclComm_t comm, cudaStream_t stream);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif