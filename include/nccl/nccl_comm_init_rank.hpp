#ifndef CUDA_NCCL_COMM_INIT_RANK
#define CUDA_NCCL_COMM_INIT_RANK

#include "nccl/api.hpp"


namespace nccl {

class NcclCommInitRank : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t comm_;

public:
  NcclCommInitRank(const Api &api, const ncclComm_t comm);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif