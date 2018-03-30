#ifndef CUDA_NCCL_ALL_REDUCE
#define CUDA_NCCL_ALL_REDUCE

#include "nccl/api.hpp"


namespace nccl {

class NcclAllReduce : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t comm_;

public:
  NcclAllReduce(const Api &api, const ncclComm_t comm);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif