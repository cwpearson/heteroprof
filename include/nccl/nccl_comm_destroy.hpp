#ifndef CUDA_NCCL_COMM_DESTROY
#define CUDA_NCCL_COMM_DESTROY

#include "nccl/api.hpp"


namespace nccl {

class NcclCommDestroy : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t comm_;

public:
  NcclCommDestroy(const Api &api, const ncclComm_t comm);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif