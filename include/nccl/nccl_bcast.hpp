#ifndef CUDA_NCCL_BCAST
#define CUDA_NCCL_BCAST

#include "nccl/api.hpp"
#include <vector>


namespace nccl {

class NcclBcast : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t comm_;
  void *buff_;

public:
  NcclBcast(const Api &api, void *buff, int count,
            ncclDataType_t datatype, int root,
            ncclComm_t comm, cudaStream_t stream);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif