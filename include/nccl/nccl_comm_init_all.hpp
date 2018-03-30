#ifndef CUDA_NCCL_COMM_INIT_ALL
#define CUDA_NCCL_COMM_INIT_ALL

#include "nccl/api.hpp"


namespace nccl {

class NcclCommInitAll : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t *comm_;
  int nGPUs_;

public:
  NcclCommInitAll(const Api &api, ncclComm_t *comms, int nGPUs,
                  const int *devList);

  //Disable for now
  // virtual json to_json() const override;
};

} // namespace cudnn


#endif