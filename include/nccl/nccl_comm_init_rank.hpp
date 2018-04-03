#ifndef CUDA_NCCL_COMM_INIT_RANK
#define CUDA_NCCL_COMM_INIT_RANK

#include "nccl/api.hpp"


namespace nccl {

class NcclCommInitRank : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t *comm_;
  std::vector<json> handle_json_;
  int ndev_;
  ncclUniqueId cliqueId_;
  int rank_;
  void fill_in_handles();
  json make_handle_json(ncclComm_t comms, int cur_gpu);

public:
  NcclCommInitRank(const Api &api, ncclComm_t *comm, int ndev,
                   ncclUniqueId cliqueId, int rank);

  virtual std::vector<json> to_json_vector() const override;
};

} // namespace cudnn


#endif