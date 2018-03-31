#ifndef CUDA_NCCL_COMM_INIT_ALL
#define CUDA_NCCL_COMM_INIT_ALL

#include "nccl/api.hpp"
#include <vector>;


namespace nccl {

class NcclCommInitAll : public nccl::Nccl {
  using json = nlohmann::json;
  using Api = nccl::Nccl;
  using tid_t = sys::tid_t;

protected:
  ncclComm_t *comm_;
  int nGPUs_;
  const int *devList_;

public:
  NcclCommInitAll(const Api &api, ncclComm_t *comms, int nGPUs,
                  const int *devList);
  virtual json to_json() const override;
  virtual std::vector<json> to_json_vector() const override;

private:
  void fill_in_handles();
  json make_handle_json(ncclComm_t comms, int cur_gpu);

  std::vector<json> handle_json_;


};

} // namespace nccl


#endif