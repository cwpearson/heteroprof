#include <cassert>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <nccl.h>

#include "profiler.hpp"
#include "cuda/driver.hpp"
#include "nccl/api.hpp"
#include "nccl/nccl_comm_destroy.hpp"
#include "nccl/nccl_all_reduce.hpp"

namespace nccl {
using sys::get_thread_id;    
Profiler *profiler_ = nullptr;
void set_profiler(Profiler &p) { profiler_ = &p; }
Profiler &profiler() {
  assert(profiler_);
  return *profiler_;
}

Nccl make_nccl_this_thread_now(std::string name) {
  auto now = std::chrono::high_resolution_clock::now();
  auto tid = get_thread_id();
  Nccl nccl(tid, name);
  nccl.set_wall_start(now);
  return nccl;
}

void finalize_api(Profiler &p){
    auto api = p.driver().this_thread().current_api();
    api->set_wall_end(std::chrono::high_resolution_clock::now());
    p.record(api->to_json());
    p.driver().this_thread().api_exit(); 
}
} // namespace nccl


using namespace nccl;

static void register_ncclBcast(uintptr_t buff, size_t count,
                               ncclDataType_t datatype, int root,
                               ncclComm_t comm) {

  // static std::mutex access;
  // // static Value rootBuffVal = Value();
  // // static std::vector<Value> dstBuffVals;
  // // const int dev = profiler::driver().device(comm);
  // // const auto &dstAS = profiler::hardware().address_space(dev);
  // const size_t numBytes = count * ncclSizeOf(datatype);

  // // Only one thread should proceed at a time from here
  // std::lock_guard<std::mutex> guard(access);

  // auto dstBuffAlloc = profiler::allocations().find(buff, numBytes, dstAS);
  // const auto &dstBuffVal = dstBuffAlloc.new_value(buff, numBytes, true);
  // assert(dstBuffVal);
  // dstBuffVals.push_back(dstBuffVal);

  // // Have the last thread set the deps and create the api call
  // int commSize;
  // ncclResult_t res = ncclCommCount(comm, &commSize);
  // if (res != ncclSuccess) {
  //   assert(0);
  // }
  // if (unsigned(commSize) == dstBuffVals.size()) {

  //   auto api = std::make_shared<ApiRecord>("ncclBcast", dev);

  //   // Find the root's buffer value and add it
  //   const auto &rootDevAS = profiler::hardware().address_space(root);
  //   rootBuffVal = profiler::allocations().find_value(buff, numBytes, rootDevAS);
  //   api->add_input(rootBuffVal);

  //   for (auto &v : dstBuffVals) {
  //     if (dstBuffVal != rootBuffVal) {
  //       api->add_output(v);
  //     }
  //   }
  //   profiler::atomic_out(api->to_json_string() + "\n");
  //   dstBuffVals.clear();
  //   rootBuffVal = Value();
  // }
}

static void register_ncclAllReduce(const uintptr_t sendbuff,
                                   const uintptr_t recvbuff, size_t count,
                                   ncclDataType_t datatype, ncclComm_t comm) {
  // static std::mutex access;
  // static std::vector<Value> sendBuffVals, recvBuffVals;
  // const int dev = profiler::driver().device(comm);
  // const auto &AS = profiler::hardware().address_space(dev);

  // // Only one thread should proceed at a time from here
  // std::lock_guard<std::mutex> guard(access);

  // // Look up and add my values
  // const size_t numBytes = ncclSizeOf(datatype) * count;
  // const auto sendBuffVal =
  //     profiler::allocations().find_value(sendbuff, numBytes, AS);
  // sendBuffVals.push_back(sendBuffVal);

  // auto recvBuffAlloc = profiler::allocations().find(recvbuff, numBytes, AS);
  // auto recvBuffVal = recvBuffAlloc.new_value(recvbuff, numBytes, true);
  // recvBuffVals.push_back(recvBuffVal);

  // // Once all values have been found, the last thread to enter allreduce can
  // // set up deps
  // assert(sendBuffVals.size() == recvBuffVals.size());
  // int commSize;
  // ncclResult_t res = ncclCommCount(comm, &commSize);
  // if (res != ncclSuccess) {
  //   assert(0);
  // }
  // if (unsigned(commSize) == sendBuffVals.size()) {

  //   auto api = std::make_shared<ApiRecord>("ncclAllReduce",
  //                                          profiler::driver().device(comm));

  //   for (const auto &sendVal : sendBuffVals) {
  //     api->add_input(sendVal);
  //   }
  //   for (const auto &v : recvBuffVals) {
  //     api->add_output(v);
  //   }
    
  //   sendBuffVals.clear();
  //   recvBuffVals.clear();
  // }
}

#define NCCL_DLSYM_BOILERPLATE(name)                                           \
  static name##Func real_##name = nullptr;                                     \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libnccl.so", RTLD_LAZY);                               \
      real_##name = (name##Func)dlsym(h, #name);                               \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");


typedef ncclResult_t (*ncclCommInitAllFunc)(ncclComm_t *comms, int nGPUs,
                                            const int *devList);
extern "C" ncclResult_t ncclCommInitAll(ncclComm_t *comms, int nGPUs,
                                        const int *devList) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitAll);

  const ncclResult_t ret = real_ncclCommInitAll(comms, nGPUs, devList);

  for (int i = 0; i < nGPUs; ++i) {
    const int dev = devList ? devList[i] : i;
    profiler::driver().register_ncclComm(comms[i], dev);
  }

  return ret;
}

typedef ncclResult_t (*ncclCommInitRankFunc)(ncclComm_t *comm, int ndev,
                                             ncclUniqueId cliqueId, int rank);
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t *comm, int ndev,
                                         ncclUniqueId cliqueId, int rank) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitRank);


  const ncclResult_t ret = real_ncclCommInitRank(comm, ndev, cliqueId, rank);
  profiler::driver().register_ncclComm(
      *comm, profiler::driver().this_thread().current_device());

  return ret;
}


typedef ncclResult_t (*ncclBcastFunc)(void *buff, int count,
                                      ncclDataType_t datatype, int root,
                                      ncclComm_t comm, cudaStream_t stream);
extern "C" ncclResult_t ncclBcast(void *buff, int count,
                                  ncclDataType_t datatype, int root,
                                  ncclComm_t comm, cudaStream_t stream) {

  NCCL_DLSYM_BOILERPLATE(ncclBcast);

  register_ncclBcast(uintptr_t(buff), count, datatype, root, comm);


  const ncclResult_t ret =
      real_ncclBcast(buff, count, datatype, root, comm, stream);


  return ret;
}

typedef ncclResult_t (*ncclAllReduceFunc)(const void *sendbuff, void *recvbuff,
                                          int count, ncclDataType_t datatype,
                                          ncclRedOp_t op, ncclComm_t comm,
                                          cudaStream_t stream);

extern "C" ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff,
                                      int count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm_t comm,
                                      cudaStream_t stream) {

  NCCL_DLSYM_BOILERPLATE(ncclAllReduce);

  register_ncclAllReduce(uintptr_t(sendbuff), uintptr_t(recvbuff), count,
                         datatype, comm);

  const ncclResult_t ret =
      real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);

  return ret;
}

//Should this be nccl?
typedef ncclResult_t (*ncclCommDestroyFunc)(ncclComm_t comm);
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  NCCL_DLSYM_BOILERPLATE(ncclCommDestroy);


  auto a = make_nccl_this_thread_now("nccCommDestroy");
  auto api = std::make_shared<NcclCommDestroy>(a, handle, n,
                                           x, incx, result);
  profiler().driver().this_thread().api_enter(api);

  const ncclResult_t ret = real_ncclCommDestroy(comm);
  finalize_api(profiler());

  return ret;
}