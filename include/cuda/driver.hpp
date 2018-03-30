#ifndef CUDA_DRIVER_STATE_HPP
#define CUDA_DRIVER_STATE_HPP

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cupti.h>
#include <nccl.h>

#include "cuda/api.hpp"
#include "cuda/configured_call.hpp"
#include "cuda/location.hpp"
#include "sys/thread.hpp"
#include "util_numa.hpp"

namespace cuda {

class ThreadState {

  using Api = cuda::Api;
  using ConfiguredCall = cuda::ConfiguredCall;

public:
  typedef std::shared_ptr<Api> ApiRef;

private:
  int currentDevice_;
  bool cuptiCallbacksEnabled_;
  std::vector<CUcontext> contextStack_;

  ConfiguredCall configuredCall_;
  std::vector<ApiRef> apiStack_;

public:
  ThreadState() : currentDevice_(0), cuptiCallbacksEnabled_(true) {}

  int current_device() const { return currentDevice_; }
  void set_device(const int device) { currentDevice_ = device; }

  void api_enter(ApiRef a);
  ApiRef api_exit();

  size_t api_stack_size() const { return apiStack_.size(); }
  const ApiRef parent_api() const;
  ApiRef current_api();

  void pause_cupti_callbacks();
  void resume_cupti_callbacks();

  void push_context(const CUcontext c) { contextStack_.push_back(c); }
  void pop_context() {
    assert(!contextStack_.empty());
    contextStack_.resize(contextStack_.size() - 1);
  }
  void set_context(const CUcontext c) {
    if (c == 0) {
      if (!contextStack_.empty()) {
        pop_context();
      }
    } else if (!contextStack_.empty()) {
      contextStack_[contextStack_.size() - 1] = c;
    } else {
      push_context(c); // FIXME: not clear if this is right from docs
    }
  }
  CUcontext current_context() const {
    assert(!contextStack_.empty());
    return contextStack_.back();
  }

  bool is_cupti_callbacks_enabled() const { return cuptiCallbacksEnabled_; }

  ConfiguredCall &configured_call() { return configuredCall_; }
};

class Driver {
public:
  typedef ThreadState mapped_type;
  typedef sys::tid_t key_type;
  typedef std::pair<key_type, mapped_type> value_type;

private:
  typedef std::map<key_type, mapped_type> ThreadMap;
  ThreadMap threadStates_;
#if false
  std::map<const cublasHandle_t, int> cublasHandleToDevice_;
  std::map<const cudnnHandle_t, int> cudnnHandleToDevice_;
  std::map<const ncclComm_t, int> ncclCommToDevice_;
#endif
  std::mutex access_mutex_;

public:
#if false
  void track_cublas_handle(const cublasHandle_t h, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    cublasHandleToDevice_[h] = device;
  }
  void track_cudnn_handle(const cudnnHandle_t h, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    cudnnHandleToDevice_[h] = device;
  }
  void register_ncclComm(const ncclComm_t c, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    ncclCommToDevice_[c] = device;
  }

  int device_from_cublas_handle(const cublasHandle_t h) {
    std::stringstream ss;
    ss << "DEBU: looking for cublas handle " << h << std::endl;
    logging::atomic_err(ss.str());
    return cublasHandleToDevice_.at(h);
  }

  int device_from_cudnn_handle(const cudnnHandle_t h) {
    std::stringstream ss;
    ss << "DEBU: looking for cudnn handle " << h << std::endl;
    logging::atomic_err(ss.str());
    return cudnnHandleToDevice_.at(h);
  }
#endif
  int device(const ncclComm_t c) {
    std::stringstream ss;
    const int dev = ncclCommToDevice_.at(c);
    ss << "DEBU: (tid= " << get_thread_id() << ") found nccl comm" << c
       << "on device " << dev << std::endl;
    logging::atomic_err(ss.str());
    return dev;
  }

  mapped_type &this_thread() { return threadStates_[sys::get_thread_id()]; }
};

} // namespace cuda

#endif
