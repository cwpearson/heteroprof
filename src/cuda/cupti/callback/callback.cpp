#include <cassert>

#include <cuda_runtime_api.h>
#include <cupti.h>

#include "cuda/cupti/callback/api.hpp"
#include "cuda/cupti/callback/callback.hpp"
#include "cuda/cupti/callback/config.hpp"
#include "cuda/location.hpp"
#include "cuda/memory.hpp"
#include "profiler.hpp"

// supported APIs
#include "cuda/cupti/callback/cu_mem_host_alloc.hpp"
#include "cuda/cupti/callback/cuda_configure_call.hpp"
#include "cuda/cupti/callback/cuda_malloc.hpp"
#include "cuda/cupti/callback/cuda_set_device.hpp"
#include "cuda/cupti/callback/cuda_setup_argument.hpp"
#include "cuda/cupti/callback/cuda_stream_create.hpp"

using cuda::Location;
using cuda::Memory;

using namespace cuda::cupti::callback;
using namespace cuda::cupti::callback::config;
using sys::get_thread_id;

void finalize_api(Profiler &p) {
  auto api = p.driver().this_thread().current_api();
  api->set_wall_end(std::chrono::high_resolution_clock::now());
  p.record(api->to_json());
  p.driver().this_thread().api_exit();
}

static void handleCudaLaunch(const CUpti_CallbackData *cbdata,
                             Profiler &profiler) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = reinterpret_cast<const cudaLaunch_v3020_params *>(
        cbdata->functionParams);
    auto func = params->func;

    auto now = std::chrono::high_resolution_clock::now();
    auto tid = get_thread_id();

    auto api = std::make_shared<CudaLaunch>(tid, cbdata, func);
    api->set_wall_start(now);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaLaunchKernel(const CUpti_CallbackData *cbdata,
                                   Profiler &profiler) {
  profiler::log() << "INFO: callback: cudaLaunchKernel preamble (tid="
                  << cprof::model::get_thread_id() << ")" << std::endl;

  auto params = ((cudaLaunchKernel_v7000_params *)(cbdata->functionParams));
  const void *func = params->func;
  profiler::log() << "launching " << func << std::endl;
  const dim3 gridDim = params->gridDim;
  const dim3 blockDim = params->blockDim;
  void *const *args = params->args;
  const size_t sharedMem = params->sharedMem;
  const cudaStream_t stream = params->stream;

  // print_backtrace();

  const char *symbolName = cbdata->symbolName;
  // const char *symbolName = (char*)func;

  // assert(0 && "Unimplemented");

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    // Some NCCL calls reach here and do not have a symbol name.
    // Sanity check to prevent crash.
    if (cbdata->symbolName != NULL) {
      auto api = std::make_shared<ApiRecord>(
          cbdata->functionName, cbdata->symbolName,
          profiler::driver().this_thread().current_device());

      profiler::atomic_out(api->to_json_string() + "\n");
    }

  } else {
    assert(0 && "How did we get here?");
  }

  profiler::log() << "callback: cudaLaunchKernel: done" << std::endl;
}

static void handleCudaMemcpy(const CUpti_CallbackData *cbdata,
                             Profiler &profiler) {

  // extract API call parameters
  auto params = ((cudaMemcpy_v3020_params *)(cbdata->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const cudaMemcpyKind kind = params->kind;
  const size_t count = params->count;
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: callback: cudaMemcpy enter" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    uint64_t endTimeStamp;
    cuptiDeviceGetTimestamp(cbdata->context, &endTimeStamp);
    // profiler::log() << "The end timestamp is " << endTimeStamp <<
    // std::endl; std::cout << "The end time is " << cbdata->end_time;
    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    record_memcpy(cbdata, allocations, api, dst, src, MemoryCopyKind(kind),
                  count, count, 0 /*unused*/, 0 /*unused */);
    profiler::log() << "INFO: callback: cudaMemcpy exit" << std::endl;

  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyAsync(const CUpti_CallbackData *cbdata,
                                  Profiler &profiler) {
  // extract API call parameters
  auto params = ((cudaMemcpyAsync_v3020_params *)(cbdata->functionParams));
  const uintptr_t dst = (uintptr_t)params->ds auto api =
      profiler.driver().this_thread().current_api();
  t;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t count = params->count;
  const cudaMemcpyKind kind = params->kind;
  // const cudaStream_t stream = params->stream;
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaMemcpyAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    record_memcpy(cbdata, allocations, api, dst, src, MemoryCopyKind(kind),
                  count, count, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpy2DAsync(const CUpti_CallbackData *cbdata, ,
                                    Profiler &profiler) {
  // extract API call parameters
  auto params = ((cudaMemcpy2DAsync_v3020_params *)(cbdata->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const size_t dpitch = params->dpitch;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t spitch = params->spitch;
  // const size_t width = params->width;
  const size_t height = params->height;
  const cudaMemcpyKind kind = params->kind;
  // const cudaStream_t stream = params->stream;
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaMemcpy2DAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    const size_t srcCount = height * spitch;
    const size_t dstCount = height * dpitch;
    record_memcpy(cbdata, allocations, api, dst, src, MemoryCopyKind(kind),
                  srcCount, dstCount, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyPeerAsync(const CUpti_CallbackData *cbdata,
                                      Profiler &profiler) {
  // extract API call parameters
  auto params = ((cudaMemcpyPeerAsync_v4000_params *)(cbdata->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const int dstDevice = params->dstDevice;
  const uintptr_t src = (uintptr_t)params->src;
  const int srcDevice = params->srcDevice;
  const size_t count = params->count;
  // const cudaStream_t stream = params->stream;
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaMemcpyPeerAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbdata);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    record_memcpy(cbdata, allocations, api, dst, src,
                  MemoryCopyKind::CudaPeer(), count, count, srcDevice,
                  dstDevice);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocManaged(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler) {
  auto params = ((cudaMallocManaged_v6000_params *)(cbdata->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  // const unsigned int flags = params->flags;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {

    profiler::log() << "INFO: [cudaMallocManaged] " << devPtr << "[" << size
                    << "]" << std::endl;

    // Create the new allocation

    const int devId = profiler::driver().this_thread().current_device();
    const int major = profiler::hardware().cuda_device(devId).major_;
    assert(major >= 3 && "cudaMallocManaged unsupported on major < 3");
    cprof::model::Memory M;
    if (major >= 6) {
      M = cprof::model::Memory::Unified6;
    } else if (major >= 3) {
      M = cprof::model::Memory::Unified3;
    } else {
      assert(0 && "How to handle?");
    }

    auto a = allocations.new_allocation(devPtr, size, AddressSpace::CudaUVA(),
                                        M, Location::Unknown());
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocHost(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMallocHost_v3020_params *)(cbdata->functionParams));
    uintptr_t ptr = (uintptr_t)(*(params->ptr));
    const size_t size = params->size;
    profiler::log() << "INFO: [cudaMallocHost] " << ptr << "[" << size << "]"
                    << std::endl;

    if ((uintptr_t) nullptr == ptr) {
      profiler::log()
          << "WARN: ignoring cudaMallocHost call that returned nullptr"
          << std::endl;
      return;
    }

    record_mallochost(allocations, ptr, size);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuLaunchKernel(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler) {
  (void)allocations;

  auto &ts = profiler::driver().this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
      (ts.parent_api()->cbid() == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
       ts.parent_api()->cbid() ==
           CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    profiler::log() << "WARN: skipping cuLaunchKernel inside cudaLaunch or "
                       "cudaLaunchKernel"
                    << std::endl;
    return;
  }

  assert(0 && "unhandled cuLaunchKernel outside of cudaLaunch!");
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: enter cuLaunchKernel" << std::endl;
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: exit cuLaunchKernel" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

#if false
static void handleCuModuleGetFunction(const CUpti_CallbackData *cbdata) {

  auto params = ((cuModuleGetFunction_params *)(cbdata->functionParams));
  const CUfunction hfunc = *(params->hfunc);
  // const CUmodule hmod = params->hmod;
  const char *name = params->name;

  profiler::log() << "INFO: cuModuleGetFunction for " << name << " @ " << hfunc
                  << std::endl;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: enter cuModuleGetFunction" << std::endl;
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: exit cuModuleGetFunction" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCuModuleGetGlobal_v2(const CUpti_CallbackData *cbdata) {

  // auto params = ((cuModuleGetGlobal_v2_params *)(cbdata->functionParams));

  // const CUdeviceptr dptr = *(params->dptr);
  // assert(params->bytes);
  // const size_t bytes = *(params->bytes);
  // const CUmodule hmod = params->hmod;
  // const char *name = params->name;

  // profiler::log() << "INFO: cuModuleGetGlobal_v2 for " << name << " @ " <<
  // dptr
  // << std::endl;
  profiler::log() << "WARN: ignoring cuModuleGetGlobal_v2" << std::endl;
  return;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: enter cuModuleGetGlobal_v2" << std::endl;
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: exit cuModuleGetGlobal_v2" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCuCtxSetCurrent(const CUpti_CallbackData *cbdata) {


  auto params = ((cuCtxSetCurrent_params *)(cbdata->functionParams));
  const CUcontext ctx = params->ctx;
  const int pid = cprof::model::get_thread_id();

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: (tid=" << pid << ") setting ctx " << ctx
                    << std::endl;
    profiler::driver().this_thread().set_context(ctx);
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

static void handleCudaFreeHost(const CUpti_CallbackData *cbdata,
                               Profiler &profiler) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaFreeHost_v3020_params *)(cbdata->functionParams));
    uintptr_t ptr = (uintptr_t)(params->ptr);
    cudaError_t ret = *static_cast<cudaError_t *>(cbdata->functionReturnValue);
    profiler::log() << "INFO: [cudaFreeHost] " << ptr << std::endl;
    if (ret != cudaSuccess) {
      profiler::log() << "WARN: unsuccessful cudaFreeHost: "
                      << cudaGetErrorString(ret) << std::endl;
    }
    assert(cudaSuccess == ret);
    assert(ptr &&
           "Must have been initialized by cudaMallocHost or cudaHostAlloc");

    const int devId = profiler::driver().this_thread().current_device();
    auto AS = profiler::hardware().address_space(devId);

    auto numFreed = allocations.free(ptr, AS);
    // Issue
    // assert(numFreed && "Freeing unallocated memory?");
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFree(const CUpti_CallbackData *cbdata,
                           Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaFree_v3020_params *)(cbdata->functionParams));
    auto devPtr = (uintptr_t)params->devPtr;
    cudaError_t ret = *static_cast<cudaError_t *>(cbdata->functionReturnValue);
    profiler::log() << "INFO: (tid=" << cprof::model::get_thread_id()
                    << ") [cudaFree] " << devPtr << std::endl;

    assert(cudaSuccess == ret);

    if (!devPtr) { // does nothing if passed 0
      profiler::log() << "WARN: cudaFree called on 0? Does nothing."
                      << std::endl;
      return;
    }

    const int devId = profiler::driver().this_thread().current_device();
    auto AS = profiler::hardware().address_space(devId);

    // Find the live matching allocation
    profiler::log() << "Looking for " << devPtr << std::endl;
    auto freeAlloc = allocations.free(devPtr, AS);
    // assert(freeAlloc && "Freeing unallocated memory?");
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaConfigureCall(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = ((cudaConfigureCall_v3020_params *)(cbdata->functionParams));
    auto gridDim = params->gridDim;
    auto blockDim = params->blockDim;
    auto sharedMem = params->sharedMem;
    auto stream = params->stream;

    auto now = std::chrono::high_resolution_clock::now();
    auto tid = get_thread_id();

    auto api = std::make_shared<CudaConfigureCall>(tid, cbdata, gridDim,
                                                   blockDim, sharedMem, stream);
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMalloc(const CUpti_CallbackData *cbdata,
                             Profiler &profiler) {
  const auto params = ((cudaMalloc_v3020_params *)(cbdata->functionParams));

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    auto tid = get_thread_id();
    auto api = std::make_shared<CudaMalloc>(tid, cbdata, size);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<CudaMalloc>(api)) {
      const void *const *devPtr = params->devPtr;
      cm->set_devptr(devPtr);
      cm->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(cm->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected CudaMalloc");
    }

  } else {
    assert(0 && "how did we get here?");
  }
}

static void handleCudaSetDevice(const CUpti_CallbackData *cbdata,
                                Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto params = ((cudaSetDevice_v3020_params *)(cbdata->functionParams));
    const int device = params->device;

    auto tid = get_thread_id();
    auto api = std::make_shared<CudaSetDevice>(tid, cbdata, device);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetupArgument(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbdata->functionParams));
    const void *arg = params->arg;
    size_t size = params->size;
    size_t offset = params->offset;

    auto tid = get_thread_id();
    auto api =
        std::make_shared<CudaSetupArgument>(tid, cbdata, arg, size, offset);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamCreate(const CUpti_CallbackData *cbdata,
                                   Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto tid = get_thread_id();
    auto api = std::make_shared<CudaStreamCreate>(tid, cbdata);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();
    if (auto csc = std::dynamic_pointer_cast<CudaStreamCreate>(api)) {
      const auto params =
          ((cudaStreamCreate_v3020_params *)(cbdata->functionParams));
      const cudaStream_t stream = *(params->pStream);

      csc->set_stream(stream);
      csc->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(csc->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected CudaMalloc");
    }

  } else {
    assert(0 && "How did we get here?");
  }
}

#if false
static void handleCudaStreamDestroy(const CUpti_CallbackData *cbdata) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    profiler::log() << "WARN: ignoring cudaStreamDestroy" << std::endl;
    // const auto params =
    //     ((cudaStreamDestroy_v3020_params *)(cbdata->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaStreamSynchronize(const CUpti_CallbackData *cbdata) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: callback: cudaStreamSynchronize entry"
                    << std::endl;
    profiler::log() << "WARN: ignoring cudaStreamSynchronize" << std::endl;
    // const auto params =
    //     ((cudaStreamSynchronize_v3020_params *)(cbdata->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

static void handleCuMemHostAlloc(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler) {

  auto params = ((cuMemHostAlloc_params *)(cbdata->functionParams));

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    const size_t bytesize = params->bytesize;
    const int Flags = params->Flags;

    auto tid = get_thread_id();
    auto api = std::make_shared<CuMemHostAlloc>(tid, cbdata, bytesize, Flags);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {

    const void *pp = (*(params->pp));

    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<CuMemHostAlloc>(api)) {
      const void *const *pp = params->pp;
      cm->set_pp(pp);
      cm->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(cm->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected CuMemHostAlloc");
    }

  } else {
    assert(0 && "How did we get here?");
  }
}

void CUPTIAPI cuptiCallbackFunction(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    CUpti_CallbackData *cbdata) {

  (void)userdata; // data supplied at subscription

  auto &profiler = cuda::cupti::callback::config::profiler();

  if (!profiler.driver().this_thread().is_cupti_callbacks_enabled()) {
    return;
  }

  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    // case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
    // handleCudaMemcpy(cbdata, profiler);
    //   break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
    // handleCudaMemcpyAsync(profiler::allocations(), cbdata);
    //   break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
    // handleCudaMemcpyPeerAsync(profiler::allocations(), cbdata);
    //   break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleCudaMalloc(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
      handleCudaMallocHost(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
      handleCudaMallocManaged(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      handleCudaFree(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
      handleCudaFreeHost(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
      handleCudaConfigureCall(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
      handleCudaSetupArgument(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      handleCudaLaunch(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020:
      handleCudaSetDevice(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
      handleCudaStreamCreate(cbdata, profiler);
      break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
    //   handleCudaStreamDestroy(cbdata);
    //   break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
    //   handleCudaStreamSynchronize(cbdata);
    //   break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
    //   handleCudaMemcpy2DAsync(profiler::allocations(), cbdata);
    //   break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
    //   handleCudaLaunchKernel(userdata, profiler::allocations(), cbdata);
    //   break;
    default:
      profiler.log() << "DEBU: ( tid= " << get_thread_id()
                     << " ) skipping runtime call " << cbdata->functionName
                     << std::endl;
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(cbdata, profiler);
      break;
    // case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    // handleCuLaunchKernel(profiler::allocations(), cbdata);
    //   break;
    // case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
    // handleCuModuleGetFunction(cbdata);
    //   break;
    // case CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2:
    // handleCuModuleGetGlobal_v2(cbdata);
    //   break;
    // case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
    // handleCuCtxSetCurrent(cbdata);
    // break;
    default:
      profiler.log() << "DEBU: ( tid= " << get_thread_id()
                     << " ) skipping driver call " << cbdata->functionName
                     << std::endl;
      break;
    }
  }
  default:
    break;
  }
}
