#include <cassert>
#include <iostream>

#include <cuda_runtime_api.h>
#include <cupti.h>

#include "cuda/cupti/callback/api.hpp"
#include "cuda/cupti/callback/callback.hpp"
#include "cuda/cupti/callback/config.hpp"
#include "profiler.hpp"

// supported APIs
#include "cuda/cupti/callback/cu_ctx_set_current.hpp"
#include "cuda/cupti/callback/cuda_configure_call.hpp"
#include "cuda/cupti/callback/cuda_launch.hpp"
#include "cuda/cupti/callback/cuda_set_device.hpp"
#include "cuda/cupti/callback/cuda_setup_argument.hpp"
#include "cuda/cupti/callback/cuda_stream_create.hpp"
#include "cuda/cupti/callback/cuda_stream_destroy.hpp"
#include "cuda/cupti/callback/device_alloc.hpp"
#include "cuda/cupti/callback/device_free.hpp"
#include "cuda/cupti/callback/host_alloc.hpp"
#include "cuda/cupti/callback/host_free.hpp"
#include "cuda/cupti/callback/managed_alloc.hpp"
#include "cuda/cupti/callback/memcpy.hpp"

using namespace cuda::cupti::callback;
using namespace cuda::cupti::callback::config;
using sys::get_thread_id;

// bool template <typename A, typename B>

Api make_api_this_thread_now(const CUpti_CallbackData *cbdata,
                             const CUpti_CallbackDomain domain) {
  auto now = std::chrono::high_resolution_clock::now();
  auto tid = get_thread_id();
  Api api(tid, cbdata, domain);
  api.set_wall_start(now);
  return api;
}

void finalize_api(Profiler &p) {
  auto api = p.driver().this_thread().current_api();
  api->set_wall_end(std::chrono::high_resolution_clock::now());
  p.record(api->to_json());
  p.driver().this_thread().api_exit();
}

static void handleCudaConfigureCall(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler,
                                    const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = ((cudaConfigureCall_v3020_params *)(cbdata->functionParams));
    auto gridDim = params->gridDim;
    auto blockDim = params->blockDim;
    auto sharedMem = params->sharedMem;
    auto stream = params->stream;

    auto a = make_api_this_thread_now(cbdata, domain);
    auto api = std::make_shared<CudaConfigureCall>(a, gridDim, blockDim,
                                                   sharedMem, stream);
    profiler.driver().this_thread().api_enter(api);
    profiler.driver().this_thread().configured_call().start();

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFreeHost(const CUpti_CallbackData *cbdata,
                               Profiler &profiler,
                               const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto params = ((cudaFreeHost_v3020_params *)(cbdata->functionParams));
    const void *ptr = params->ptr;

    const auto api = make_api_this_thread_now(cbdata, domain);
    auto hf = std::make_shared<HostFree>(api, ptr);
    profiler.driver().this_thread().api_enter(hf);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFree(const CUpti_CallbackData *cbdata, Profiler &profiler,
                           const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto params =
        reinterpret_cast<const cudaFree_v3020_params *>(cbdata->functionParams);
    const void *devPtr = params->devPtr;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto df = std::make_shared<DeviceFree>(api, devPtr);
    profiler.driver().this_thread().api_enter(df);
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaHostAlloc(const CUpti_CallbackData *cbdata,
                                Profiler &profiler,
                                const CUpti_CallbackDomain domain) {
  auto params = reinterpret_cast<const cudaHostAlloc_v3020_params *>(
      cbdata->functionParams);

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    unsigned int flags = params->flags;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto ha =
        std::make_shared<HostAlloc>(api, size, flags, 0 /*no driver flags */);
    profiler.driver().this_thread().api_enter(ha);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<HostAlloc>(api)) {
      const void *const *ptr = params->pHost;
      cm->set_ptr(*ptr);
      cm->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(cm->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected HostAlloc");
    }
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaLaunch(const CUpti_CallbackData *cbdata,
                             Profiler &profiler,
                             const CUpti_CallbackDomain domain) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    profiler.log() << "entering cudaLaunch\n";

    auto params = reinterpret_cast<const cudaLaunch_v3020_params *>(
        cbdata->functionParams);
    auto func = params->func;

    std::vector<CudaLaunchParams> launchParams;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto cl = std::make_shared<CudaLaunch>(api, func, launchParams);
    profiler.driver().this_thread().api_enter(cl);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
    profiler.driver().this_thread().configured_call().finish();
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaLaunchKernel(const CUpti_CallbackData *cbdata,
                                   Profiler &profiler,
                                   const CUpti_CallbackDomain domain) {

  auto params = ((cudaLaunchKernel_v7000_params *)(cbdata->functionParams));
  const void *func = params->func;
  const dim3 gridDim = params->gridDim;
  const dim3 blockDim = params->blockDim;
  void *const *args = params->args;
  const size_t sharedMem = params->sharedMem;
  const cudaStream_t stream = params->stream;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    const size_t numArgs = 0; // FIXME, how to get this

    std::vector<uintptr_t> launchArgs(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      launchArgs[i] = reinterpret_cast<uintptr_t>(args[i]);
    }
    std::vector<CudaLaunchParams> launchParams;
    launchParams.push_back(
        CudaLaunchParams(gridDim, blockDim, launchArgs, sharedMem, stream));
    auto api = make_api_this_thread_now(cbdata, domain);
    auto cl = std::make_shared<CudaLaunch>(api, func, launchParams);
    profiler.driver().this_thread().api_enter(cl);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpy(const CUpti_CallbackData *cbdata,
                             Profiler &profiler,
                             const CUpti_CallbackDomain domain) {

  // extract API call parameters
  auto params = ((cudaMemcpy_v3020_params *)(cbdata->functionParams));
  const void *dst = params->dst;
  const void *src = params->src;
  const size_t count = params->count;
  const cudaMemcpyKind kind = params->kind;
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto api = make_api_this_thread_now(cbdata, domain);
    auto m = std::make_shared<Memcpy>(api, dst, src, count, kind);
    profiler.driver().this_thread().api_enter(m);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyAsync(const CUpti_CallbackData *cbdata,
                                  Profiler &profiler,
                                  const CUpti_CallbackDomain domain) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = ((cudaMemcpyAsync_v3020_params *)(cbdata->functionParams));
    const void *dst = params->dst;
    const void *src = params->src;
    const size_t count = params->count;
    const cudaMemcpyKind kind = params->kind;
    const cudaStream_t stream = params->stream;

    auto api = make_api_this_thread_now(cbdata, domain);
    Memcpy m(api, dst, src, count, kind);
    auto ma = std::make_shared<MemcpyAsync>(m, stream);
    profiler.driver().this_thread().api_enter(ma);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpy2DAsync(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler,
                                    const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = reinterpret_cast<const cudaMemcpy2DAsync_v3020_params *>(
        cbdata->functionParams);
    const void *dst = params->dst;
    const size_t dpitch = params->dpitch;
    const void *src = params->src;
    const size_t spitch = params->spitch;
    // const size_t width = params->width;
    const size_t height = params->height;
    const cudaMemcpyKind kind = params->kind;
    const cudaStream_t stream = params->stream;

    const size_t count = height * std::min(dpitch, spitch);
    auto api = make_api_this_thread_now(cbdata, domain);
    Memcpy m(api, dst, src, count, kind);
    auto ma = std::make_shared<MemcpyAsync>(m, stream);
    profiler.driver().this_thread().api_enter(ma);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyPeerAsync(const CUpti_CallbackData *cbdata,
                                      Profiler &profiler,
                                      const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = reinterpret_cast<const cudaMemcpyPeerAsync_v4000_params *>(
        cbdata->functionParams);
    const void *dst = params->dst;
    const int dstDevice = params->dstDevice;
    const void *src = params->src;
    const int srcDevice = params->srcDevice;
    const size_t count = params->count;
    const cudaStream_t stream = params->stream;

    auto api = make_api_this_thread_now(cbdata, domain);
    Memcpy m(api, dst, src, count, cudaMemcpyDeviceToDevice);
    MemcpyAsync ma(m, stream);
    auto mpa = std::make_shared<MemcpyPeerAsync>(ma, dstDevice, srcDevice);
    profiler.driver().this_thread().api_enter(mpa);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMalloc(const CUpti_CallbackData *cbdata,
                             Profiler &profiler,
                             const CUpti_CallbackDomain domain) {
  const auto params = ((cudaMalloc_v3020_params *)(cbdata->functionParams));

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    auto api = make_api_this_thread_now(cbdata, domain);
    auto da = std::make_shared<DeviceAlloc>(api, size);
    profiler.driver().this_thread().api_enter(da);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto da = std::dynamic_pointer_cast<DeviceAlloc>(api)) {
      const void *const *devPtr = params->devPtr;
      assert(devPtr);
      da->set_ptr(*devPtr);
      da->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(da->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected DeviceAlloc");
    }

  } else {
    assert(0 && "how did we get here?");
  }
}

static void handleCudaMallocHost(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler,
                                 const CUpti_CallbackDomain domain) {
  auto params = ((cudaMallocHost_v3020_params *)(cbdata->functionParams));
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto ha = std::make_shared<HostAlloc>(api, size,
                                          0 /* like cudaHostAllocDefault */, 0);
    profiler.driver().this_thread().api_enter(ha);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto ha = std::dynamic_pointer_cast<HostAlloc>(api)) {
      const void *const *ptr = params->ptr;
      ha->set_ptr(ptr);
      ha->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(ha->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected CudaMalloc");
    }
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocManaged(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler,
                                    const CUpti_CallbackDomain domain) {
  auto params = reinterpret_cast<const cudaMallocManaged_v6000_params *>(
      cbdata->functionParams);

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    const unsigned int flags = params->flags;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto ma = std::make_shared<ManagedAlloc>(api, size, flags);

    profiler.driver().this_thread().api_enter(ma);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<ManagedAlloc>(api)) {
      const void *const *devPtr = params->devPtr;
      cm->set_ptr(*devPtr);
      cm->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(cm->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected ManagedAlloc");
    }
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetDevice(const CUpti_CallbackData *cbdata,
                                Profiler &profiler,
                                const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto params = ((cudaSetDevice_v3020_params *)(cbdata->functionParams));
    const int device = params->device;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto csd = std::make_shared<CudaSetDevice>(api, device);
    profiler.driver().this_thread().api_enter(csd);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetupArgument(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler,
                                    const CUpti_CallbackDomain domain) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbdata->functionParams));
    const void *arg = params->arg;
    size_t size = params->size;
    size_t offset = params->offset;
    auto api = make_api_this_thread_now(cbdata, domain);
    auto csa = std::make_shared<CudaSetupArgument>(api, arg, size, offset);
    profiler.driver().this_thread().api_enter(csa);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamCreate(const CUpti_CallbackData *cbdata,
                                   Profiler &profiler,
                                   const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto api = make_api_this_thread_now(cbdata, domain);
    auto csc = std::make_shared<CudaStreamCreate>(api);
    profiler.driver().this_thread().api_enter(csc);
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

template <typename PARAM_TYPE>
static void handleCudaStreamDestroy(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler,
                                    const CUpti_CallbackDomain domain) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    const auto params =
        reinterpret_cast<const PARAM_TYPE *>(cbdata->functionParams);
    const cudaStream_t stream = params->stream;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto csd = std::make_shared<CudaStreamDestroy>(api, stream);
    profiler.driver().this_thread().api_enter(csd);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuCtxSetCurrent(Profiler &profiler,
                                  const CUpti_CallbackData *cbdata,
                                  const CUpti_CallbackDomain domain) {

  auto params = ((cuCtxSetCurrent_params *)(cbdata->functionParams));
  const CUcontext ctx = params->ctx;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto api = make_api_this_thread_now(cbdata, domain);

    auto ccsc = std::make_shared<CuCtxSetCurrent>(api, ctx);
    profiler.driver().this_thread().api_enter(ccsc);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuLaunchKernel(Profiler &profiler,
                                 const CUpti_CallbackData *cbdata,
                                 const CUpti_CallbackDomain domain) {
  auto params = reinterpret_cast<const cuLaunchKernel_ptsz_params_st *>(
      cbdata->functionParams);
  CUfunction f = params->f;
  unsigned int gridDimX = params->gridDimX;
  unsigned int gridDimY = params->gridDimY;
  unsigned int gridDimZ = params->gridDimZ;
  unsigned int blockDimX = params->blockDimX;
  unsigned int blockDimY = params->blockDimY;
  unsigned int blockDimZ = params->blockDimZ;
  unsigned int sharedMemBytes = params->sharedMemBytes;
  CUstream hStream = params->hStream;
  void **kernelParams = params->kernelParams;
  void **extra = params->extra;

  // Look for args in extra field
  void *extraParamBuffer = nullptr;
  size_t extraParamBufferSize = 0;
  for (size_t i = 0; (extra[i] != NULL) && (extra[i] != CU_LAUNCH_PARAM_END);
       ++i) {
    if (CU_LAUNCH_PARAM_BUFFER_POINTER == extra[i]) {
      extraParamBuffer = extra[++i];
    } else if (CU_LAUNCH_PARAM_BUFFER_SIZE == extra[i]) {
      extraParamBufferSize = *reinterpret_cast<size_t *>(extra[++i]);
    } else {
      assert(0 && "how did we get here");
    }
  }

  if (extraParamBuffer) {
    profiler.log() << "WARN: not recording kernel args" << std::endl;
  } else {
    if (cbdata->callbackSite == CUPTI_API_ENTER) {

      auto numArgs =
          profiler.driver().this_thread().configured_call().num_args();

      std::vector<uintptr_t> launchArgs(numArgs);
      for (size_t i = 0; i < numArgs; ++i) {
        launchArgs[i] = reinterpret_cast<uintptr_t>(kernelParams[i]);
      }

      const dim3 gridDim(gridDimX, gridDimY, gridDimZ);
      const dim3 blockDim(blockDimX, blockDimY, blockDimZ);
      std::vector<CudaLaunchParams> launchParams;
      launchParams.push_back(CudaLaunchParams(gridDim, blockDim, launchArgs,
                                              sharedMemBytes, hStream));
      auto api = make_api_this_thread_now(cbdata, domain);
      auto cl = std::make_shared<CudaLaunch>(api, f, launchParams);
      profiler.driver().this_thread().api_enter(cl);
    } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
      profiler.driver().this_thread().configured_call().finish();
      finalize_api(profiler);
    } else {
      assert(0 && "How did we get here?");
    }
  }
}

static void handleCuMemHostAlloc(Profiler &profiler,
                                 const CUpti_CallbackData *cbdata,
                                 const CUpti_CallbackDomain domain) {

  auto params = ((cuMemHostAlloc_params *)(cbdata->functionParams));

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    const size_t bytesize = params->bytesize;
    const int Flags = params->Flags;

    auto api = make_api_this_thread_now(cbdata, domain);
    auto ha = std::make_shared<HostAlloc>(api, bytesize, 0 /*no driver flags*/,
                                          Flags);
    profiler.driver().this_thread().api_enter(ha);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<HostAlloc>(api)) {
      const void *const *pp = params->pp;
      cm->set_ptr(*pp);
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

  // If we're not nesting, only handle API exits for whatever API we're in
  if (profiler.driver().this_thread().api_stack_size()) {
    if (auto api = std::dynamic_pointer_cast<cuda::cupti::callback::Api>(
            profiler.driver().this_thread().current_api())) {
      if (api->domain() != domain) {
        profiler.log() << "Not drilling down CUPTI API: domain mismatch\n";
        return;
      }
    }
  }

  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleCudaMemcpy(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      handleCudaMemcpyAsync(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      handleCudaMemcpyPeerAsync(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleCudaMalloc(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
      handleCudaMallocHost(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
      handleCudaMallocManaged(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      handleCudaFree(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
      handleCudaFreeHost(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
      handleCudaConfigureCall(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
      handleCudaSetupArgument(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      handleCudaLaunch(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020:
      handleCudaSetDevice(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
      handleCudaStreamCreate(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050:
      handleCudaStreamDestroy<cudaStreamDestroy_v5050_params>(cbdata, profiler,
                                                              domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
      handleCudaStreamDestroy<cudaStreamDestroy_v3020_params>(cbdata, profiler,
                                                              domain);
      break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
    //   handleCudaStreamSynchronize(cbdata);
    //   break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
      handleCudaMemcpy2DAsync(cbdata, profiler, domain);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      handleCudaLaunchKernel(cbdata, profiler, domain);
      break;
    default:
      profiler.log() << "DEBU: ( tid= " << get_thread_id()
                     << " ) skipping runtime call " << cbdata->functionName
                     << std::endl;
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      handleCuLaunchKernel(profiler, cbdata, domain);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(profiler, cbdata, domain);
      break;
    // case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
    // handleCuModuleGetFunction(cbdata);
    //   break;
    // case CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2:
    // handleCuModuleGetGlobal_v2(cbdata);
    //   break;
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
      handleCuCtxSetCurrent(profiler, cbdata, domain);
      break;
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
