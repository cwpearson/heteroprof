#include <cassert>

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

Api make_api_this_thread_now(const CUpti_CallbackData *cbdata) {
  auto now = std::chrono::high_resolution_clock::now();
  auto tid = get_thread_id();
  Api api(tid, cbdata);
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
                                    Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = ((cudaConfigureCall_v3020_params *)(cbdata->functionParams));
    auto gridDim = params->gridDim;
    auto blockDim = params->blockDim;
    auto sharedMem = params->sharedMem;
    auto stream = params->stream;

    auto a = make_api_this_thread_now(cbdata);
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
                               Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto params = ((cudaFreeHost_v3020_params *)(cbdata->functionParams));
    const void *ptr = params->ptr;

    auto api = make_api_this_thread_now(cbdata);
    auto hf = std::make_shared<HostFree>(api, ptr);
    profiler.driver().this_thread().api_enter(hf);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFree(const CUpti_CallbackData *cbdata,
                           Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto params =
        reinterpret_cast<const cudaFree_v3020_params *>(cbdata->functionParams);
    const void *devPtr = params->devPtr;

    auto api = make_api_this_thread_now(cbdata);
    auto df = std::make_shared<DeviceFree>(api, devPtr);
    profiler.driver().this_thread().api_enter(df);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaHostAlloc(const CUpti_CallbackData *cbdata,
                                Profiler &profiler) {
  auto params = reinterpret_cast<const cudaHostAlloc_v3020_params *>(
      cbdata->functionParams);

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    unsigned int flags = params->flags;

    auto now = std::chrono::high_resolution_clock::now();
    auto tid = get_thread_id();

    auto api = std::make_shared<HostAlloc>(tid, cbdata, size, flags,
                                           0 /*no driver flags */);
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);
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
                             Profiler &profiler) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = reinterpret_cast<const cudaLaunch_v3020_params *>(
        cbdata->functionParams);
    auto func = params->func;

    std::vector<CudaLaunchParams> launchParams;

    auto api = make_api_this_thread_now(cbdata);
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
                                   Profiler &profiler) {

  auto params = ((cudaLaunchKernel_v7000_params *)(cbdata->functionParams));
  const void *func = params->func;
  const dim3 gridDim = params->gridDim;
  const dim3 blockDim = params->blockDim;
  void *const *args = params->args;
  const size_t sharedMem = params->sharedMem;
  const cudaStream_t stream = params->stream;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto numArgs = profiler.driver().this_thread().configured_call().num_args();

    std::vector<uintptr_t> launchArgs(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      launchArgs[i] = reinterpret_cast<uintptr_t>(args[i]);
    }
    std::vector<CudaLaunchParams> launchParams;
    launchParams.push_back(
        CudaLaunchParams(gridDim, blockDim, launchArgs, sharedMem, stream));
    auto api = make_api_this_thread_now(cbdata);
    auto cl = std::make_shared<CudaLaunch>(api, func, launchParams);
    profiler.driver().this_thread().api_enter(cl);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    profiler.driver().this_thread().configured_call().finish();
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpy(const CUpti_CallbackData *cbdata,
                             Profiler &profiler) {

  // extract API call parameters
  auto params = ((cudaMemcpy_v3020_params *)(cbdata->functionParams));
  const void *dst = params->dst;
  const void *src = params->src;
  const size_t count = params->count;
  const cudaMemcpyKind kind = params->kind;
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto api = make_api_this_thread_now(cbdata);
    auto m = std::make_shared<Memcpy>(api, dst, src, count, kind);
    profiler.driver().this_thread().api_enter(m);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyAsync(const CUpti_CallbackData *cbdata,
                                  Profiler &profiler) {

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = ((cudaMemcpyAsync_v3020_params *)(cbdata->functionParams));
    const void *dst = params->dst;
    const void *src = params->src;
    const size_t count = params->count;
    const cudaMemcpyKind kind = params->kind;
    const cudaStream_t stream = params->stream;

    auto api = make_api_this_thread_now(cbdata);
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
                                    Profiler &profiler) {
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
    auto api = make_api_this_thread_now(cbdata);
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
                                      Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto params = reinterpret_cast<const cudaMemcpyPeerAsync_v4000_params *>(
        cbdata->functionParams);
    const void *dst = params->dst;
    const int dstDevice = params->dstDevice;
    const void *src = params->src;
    const int srcDevice = params->srcDevice;
    const size_t count = params->count;
    const cudaStream_t stream = params->stream;

    auto api = make_api_this_thread_now(cbdata);
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
                             Profiler &profiler) {
  const auto params = ((cudaMalloc_v3020_params *)(cbdata->functionParams));

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    auto tid = get_thread_id();
    auto api = std::make_shared<DeviceAlloc>(tid, cbdata, size);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<DeviceAlloc>(api)) {
      const void *const *devPtr = params->devPtr;
      assert(devPtr);
      cm->set_ptr(*devPtr);
      cm->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(cm->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected DeviceAlloc");
    }

  } else {
    assert(0 && "how did we get here?");
  }
}

static void handleCudaMallocHost(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler) {
  auto params = ((cudaMallocHost_v3020_params *)(cbdata->functionParams));
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;

    auto now = std::chrono::high_resolution_clock::now();
    auto tid = get_thread_id();

    auto api = std::make_shared<HostAlloc>(
        tid, cbdata, size, 0 /* like cudaHostAllocDefault */, 0);
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler.driver().this_thread().current_api();

    if (auto cm = std::dynamic_pointer_cast<HostAlloc>(api)) {
      const void *const *ptr = params->ptr;
      cm->set_ptr(ptr);
      cm->set_wall_end(std::chrono::high_resolution_clock::now());
      profiler.record(cm->to_json());
      profiler.driver().this_thread().api_exit();
    } else {
      assert(0 && "expected CudaMalloc");
    }
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocManaged(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler) {
  auto params = reinterpret_cast<const cudaMallocManaged_v6000_params *>(
      cbdata->functionParams);

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    const size_t size = params->size;
    const unsigned int flags = params->flags;

    auto api = make_api_this_thread_now(cbdata);
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

static void handleCuCtxSetCurrent(const CUpti_CallbackData *cbdata,
                                  Profiler &profiler) {

  auto params = ((cuCtxSetCurrent_params *)(cbdata->functionParams));
  const CUcontext ctx = params->ctx;

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    auto now = std::chrono::high_resolution_clock::now();
    auto tid = get_thread_id();

    auto api = std::make_shared<CuCtxSetCurrent>(tid, cbdata, ctx);
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
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

template <typename PARAM_TYPE>
static void handleCudaStreamDestroy(const CUpti_CallbackData *cbdata,
                                    Profiler &profiler) {
  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    const auto params =
        reinterpret_cast<const PARAM_TYPE *>(cbdata->functionParams);
    const cudaStream_t stream = params->stream;

    auto tid = get_thread_id();
    auto api = std::make_shared<CudaStreamDestroy>(tid, cbdata, stream);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuLaunchKernel(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler) {
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

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    auto numArgs = profiler.driver().this_thread().configured_call().num_args();

    // Look for args in extra field
    for (auto *p = reinterpret_cast<unsigned char *>(extra[0]);
         (p != NULL) && (p != CU_LAUNCH_PARAM_END); p += sizeof(void *)) {
      assert(0 && "Unimplemented: need to know arg sizes (from PTX?)");
    }

    std::vector<uintptr_t> launchArgs(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      launchArgs[i] = reinterpret_cast<uintptr_t>(kernelParams[i]);
    }

    const dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    const dim3 blockDim(blockDimX, blockDimY, blockDimZ);
    std::vector<CudaLaunchParams> launchParams;
    launchParams.push_back(CudaLaunchParams(gridDim, blockDim, launchArgs,
                                            sharedMemBytes, hStream));
    auto api = make_api_this_thread_now(cbdata);
    auto cl = std::make_shared<CudaLaunch>(api, f, launchParams);
    profiler.driver().this_thread().api_enter(cl);
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    profiler.driver().this_thread().configured_call().finish();
    finalize_api(profiler);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuMemHostAlloc(const CUpti_CallbackData *cbdata,
                                 Profiler &profiler) {

  auto params = ((cuMemHostAlloc_params *)(cbdata->functionParams));

  if (cbdata->callbackSite == CUPTI_API_ENTER) {

    const size_t bytesize = params->bytesize;
    const int Flags = params->Flags;

    auto tid = get_thread_id();
    auto api = std::make_shared<HostAlloc>(tid, cbdata, bytesize,
                                           0 /*no driver flags*/, Flags);
    auto now = std::chrono::high_resolution_clock::now();
    api->set_wall_start(now);
    profiler.driver().this_thread().api_enter(api);

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

  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleCudaMemcpy(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      handleCudaMemcpyAsync(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      handleCudaMemcpyPeerAsync(cbdata, profiler);
      break;
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
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050:
      handleCudaStreamDestroy<cudaStreamDestroy_v5050_params>(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
      handleCudaStreamDestroy<cudaStreamDestroy_v3020_params>(cbdata, profiler);
      break;
    // case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
    //   handleCudaStreamSynchronize(cbdata);
    //   break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
      handleCudaMemcpy2DAsync(cbdata, profiler);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      handleCudaLaunchKernel(cbdata, profiler);
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
      handleCuLaunchKernel(cbdata, profiler);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(cbdata, profiler);
      break;
    // case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
    // handleCuModuleGetFunction(cbdata);
    //   break;
    // case CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2:
    // handleCuModuleGetGlobal_v2(cbdata);
    //   break;
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
      handleCuCtxSetCurrent(cbdata, profiler);
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
