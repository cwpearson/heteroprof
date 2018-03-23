#include <cassert>

#include <cuda_runtime_api.h>
#include <cupti.h>

#include "model/cuda/cupti/callback/api.hpp"
#include "model/cuda/memory.hpp"
#include "profiler.hpp"

using model::cuda::Location;
using model::cuda::Memory;

// Function that is called when a Kernel is called
// Record timing in this
#if false
static void handleCudaLaunch(void *userdata, const CUpti_CallbackData *cbInfo) {
  profiler::log() << "(tid=" << cprof::model::get_thread_id() << ") launching "
                  << symbolName << std::endl;

  // print_backtrace();

  // Get the current stream
  // const cudaStream_t stream =
  // profiler::driver().this_thread().configured_call().stream;
  const char *symbolName;
  if (!cbInfo->symbolName) {
    profiler::log() << "WARN: empty symbol name" << std::endl;
    symbolName = "[unknown symbol name]";
  } else {
    symbolName = cbInfo->symbolName;
  }

  const int devId = profiler::driver().this_thread().current_device();
  auto AS = profiler::hardware().address_space(devId);
  auto api = profiler::driver().this_thread().current_api();

  // Find all values that are used by arguments
  const auto &args = profiler::driver().this_thread().configured_call().args_;
  for (const auto &arg : args) {
    api->add_arg(arg);
  }

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaLaunch entry" << std::endl;
    assert(cbInfo);
    api->set_wall_start(cprof::now());

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    api->set_wall_end(cprof::now());

    // The kernel could have modified any argument values.
    // Hash each value and compare to the one recorded at kernel launch
    // If there is a difference, create a new value

    profiler::record_json(api.to_json);
    profiler::driver().this_thread().configured_call().valid_ = false;
    profiler::driver().this_thread().configured_call().args_.clear();
  } else {
    assert(0 && "How did we get here?");
  }

  profiler::log() << "callback: cudaLaunch: done" << std::endl;
  if (profiler::driver().this_thread().configured_call().valid_) {
    // Potentially save launch arguments here
  }
}
#endif

#if false
static void handleCudaLaunchKernel(void *userdata, Allocations &allocations,
                                   const CUpti_CallbackData *cbInfo) {
  profiler::log() << "INFO: callback: cudaLaunchKernel preamble (tid="
                  << cprof::model::get_thread_id() << ")" << std::endl;

  auto params = ((cudaLaunchKernel_v7000_params *)(cbInfo->functionParams));
  const void *func = params->func;
  profiler::log() << "launching " << func << std::endl;
  const dim3 gridDim = params->gridDim;
  const dim3 blockDim = params->blockDim;
  void *const *args = params->args;
  const size_t sharedMem = params->sharedMem;
  const cudaStream_t stream = params->stream;

  // print_backtrace();

  const char *symbolName = cbInfo->symbolName;
  // const char *symbolName = (char*)func;

  // assert(0 && "Unimplemented");

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // Some NCCL calls reach here and do not have a symbol name.
    // Sanity check to prevent crash.
    if (cbInfo->symbolName != NULL) {
      auto api = std::make_shared<ApiRecord>(
          cbInfo->functionName, cbInfo->symbolName,
          profiler::driver().this_thread().current_device());

      profiler::atomic_out(api->to_json_string() + "\n");
    }

  } else {
    assert(0 && "How did we get here?");
  }

  profiler::log() << "callback: cudaLaunchKernel: done" << std::endl;
}
#endif

#if false
void record_memcpy(const CUpti_CallbackData *cbInfo, Allocations &allocations,
                   const ApiRecordRef &api, const uintptr_t dst,
                   const uintptr_t src, const MemoryCopyKind &kind,
                   const size_t srcCount, const size_t dstCount,
                   const int peerSrc, const int peerDst) {

    (void)peerSrc;
    (void)peerDst;

    Allocation srcAlloc, dstAlloc;

    const int devId = profiler::driver().this_thread().current_device();

    // guess the src and dst address space
    auto srcAS = AddressSpace::Invalid();
    auto dstAS = AddressSpace::Invalid();
    if (profiler::hardware().cuda_device(devId).unifiedAddressing_) {
      srcAS = dstAS = profiler::hardware().address_space(devId);
    } else if (MemoryCopyKind::CudaHostToDevice() == kind) {
      srcAS = AddressSpace::Host();
      dstAS = profiler::hardware().address_space(devId);
    } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
      dstAS = AddressSpace::Host();
      srcAS = profiler::hardware().address_space(devId);
    } else if (MemoryCopyKind::CudaDefault() == kind) {
      srcAS = dstAS = AddressSpace::CudaUVA();
    } else {
      assert(0 && "Unhandled MemoryCopyKind");
    }

    assert(srcAS.is_valid());
    assert(dstAS.is_valid());
    // Set address space, and create missing allocations along the way
    if (MemoryCopyKind::CudaHostToDevice() == kind) {
      profiler::log() << src << "--[h2d]--> " << dst << std::endl;

      // Source allocation may not have been created by a CUDA api
      srcAlloc = allocations.find(src, srcCount, srcAS);
      if (!srcAlloc) {

        const auto numaNode = get_numa_node(src);
        const auto loc = Location::Host(numaNode);

        srcAlloc = allocations.new_allocation(src, srcCount, srcAS,
                                              Memory::Unknown, loc);
        profiler::log() << "WARN: Couldn't find src alloc. Created implict host allocation= {"
                        << srcAS.str() << "}[ " << src << " , +" << srcCount
                        << " )" << std::endl;
      }
    } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
      profiler::log() << src << "--[d2h]--> " << dst << std::endl;

      // Destination allocation may not have been created by a CUDA api
      // FIXME: we may be copying only a slice of an existing allocation. if
      // it overlaps, it should be joined
      dstAlloc = allocations.find(dst, dstCount, dstAS);
      if (!dstAlloc) {

        const auto numaNode = get_numa_node(dst);
        const auto loc = Location::Host(numaNode);

        dstAlloc = allocations.new_allocation(dst, dstCount, dstAS,
                                              Memory::Unknown, loc);
        profiler::log() << "WARN: Couldn't find dst alloc. Created implict host allocation= {"
                        << dstAS.str() << "}[ " << dst << " , +" << dstCount
                        << " )" << std::endl;
      }
    }

    // Look for existing src / dst allocations.
    // Either we just made it, or it should already exist.
    if (!srcAlloc) {
      srcAlloc = allocations.find(src, srcCount, srcAS);
      assert(srcAlloc);
    }
    if (!dstAlloc) {
      dstAlloc = allocations.find(dst, dstCount, dstAS);
      assert(dstAlloc);
    }

    assert(srcAlloc && "Couldn't find or create src allocation");
    assert(dstAlloc && "Couldn't find or create dst allocation");
    // There may not be a source value, because it may have been initialized
    // on the host
    auto srcVal = allocations.find_value(src, srcCount,
    srcAlloc.address_space()); assert(srcVal && "Value should have been created with allocation"); profiler::log() << "memcpy: found src value srcId=" <<
    srcVal.id()
                    << std::endl;
    profiler::log() << "WARN: Setting srcVal size by memcpy count" << std::endl;
    srcVal.set_size(srcCount);

    // always create a new dst value
    assert(srcVal);
    auto dstVal = dstAlloc.new_value(dst, dstCount, srcVal.initialized());
    assert(dstVal);
    // dstVal->record_meta_append(cbInfo->functionName); // FIXME

    api->add_input(srcVal);
    api->add_output(dstVal);
    api->add_kv("kind", kind.str());
    api->add_kv("srcCount", srcCount);
    api->add_kv("dstCount", dstCount);
    profiler::atomic_out(api->to_json_string() + "\n");

    if (Profiler::instance().is_zipkin_enabled()) {
      auto b = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                   api->wall_start())
                   .time_since_epoch();

      auto span = Profiler::instance().memcpyTracer_->StartSpan(
          std::to_string(cbInfo->correlationId),
          {ChildOf(&Profiler::instance().rootSpan_->context()),
           opentracing::StartTimestamp(b)});

      // span->SetTag("Transfer size", memcpyRecord->bytes);
      // span->SetTag("Transfer type",
      // memcpy_type_to_string(memcpyRecord->copyKind)); span->SetTag("Host
      // Thread", std::to_string(threadId));

      // auto timeElapsed = memcpyRecord->end - memcpyRecord->start;
      // span->SetTag("CUPTI Duration", std::to_string(timeElapsed));
      // auto err = tracer->Inject(current_span->context(), carrier);
      auto e =
          std::chrono::time_point_cast<std::chrono::nanoseconds>(api->wall_end())
              .time_since_epoch();

      span->Finish({opentracing::FinishTimestamp(e)});
    }
}
#endif

#if false
static void handleCudaMemcpy(Allocations &allocations,
                             const CUpti_CallbackData *cbInfo) {

  // extract API call parameters
  auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const cudaMemcpyKind kind = params->kind;
  const size_t count = params->count;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: callback: cudaMemcpy enter" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t endTimeStamp;
    cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
    // profiler::log() << "The end timestamp is " << endTimeStamp <<
    // std::endl; std::cout << "The end time is " << cbInfo->end_time;
    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    record_memcpy(cbInfo, allocations, api, dst, src, MemoryCopyKind(kind),
                  count, count, 0 /*unused*/, 0 /*unused */);
    profiler::log() << "INFO: callback: cudaMemcpy exit" << std::endl;

  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaMemcpyAsync(Allocations &allocations,
                                  const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpyAsync_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t count = params->count;
  const cudaMemcpyKind kind = params->kind;
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaMemcpyAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    record_memcpy(cbInfo, allocations, api, dst, src, MemoryCopyKind(kind),
                  count, count, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaMemcpy2DAsync(Allocations &allocations,
                                    const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpy2DAsync_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const size_t dpitch = params->dpitch;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t spitch = params->spitch;
  // const size_t width = params->width;
  const size_t height = params->height;
  const cudaMemcpyKind kind = params->kind;
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaMemcpy2DAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    const size_t srcCount = height * spitch;
    const size_t dstCount = height * dpitch;
    record_memcpy(cbInfo, allocations, api, dst, src, MemoryCopyKind(kind),
                  srcCount, dstCount, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaMemcpyPeerAsync(Allocations &allocations,
                                      const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpyPeerAsync_v4000_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const int dstDevice = params->dstDevice;
  const uintptr_t src = (uintptr_t)params->src;
  const int srcDevice = params->srcDevice;
  const size_t count = params->count;
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaMemcpyPeerAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_start(cprof::now());
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->set_wall_end(cprof::now());

    record_memcpy(cbInfo, allocations, api, dst, src,
                  MemoryCopyKind::CudaPeer(), count, count, srcDevice,
                  dstDevice);
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaMallocManaged(Allocations &allocations,
                                    const CUpti_CallbackData *cbInfo) {
  auto params = ((cudaMallocManaged_v6000_params *)(cbInfo->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  // const unsigned int flags = params->flags;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

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
#endif

#if false
void record_mallochost(Allocations &allocations, const uintptr_t ptr,
                       const size_t size) {
  auto AM = cprof::model::Memory::Pagelocked;

  const int devId = profiler::driver().this_thread().current_device();
  auto AS = profiler::hardware().address_space(devId);
  const int numaNode = get_numa_node(ptr);

  Allocation alloc =
      allocations.new_allocation(ptr, size, AS, AM, Location::Host(numaNode));
  profiler::log() << "INFO: made new mallochost @ " << ptr
                  << " [nn=" << numaNode << "]" << std::endl;

  assert(alloc);
}

static void handleCudaMallocHost(Allocations &allocations,
                                 const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMallocHost_v3020_params *)(cbInfo->functionParams));
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
#endif

#if false
static void handleCuMemHostAlloc(Allocations &allocations,
                                 const CUpti_CallbackData *cbInfo) {

  auto &ts = profiler::driver().this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
      ts.parent_api()->cbid() ==
          CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020) {
    profiler::log() << "WARN: skipping cuMemHostAlloc inside cudaMallocHost"
                    << std::endl;
    return;
  }

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    auto params = ((cuMemHostAlloc_params *)(cbInfo->functionParams));
    uintptr_t pp = (uintptr_t)(*(params->pp));
    const size_t bytesize = params->bytesize;
    const int Flags = params->Flags;
    if (Flags & CU_MEMHOSTALLOC_PORTABLE) {
      // FIXME
      profiler::log()
          << "WARN: cuMemHostAlloc with unhandled CU_MEMHOSTALLOC_PORTABLE"
          << std::endl;
    }
    if (Flags & CU_MEMHOSTALLOC_DEVICEMAP) {
      // FIXME
      profiler::log()
          << "WARN: cuMemHostAlloc with unhandled CU_MEMHOSTALLOC_DEVICEMAP"
          << std::endl;
    }
    if (Flags & CU_MEMHOSTALLOC_WRITECOMBINED) {
      // FIXME
      profiler::log() << "WARN: cuMemHostAlloc with unhandled "
                         "CU_MEMHOSTALLOC_WRITECOMBINED"
                      << std::endl;
    }
    profiler::log() << "INFO: [cuMemHostAlloc] " << pp << "[" << bytesize << "]"
                    << std::endl;

    record_mallochost(allocations, pp, bytesize);
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCuLaunchKernel(Allocations &allocations,
                                 const CUpti_CallbackData *cbInfo) {
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
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: enter cuLaunchKernel" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: exit cuLaunchKernel" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCuModuleGetFunction(const CUpti_CallbackData *cbInfo) {

  auto params = ((cuModuleGetFunction_params *)(cbInfo->functionParams));
  const CUfunction hfunc = *(params->hfunc);
  // const CUmodule hmod = params->hmod;
  const char *name = params->name;

  profiler::log() << "INFO: cuModuleGetFunction for " << name << " @ " << hfunc
                  << std::endl;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: enter cuModuleGetFunction" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: exit cuModuleGetFunction" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCuModuleGetGlobal_v2(const CUpti_CallbackData *cbInfo) {

  // auto params = ((cuModuleGetGlobal_v2_params *)(cbInfo->functionParams));

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

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: enter cuModuleGetGlobal_v2" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: exit cuModuleGetGlobal_v2" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCuCtxSetCurrent(const CUpti_CallbackData *cbInfo) {


  auto params = ((cuCtxSetCurrent_params *)(cbInfo->functionParams));
  const CUcontext ctx = params->ctx;
  const int pid = cprof::model::get_thread_id();

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: (tid=" << pid << ") setting ctx " << ctx
                    << std::endl;
    profiler::driver().this_thread().set_context(ctx);
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaFreeHost(Allocations &allocations,
                               const CUpti_CallbackData *cbInfo) {

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaFreeHost_v3020_params *)(cbInfo->functionParams));
    uintptr_t ptr = (uintptr_t)(params->ptr);
    cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
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
#endif

#if false
static void handleCudaMalloc(Allocations &allocations,
                             const CUpti_CallbackData *cbInfo) {
  const auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: cudaMalloc: [" << devPtr << ", +" << size
                    << ") entry" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    const cudaError_t res =
        *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
    profiler::log() << "INFO: " << res << " = cudaMalloc: [" << devPtr << ", +"
                    << size << ")" << std::endl;
    if (res != cudaSuccess) {
      profiler::log() << "WARN: cudaMalloc had an error" << std::endl;
      return;
    }

    // Create the new allocation
    // FIXME: need to check which address space this is in
    const int devId = profiler::driver().this_thread().current_device();
    auto AS = profiler::hardware().address_space(devId);
    auto AM = cprof::model::Memory::Pageable;

    Allocation a = allocations.new_allocation(devPtr, size, AS, AM,
                                              Location::CudaDevice(devId));
    profiler::log() << "INFO: (tid=" << cprof::model::get_thread_id()
                    << ") [cudaMalloc] new alloc=" << (uintptr_t)a.id()
                    << " pos=" << a.pos() << std::endl;

    // Create new database allocation record
    // auto dependency_tracking = DependencyTracking::instance();
    // dependency_tracking.memory_ptr_create(a->pos());

    // auto digest = hash_device(devPtr, size);
    // profiler::log() <<"uninitialized digest: %llu\n", digest);
  } else {
    assert(0 && "how did we get here?");
  }
}
#endif

#if false
static void handleCudaFree(Allocations &allocations,
                           const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaFree_v3020_params *)(cbInfo->functionParams));
    auto devPtr = (uintptr_t)params->devPtr;
    cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
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
#endif

#if false
static void handleCudaSetDevice(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaSetDevice entry" << std::endl;
    auto params = ((cudaSetDevice_v3020_params *)(cbInfo->functionParams));
    const int device = params->device;

    profiler::driver().this_thread().set_device(device);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaConfigureCall(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: ( tid=" << cprof::model::get_thread_id()
                    << " ) callback: cudaConfigureCall entry" << std::endl;

    assert(!profiler::driver().this_thread().configured_call().valid_ &&
           "call is already configured?\n");

    auto params = ((cudaConfigureCall_v3020_params *)(cbInfo->functionParams));
    profiler::driver().this_thread().configured_call().gridDim_ =
        params->gridDim;
    profiler::driver().this_thread().configured_call().blockDim_ =
        params->blockDim;
    profiler::driver().this_thread().configured_call().sharedMem_ =
        params->sharedMem;
    profiler::driver().this_thread().configured_call().stream_ = params->stream;
    profiler::driver().this_thread().configured_call().valid_ = true;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaSetupArgument(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "callback: cudaSetupArgument entry (tid="
                    << cprof::model::get_thread_id() << ")" << std::endl;
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbInfo->functionParams));

    assert(profiler::driver().this_thread().configured_call().valid_);
    profiler::driver().this_thread().configured_call().args_.push_back(
        params->arg);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaStreamCreate(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::log() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    // const auto params =
    //     ((cudaStreamCreate_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = *(params->pStream);
    profiler::log() << "WARN: ignoring cudaStreamCreate" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaStreamDestroy(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    profiler::log() << "WARN: ignoring cudaStreamDestroy" << std::endl;
    // const auto params =
    //     ((cudaStreamDestroy_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

#if false
static void handleCudaStreamSynchronize(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::log() << "INFO: callback: cudaStreamSynchronize entry"
                    << std::endl;
    profiler::log() << "WARN: ignoring cudaStreamSynchronize" << std::endl;
    // const auto params =
    //     ((cudaStreamSynchronize_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}
#endif

void CUPTIAPI cuptiCallbackFunction(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    const CUpti_CallbackData *cbInfo) {

  (void)userdata; // data supplied at subscription

  if (!profiler::driver().this_thread().is_cupti_callbacks_enabled()) {
    return;
  }

  // profiler::timer().callback_add_annotations(cbInfo, cbid);

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {

      auto a = model::cuda::cupti::callback::Api(model::sys::get_thread_id(),
                                                 cbInfo);

      // profiler::driver().this_thread().api_enter(a);
    }
  }
  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      // handleCudaMemcpy(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      // handleCudaMemcpyAsync(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      // handleCudaMemcpyPeerAsync(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      // handleCudaMalloc(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
      // handleCudaMallocHost(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
      // handleCudaMallocManaged(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      // handleCudaFree(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
      // handleCudaFreeHost(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
      // handleCudaConfigureCall(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
      // handleCudaSetupArgument(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      // handleCudaLaunch(userdata, profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020:
      // handleCudaSetDevice(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
      // handleCudaStreamCreate(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
      // handleCudaStreamDestroy(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
      // handleCudaStreamSynchronize(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
      // handleCudaMemcpy2DAsync(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      // handleCudaLaunchKernel(userdata, profiler::allocations(), cbInfo);
      break;
    default:
      profiler::log() << "DEBU: ( tid= " << model::sys::get_thread_id()
                      << " ) skipping runtime call " << cbInfo->functionName
                      << std::endl;
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      // handleCuMemHostAlloc(profiler::allocations(), cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      // handleCuLaunchKernel(profiler::allocations(), cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
      // handleCuModuleGetFunction(cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2:
      // handleCuModuleGetGlobal_v2(cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
      // handleCuCtxSetCurrent(cbInfo);
      break;
    default:
      profiler::log() << "DEBU: ( tid= " << model::sys::get_thread_id()
                      << " ) skipping driver call " << cbInfo->functionName
                      << std::endl;
      break;
    }
  }
  default:
    break;
  }

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      // profiler::driver().this_thread().api_exit();
    }
  }
}
