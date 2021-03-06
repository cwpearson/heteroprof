#include <cassert>
#include <dlfcn.h>

#include <cudnn.h>

#include "cudnn/api.hpp"
#include "cudnn/preload.hpp"
#include "cudnn/util.hpp"

//Include classes for different functions
#include "cudnn/cudnn_create.hpp"
#include "cudnn/cudnn_destroy.hpp"
#include "cudnn/cudnn_activation_forward.hpp"
#include "cudnn/cudnn_add_tensor.hpp"
#include "cudnn/cudnn_activation_backward.hpp"
#include "cudnn/cudnn_convolution_backward_bias.hpp"
#include "cudnn/cudnn_convolution_backward_filter.hpp"
#include "cudnn/cudnn_convolution_forward.hpp"
#include "cudnn/cudnn_pooling_forward.hpp"
#include "cudnn/cudnn_softmax_forward.hpp"
#include "cudnn/cudnn_convolution_backward_data.hpp"

namespace cudnn {
using sys::get_thread_id;    
Profiler *profiler_ = nullptr;
void set_profiler(Profiler &p) { profiler_ = &p; }
Profiler &profiler() {
  assert(profiler_);
  return *profiler_;
}

Cudnn make_cudnn_this_thread_now(std::string name) {
  auto now = std::chrono::high_resolution_clock::now();
  auto tid = get_thread_id();
  Cudnn cudnn(tid, name);
  cudnn.set_wall_start(now);
  return cudnn;
}

void finalize_api(Profiler &p){
    auto api = p.driver().this_thread().current_api();
    api->set_wall_end(std::chrono::high_resolution_clock::now());
    p.record(api->to_json());
    p.driver().this_thread().api_exit(); 
}
} // namespace cudnn

using namespace cudnn;
using sys::get_thread_id;

size_t tensorSize(const cudnnTensorDescriptor_t tensorDesc) {
  size_t size;
  CUDNN_CHECK(cudnnGetTensorSizeInBytes(tensorDesc, &size), profiler().log());
  return size;
}

#define CUDNN_DLSYM_BOILERPLATE(name)                                          \
  static name##Func real_##name = nullptr;                                     \
  profiler().log() << "LD_PRELOAD intercept: " #name << std::endl;             \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libcudnn.so", RTLD_LAZY);                              \
      real_##name = (name##Func)dlsym(h, #name);                               \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");




typedef cudnnStatus_t (*cudnnCreateFunc)(cudnnHandle_t *handle);
extern "C" cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
  CUDNN_DLSYM_BOILERPLATE(cudnnCreate);

  auto a = make_cudnn_this_thread_now("cudnnCreate");
  auto api = std::make_shared<CudnnCreate>(a, handle);

//Disable this for now
//   profiler().log() << "WARN: tid " << get_thread_id()
//                    << " disabling CUPTI callbacks during cudnnCreate call"
//                    << std::endl;

//   profiler().driver().this_thread().pause_cupti_callbacks();

  profiler().driver().this_thread().api_enter(api);
  // //profiler().driver().this_thread().configured_call().start();  

  const cudnnStatus_t ret = real_cudnnCreate(handle);
  finalize_api(profiler());
  return ret;
}

typedef cudnnStatus_t (*cudnnDestroyFunc)(cudnnHandle_t handle);
extern "C" cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  CUDNN_DLSYM_BOILERPLATE(cudnnDestroy);

//Disable this for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnDestroy(handle);
//   }
//   profiler::err() << "WARN: disabling CUPTI callbacks during cudnnDestroy call"
//                   << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();
//   profiler::driver().this_thread().resume_cupti_callbacks();

  auto a = make_cudnn_this_thread_now("cudnnDestroy");
  auto api = std::make_shared<CudnnDestroy>(a, handle);
  profiler().driver().this_thread().api_enter(api);
  // //profiler().driver().this_thread().configured_call().start();  

  const cudnnStatus_t ret = real_cudnnDestroy(handle);
  finalize_api(profiler());
  return ret;
}


typedef cudnnStatus_t (*cudnnActivationForwardFunc)(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);

extern "C" cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {

  CUDNN_DLSYM_BOILERPLATE(cudnnActivationForward);

//Disabled for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnActivationForward(handle, activationDesc, alpha, xDesc, x,
//                                        beta, yDesc, y);
//   }
//   auto &allocations = profiler::allocations();
//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);
  // Get src value
//   auto xVal = allocations.find_value((uintptr_t)x, AS);
//   assert(xVal && "x should be on device");
  // Get dst allocation
//   auto yAlloc = allocations.find((uintptr_t)y, AS);
//   assert(yAlloc && "y alloc should be on device");
//   auto api = std::make_shared<ApiRecord>("cudnnActivationForward", devId);
  // FIXME - also depends on alpha, beta
//   auto yVal = yAlloc.new_value((uintptr_t)y, tensorSize(yDesc), true);
//   profiler::err()
//       << "WARN: disabling CUPTI callbacks during cudnnActivationForward call"
//       << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();
    auto a = make_cudnn_this_thread_now("cudnnActivationForward");
    auto api = std::make_shared<CudnnActivationForward>(a, handle, activationDesc, 
                                                        alpha, xDesc, x, beta,
                                                        yDesc, y);
    profiler().driver().this_thread().api_enter(api);
    // profiler().driver().this_thread().configured_call().start();  


    const cudnnStatus_t ret =real_cudnnActivationForward(handle,
                                                         activationDesc, 
                                                         alpha, xDesc, 
                                                         x, beta, yDesc, y);

    finalize_api(profiler());
//   profiler::driver().this_thread().resume_cupti_callbacks();

//   api->add_output(yVal);
//   api->add_input(xVal);
//   profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnAddTensorFunc)(cudnnHandle_t handle,
                                            const void *alpha,
                                            const cudnnTensorDescriptor_t aDesc,
                                            const void *A, const void *beta,
                                            const cudnnTensorDescriptor_t cDesc,
                                            void *C);
extern "C" cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void *alpha,
                                        const cudnnTensorDescriptor_t aDesc,
                                        const void *A, const void *beta,
                                        const cudnnTensorDescriptor_t cDesc,
                                        void *C) {
  CUDNN_DLSYM_BOILERPLATE(cudnnAddTensor);

//Disable for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
//   }
//   // FIXME - alpha and beta
//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);
//   auto &allocations = profiler::allocations();
//   // Get src value
//   auto aVal = allocations.find_value((uintptr_t)A, 1, AS);
//   assert(aVal && "A should be on device");
//   auto cVal = allocations.find_value((uintptr_t)C, 1, AS);
//   assert(cVal && "C should be on device");
//   auto api = std::make_shared<ApiRecord>("cudnnAddTensor", devId);
//   auto dstVal = allocations.duplicate_value(cVal, true);
//   profiler::err() << "WARN: thread " << cprof::model::get_thread_id()
//                   << " disabling CUPTI callbacks during cudnnAddTensor call"
//                   << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();
    auto a = make_cudnn_this_thread_now("cudnnAddTensor");
    auto api = std::make_shared<CudnnAddTensor>(a, handle, alpha, aDesc, A, beta, cDesc, C);
    profiler().driver().this_thread().api_enter(api);
    //profiler().driver().this_thread().configured_call().start();  
    const cudnnStatus_t ret = real_cudnnAddTensor(handle, alpha, aDesc, 
                                                  A, beta, cDesc, C);

    finalize_api(profiler());
//   profiler::driver().this_thread().resume_cupti_callbacks();
//   api->add_output(dstVal);
//   api->add_input(aVal);
//   api->add_input(cVal);
//   profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnActivationBackwardFunc)(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
extern "C" cudnnStatus_t cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {

  CUDNN_DLSYM_BOILERPLATE(cudnnActivationBackward);

//Disable for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y,
//                                         dyDesc, dy, xDesc, x, beta, dxDesc, dx);
//   }
//   auto &allocations = profiler::allocations();

//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);

  // Get src value
//   auto yVal = allocations.find_value((uintptr_t)y, 1, AS);
//   auto dyVal = allocations.find_value((uintptr_t)dy, 1, AS);
//   auto xVal = allocations.find_value((uintptr_t)x, 1, AS);
//   assert(yVal && "y should be on device");
//   assert(dyVal && "dy should be on device");
//   assert(xVal && "x should be on device");

//   Get dst allocation
//   auto dxAlloc = allocations.find((uintptr_t)dx, AS);
//   assert(dxAlloc && "dx alloc should be on device");
//   auto api = std::make_shared<ApiRecord>(
    //   "cudnnActivationBackward",
    //   profiler::driver().device_from_cudnn_handle(handle));

//   auto dxVal = dxAlloc.new_value((uintptr_t)dx, tensorSize(dxDesc), true);

//   profiler::err()
    //   << "WARN: disabling CUPTI callbacks during cudnnActivationBackward call"
    //   << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();

    auto a = make_cudnn_this_thread_now("cudnnActivationBackward");
    auto api = std::make_shared<CudnnActivationBackward>(a, handle, activationDesc,
                                                alpha, yDesc, y,
                                                dyDesc, dy,
                                                xDesc, x, beta,
                                                dxDesc, dx);
    profiler().driver().this_thread().api_enter(api);
    //profiler().driver().this_thread().configured_call().start();  
    const cudnnStatus_t ret = real_cudnnActivationBackward(handle, activationDesc, alpha, yDesc,
      y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
//   profiler::driver().this_thread().resume_cupti_callbacks();

  // FIXME: also depends on alpha, beta
//   api->add_output(dxVal);
//   api->add_input(xVal);
//   api->add_input(yVal);
//   api->add_input(dyVal);
//   profiler::atomic_out(api->to_json_string() + "\n");

  finalize_api(profiler());
  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionBackwardDataFunc)(
    cudnnHandle_t, const void *, const cudnnFilterDescriptor_t, const void *,
    const cudnnTensorDescriptor_t, const void *,
    const cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, void *,
    size_t, const void *, const cudnnTensorDescriptor_t, void *);
extern "C" cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionBackwardData);

//Disable this for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnConvolutionBackwardData(
//         handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
//         workSpaceSizeInBytes, beta, dxDesc, dx);
//   }

//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);
//   auto &allocations = profiler::allocations();

  // Find input values
//   auto dyVal = allocations.find_value((uintptr_t)dy, AS);
//   auto wVal = allocations.find_value((uintptr_t)w, AS);
//   auto workSpaceVal = allocations.find_value((uintptr_t)workSpace, AS);
//   auto dxVal = allocations.find_value((uintptr_t)dx, AS);

//   assert(dyVal &&
        //  "Couldn't find cudnnConvolutionBackwardData dy value on device");
//   assert(wVal && workSpaceVal && dxVal);
//   auto api = std::make_shared<ApiRecord>(
    //   "cudnnConvolutionBackwardData",
    //   profiler::driver().device_from_cudnn_handle(handle));

  // Create output value
//   auto outVal = allocations.duplicate_value(dxVal, true);

  // Do the actual call
//   profiler::err() << "WARN: disabling CUPTI callbacks during "
                    //  "cudnnConvolutionBackwardData call"
                //   << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();

    auto a = make_cudnn_this_thread_now("cudnnConvolutionBackwardData");
    auto api = std::make_shared<CudnnConvolutionBackwardData>(a, handle, alpha,
                                                              wDesc, w,
                                                              dyDesc, dy,
                                                              convDesc, algo, workSpace,
                                                              workSpaceSizeInBytes, beta,
                                                              dxDesc, dx);
    profiler().driver().this_thread().api_enter(api);
    //profiler().driver().this_thread().configured_call().start();  
    const cudnnStatus_t ret = real_cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc,
                                                                dy, convDesc, algo, workSpace, 
                                                                workSpaceSizeInBytes, beta, dxDesc, dx);

  finalize_api(profiler());
//   profiler::driver().this_thread().resume_cupti_callbacks();
//   api->add_output(outVal);
//   api->add_input(wVal);
//   api->add_input(dyVal);
//   api->add_input(workSpaceVal);
//   api->add_input(dxVal);
//   profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionBackwardBiasFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dbDesc, void *db);
extern "C" cudnnStatus_t
cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void *alpha,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy, const void *beta,
                             const cudnnTensorDescriptor_t dbDesc, void *db) {
  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionBackwardBias);

//Disabled for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta,
//                                              dbDesc, db);
//   }
//   auto &allocations = profiler::allocations();
//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);

//   // Find input values
//   auto dyVal = allocations.find_value((uintptr_t)dy, AS);

//   assert(dyVal &&
//          "Couldn't find cudnnConvolutionBackwardBias dy value on device");

  // Create output value
//   auto dbAlloc = allocations.find((uintptr_t)db, 1, AS);
//   assert(dbAlloc && "y allocation should be on device");
//   auto api = std::make_shared<ApiRecord>(
//       "cudnnConvolutionBackwardBias",
//       profiler::driver().device_from_cudnn_handle(handle));
//   auto dbVal = dbAlloc.new_value((uintptr_t)db, tensorSize(dbDesc),
//                                  true /*initialized*/);

  // Do the actual call
//   profiler::err() << "WARN: disabling CUPTI callbacks during "
//                      "cudnnConvolutionBackwardBias call"
//                   << std::endl;

//   profiler::driver().this_thread().pause_cupti_callbacks();

    auto a = make_cudnn_this_thread_now("cudnnConvolutionBackwardBias");
    auto api = std::make_shared<CudnnConvolutionBackwardBias>(a, handle, alpha,
                                                              dyDesc, dy, beta, 
                                                              dbDesc, db);
    profiler().driver().this_thread().api_enter(api);
    //profiler().driver().this_thread().configured_call().start();  

    const cudnnStatus_t ret =real_cudnnConvolutionBackwardBias(handle, alpha,
                                                               dyDesc, dy, beta, 
                                                               dbDesc, db);

  
    finalize_api(profiler());
//   profiler::driver().this_thread().resume_cupti_callbacks();

//   api->add_output(dbVal);
//   api->add_input(dyVal);
//   profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionBackwardFilterFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw);
extern "C" cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {

  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionBackwardFilter);

//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnConvolutionBackwardFilter(
//         handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
//         workSpaceSizeInBytes, beta, dwDesc, dw);
//   }

//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);
//   auto &allocations = profiler::allocations();

//   // Find input values
//   auto xVal = allocations.find_value((uintptr_t)x, AS);
//   auto dyVal = allocations.find_value((uintptr_t)dy, AS);
//   auto workSpaceVal = allocations.find_value((uintptr_t)workSpace, AS);
//   auto dwVal = allocations.find_value((uintptr_t)dw, AS);
//   assert(
//       xVal && dyVal && workSpaceVal && dwVal &&
//       "Couldn't find cudnnConvolutionBackwardFilter argument value on device");
//   auto api = std::make_shared<ApiRecord>(
//       "cudnnConvolutionBackwardFilter",
//       profiler::driver().device_from_cudnn_handle(handle));

//   // See if there is an existing output value to take info from
//   auto outVal = allocations.duplicate_value(dwVal, true);

//   profiler::err() << "[cudnnConvolutionBackwardFilter] " << outVal
//                   << " deps on " << xVal << " " << dyVal << " " << workSpaceVal
//                   << " " << dwVal << std::endl;

//   profiler::err() << "WARN: disabling CUPTI callbacks during "
//                      "cudnnConvolutionBackwardFilter call"
//                   << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();
    auto a = make_cudnn_this_thread_now("cudnnConvolutionBackwardFilter");
    auto api = std::make_shared<CudnnConvolutionBackwardFilter>(a, handle, alpha,
                                                                xDesc, x,
                                                                dyDesc, dy,
                                                                convDesc, algo, workSpace,
                                                                workSpaceSizeInBytes, beta,
                                                                dwDesc, dw);
    profiler().driver().this_thread().api_enter(api);
    //profiler().driver().this_thread().configured_call().start();  

    const cudnnStatus_t ret = real_cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc,
                                                                  dy, convDesc, algo, workSpace, 
                                                                  workSpaceSizeInBytes, beta, dwDesc, dw);
    finalize_api(profiler());
//   profiler::driver().this_thread().resume_cupti_callbacks();

//   api->add_output(outVal);
//   api->add_input(xVal);
//   api->add_input(dyVal);
//   api->add_input(workSpaceVal);
//   api->add_input(dwVal);
//   profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionForwardFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

extern "C" cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {

  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionForward);

//Disable this for now
//   if (preload_cudnn::is_passthrough()) {
//     return real_cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w,
//                                         convDesc, algo, workSpace,
//                                         workSpaceSizeInBytes, beta, yDesc, y);
//   }

//   const int devId = profiler::driver().device_from_cudnn_handle(handle);
//   AddressSpace AS = profiler::hardware().address_space(devId);
//   auto &allocations = profiler::allocations();

//   // Find input values
//   profiler::err() << "Looking for x=" << (uintptr_t)x << ", w=" << (uintptr_t)w
//                   << ", workSpace=" << (uintptr_t)workSpace << std::endl;
//   auto xVal = allocations.find_value((uintptr_t)x, AS);
//   auto wVal = allocations.find_value((uintptr_t)w, AS);
//   auto workSpaceVal = allocations.find_value((uintptr_t)workSpace, AS);
//   auto yVal = allocations.find_value((uintptr_t)y, AS);
//   assert(xVal && wVal && workSpaceVal && yVal &&
//          "Couldn't find cudnnConvolutionForward argument value on device");
//   auto api = std::make_shared<ApiRecord>(
//       "cudnnConvolutionForward",
//       profiler::driver().device_from_cudnn_handle(handle));

//   // See if there is an existing output value to take info from
//   auto outVal = allocations.duplicate_value(yVal, true);

//   profiler::err() << "[cudnnConvolutionForward] " << outVal << " deps on "
//                   << yVal << " " << xVal << " " << wVal << " " << workSpaceVal
//                   << std::endl;
//   profiler::err()
//       << "WARN: thread " << cprof::model::get_thread_id()
//       << " disabling CUPTI callbacks during cudnnConvolutionForward call"
//       << std::endl;
//   profiler::driver().this_thread().pause_cupti_callbacks();


  auto a = make_cudnn_this_thread_now("cudnnConvolutionBackwardFilter");
  auto api = std::make_shared<CudnnConvolutionForward>(a, handle, alpha,
                                                       xDesc, x,
                                                       wDesc, w,
                                                       convDesc,
                                                       algo, workSpace,
                                                       workSpaceSizeInBytes, beta,
                                                       yDesc, y);
  profiler().driver().this_thread().api_enter(api);
  //profiler().driver().this_thread().configured_call().start();  

  const cudnnStatus_t ret = real_cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w,
                                                         convDesc, algo, workSpace, 
                                                         workSpaceSizeInBytes, beta, yDesc, y);

  finalize_api(profiler());
  // profiler::driver().this_thread().resume_cupti_callbacks();

  // api->add_output(outVal);
  // api->add_input(xVal);
  // api->add_input(wVal);
  // api->add_input(workSpaceVal);
  // api->add_input(yVal);
  // profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnSoftmaxForwardFunc)(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
extern "C" cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {

  CUDNN_DLSYM_BOILERPLATE(cudnnSoftmaxForward);

//Disable
  // if (preload_cudnn::is_passthrough()) {
  //   return real_cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta,
  //                                   yDesc, y);
  // }

  // auto &allocations = profiler::allocations();

  // const int devId = profiler::driver().device_from_cudnn_handle(handle);
  // AddressSpace AS = profiler::hardware().address_space(devId);

  // // Find input values
  // auto xVal = allocations.find_value((uintptr_t)x, AS);
  // assert(xVal && "Couldn't find cudnnSoftmaxForward x value on device");
  // auto api = std::make_shared<ApiRecord>("cudnnSoftmaxForward", devId);

  // // Create output value
  // auto yAlloc = allocations.find((uintptr_t)y, 1, AS);
  // assert(yAlloc && "y allocation should be on device");

  // auto yVal =
  //     yAlloc.new_value((uintptr_t)y, tensorSize(yDesc), true /*initialized*/);


  // // Do the actual call
  // profiler::err()
  //     << "WARN: disabling CUPTI callbacks during cudnnSoftmaxForward call"
  //     << std::endl;
  // profiler::driver().this_thread().pause_cupti_callbacks();


  auto a = make_cudnn_this_thread_now("cudnnSoftmaxForward");
  auto api = std::make_shared<CudnnSoftmaxForward>(a, handle, algo, mode,
                                                   alpha, xDesc, x,
                                                   beta, yDesc, y);
  profiler().driver().this_thread().api_enter(api);
  //profiler().driver().this_thread().configured_call().start();  


  const cudnnStatus_t ret =
      real_cudnnSoftmaxForward(handle, algo, mode,
                        alpha, xDesc, x, beta, yDesc, y);
  finalize_api(profiler());
  // profiler::driver().this_thread().resume_cupti_callbacks();

  // api->add_output(yVal);
  // api->add_input(xVal);
  // profiler::atomic_out(api->to_json_string() + "\n");

  return ret;
}

typedef cudnnStatus_t (*cudnnPoolingForwardFunc)(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
extern "C" cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {

  CUDNN_DLSYM_BOILERPLATE(cudnnPoolingForward);
  // if (preload_cudnn::is_passthrough()) {
  //   return real_cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta,
  //                                   yDesc, y);
  // }

  // auto &allocations = profiler::allocations();

  // const int devId = profiler::driver().device_from_cudnn_handle(handle);
  // AddressSpace AS = profiler::hardware().address_space(devId);

  // // Find input values
  // auto xVal = allocations.find_value((uintptr_t)x, AS);
  // assert(xVal && "Couldn't find cudnnPoolingForward x value on device");
  // auto api = std::make_shared<ApiRecord>("cudnnPoolingForward", devId);

  // // FIXME: ignoring alpha, beta

  // // Create output value
  // const size_t ySize = tensorSize(yDesc);
  // auto yAlloc = allocations.find((uintptr_t)y, ySize, AS);
  // assert(yAlloc && "y allocation should be on device");

  // auto yVal = yAlloc.new_value((uintptr_t)y, ySize, true /*initialized*/);

  // // Do the actual call
  // profiler::err()
  //     << "WARN: disabling CUPTI callbacks during cudnnPoolingForward call"
  //     << std::endl;
  // profiler::driver().this_thread().pause_cupti_callbacks();
  auto a = make_cudnn_this_thread_now("cudnnSoftmaxForward");
  auto api = std::make_shared<CudnnPoolingForward>(a, handle, poolingDesc,
                                                   alpha, xDesc, x,
                                                   beta, yDesc, y);
  profiler().driver().this_thread().api_enter(api);
  //profiler().driver().this_thread().configured_call().start();  


  const cudnnStatus_t ret =
      real_cudnnPoolingForward(handle, poolingDesc,
                        alpha, xDesc, x, beta, yDesc, y);

  finalize_api(profiler());
  // profiler::driver().this_thread().resume_cupti_callbacks();

  // api->add_output(yVal);
  // api->add_input(xVal);
  // profiler::atomic_out(api->to_json_string() + "\n");
  return ret;
}

// cudnnPoolingBackward
// cudnnSoftmaxBackward
// cudnnSpatialTfGridGeneratorForward
// cudnnLRNCrossChannelBackward
// cudnnBatchNormalizationBackward
// cudnnBatchNormalizationForwardInference
// cudnnSpatialTfSamplerForward
// cudnnSpatialTfGridGeneratorBackward
// cudnnRNNForwardTraining
// cudnnRNNForwardInference
// cudnnRNNBackwardWeights
// cudnnRNNBackwardData
