#include <cassert>
#include <dlfcn.h>

#include <cublas_v2.h>

#include "cublas/preload.hpp"

// #include "cprof/model/driver.hpp"
// #include "cprof/model/thread.hpp"
// #include "cprof/util_numa.hpp"

#include "profiler.hpp"

//Beginning of include files
#include "cublas/cublas_create.hpp"
#include "cublas/cublas_destroy.hpp"
#include "cublas/cublas_dgemm.hpp"
#include "cublas/cublas_dgemv.hpp"
#include "cublas/cublas_sasum.hpp"
#include "cublas/cublas_saxpy.hpp"
#include "cublas/cublas_sdot.hpp"
#include "cublas/cublas_sgemm.hpp"
#include "cublas/cublas_sgemv.hpp"
#include "cublas/cublas_sscal.hpp"

namespace cublas {
using sys::get_thread_id;    
Profiler *profiler_ = nullptr;
void set_profiler(Profiler &p) { profiler_ = &p; }
Profiler &profiler() {
  assert(profiler_);
  return *profiler_;
}

Cublas make_cublas_this_thread_now(std::string name) {
  auto now = std::chrono::high_resolution_clock::now();
  auto tid = get_thread_id();
  Cublas cublas(tid, name);
  cublas.set_wall_start(now);
  return cublas;
}

void finalize_api(Profiler &p){
    auto api = p.driver().this_thread().current_api();
    api->set_wall_end(std::chrono::high_resolution_clock::now());
    p.record(api->to_json());
    p.driver().this_thread().api_exit(); 
}
} // namespace cublas

using namespace cublas;
using sys::get_thread_id;

#define CUBLAS_DLSYM_BOILERPLATE(name)                                         \
  static name##Func real_##name = nullptr;                                     \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libcublas.so", RTLD_LAZY);                             \
      real_##name = (name##Func)dlsym(h, #name "_v2");                         \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");



typedef cublasStatus_t (*cublasCreateFunc)(cublasHandle_t *handle);
extern "C" cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  CUBLAS_DLSYM_BOILERPLATE(cublasCreate);

  auto a = make_cublas_this_thread_now("cublasCreate");
  auto api = std::make_shared<CublasCreate>(a, handle);

  profiler().driver().this_thread().api_enter(api);

  const cublasStatus_t ret = real_cublasCreate(handle);
  finalize_api(profiler());

  return ret;
}

typedef cublasStatus_t (*cublasDestroyFunc)(cublasHandle_t handle);
extern "C" cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  CUBLAS_DLSYM_BOILERPLATE(cublasDestroy);

  auto a = make_cublas_this_thread_now("cublasDestroy");
  auto api = std::make_shared<CublasDestroy>(a, handle);

  profiler().driver().this_thread().api_enter(api);

  const cublasStatus_t ret = real_cublasDestroy(handle);
  finalize_api(profiler());

  return ret;
}

typedef cublasStatus_t (*cublasDgemmFunc)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc);
extern "C" cublasStatus_t
cublasDgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k, const double *alpha,
            const double *A, int lda, const double *B, int ldb,
            const double *beta, double *C, int ldc) {
  CUBLAS_DLSYM_BOILERPLATE(cublasDgemm);

  auto a = make_cublas_this_thread_now("cublasDgemm");
  auto api = std::make_shared<CublasDGemm>(a, handle, transa,
                                           transb, m, n, k, alpha,
                                           A, lda, B, ldb,
                                           beta, C, ldc);
  profiler().driver().this_thread().api_enter(api);

  const cublasStatus_t ret = real_cublasDgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);

  finalize_api(profiler());

  return ret;
}

typedef cublasStatus_t (*cublasSaxpyFunc)(
    cublasHandle_t handle, int n,
    const float *alpha, /* host or device pointer */
    const float *x, int incx, float *y, int incy);
extern "C" cublasStatus_t
cublasSaxpy(cublasHandle_t handle, int n,
            const float *alpha, /* host or device pointer */
            const float *x, int incx, float *y, int incy) {

  CUBLAS_DLSYM_BOILERPLATE(cublasSaxpy);



  auto a = make_cublas_this_thread_now("cublasSaxpy");
  auto api = std::make_shared<CublasSaxpy>(a, handle, n,
                                           alpha, /* host or device pointer */
                                           x, incx, y, incy);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  
  const cublasStatus_t ret = real_cublasSaxpy(handle, n,
                                              alpha, x, incx, y, incy);

  finalize_api(profiler());
  return ret;
}

typedef cublasStatus_t (*cublasSgemmFunc)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, /* host or device pointer */
    const float *A, int lda, const float *B, int ldb,
    const float *beta, /* host or device pointer */
    float *C, int ldc);
extern "C" cublasStatus_t
cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k,
            const float *alpha, /* host or device pointer */
            const float *A, int lda, const float *B, int ldb,
            const float *beta, /* host or device pointer */
            float *C, int ldc) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSgemm);

  auto a = make_cublas_this_thread_now("cublasSgemm");
  auto api = std::make_shared<CublasSgemm>(a, handle, transa,
                                           transb, m, n, k,
                                           alpha, /* host or device pointer */
                                           A, lda, B, ldb,
                                           beta, /* host or device pointer */
                                           C, ldc);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  

  const cublasStatus_t ret =
      real_cublasSgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);

  finalize_api(profiler());

  return ret;
}

typedef cublasStatus_t (*cublasDgemvFunc)(cublasHandle_t, cublasOperation_t,
                                          int, int, const double *,
                                          const double *, int, const double *,
                                          int, const double *, double *, int);
extern "C" cublasStatus_t cublasDgemv(cublasHandle_t handle,
                                      cublasOperation_t trans, int m, int n,
                                      const double *alpha, const double *A,
                                      int lda, const double *x, int incx,
                                      const double *beta, double *y, int incy) {
  CUBLAS_DLSYM_BOILERPLATE(cublasDgemv);

  auto a = make_cublas_this_thread_now("cublasDgemv");
  auto api = std::make_shared<CublasDgemv>(a, handle,
                                           trans, m, n,
                                           alpha, A,
                                           lda, x, incx,
                                           beta, y, incy);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  

  const cublasStatus_t ret =
      real_cublasDgemv(handle, trans, m, n, alpha, A,
                        lda, x, incx, beta, y, incy);

  finalize_api(profiler());
  return ret;
}

typedef cublasStatus_t (*cublasSgemvFunc)(cublasHandle_t handle,
                                          cublasOperation_t trans, int m, int n,
                                          const float *alpha, const float *A,
                                          int lda, const float *x, int incx,
                                          const float *beta, float *y,
                                          int incy);
extern "C" cublasStatus_t cublasSgemv(cublasHandle_t handle,
                                      cublasOperation_t trans, int m, int n,
                                      const float *alpha, const float *A,
                                      int lda, const float *x, int incx,
                                      const float *beta, float *y, int incy) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSgemv);

  auto a = make_cublas_this_thread_now("cublasSgemv");
  auto api = std::make_shared<CublasSgemv>(a, handle,
                                           trans, m, n,
                                           alpha, A,
                                           lda, x, incx,
                                           beta, y, incy);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  

  const cublasStatus_t ret =
      real_cublasSgemv(handle, trans, m, n, alpha, A,
                        lda, x, incx, beta, y, incy);
  finalize_api(profiler());

  return ret;
}

typedef cublasStatus_t (*cublasSasumFunc)(cublasHandle_t, int, const float *,
                                          int, float *);
extern "C" cublasStatus_t cublasSasum(cublasHandle_t handle, int n,
                                      const float *x, int incx, float *result) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSasum);

  auto a = make_cublas_this_thread_now("cublasSasum");
  auto api = std::make_shared<CublasSasum>(a, handle, n,
                                           x, incx, result);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  

  const cublasStatus_t ret =
      real_cublasSasum(handle, n, x, incx, result);

  finalize_api(profiler());
  return ret;
}

typedef cublasStatus_t (*cublasSscalFunc)(
    cublasHandle_t handle, int n,
    const float *alpha, /* host or device pointer */
    float *x, int incx);
extern "C" cublasStatus_t
cublasSscal(cublasHandle_t handle, int n,
            const float *alpha, /* host or device pointer */
            float *x, int incx) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSscal);

  auto a = make_cublas_this_thread_now("cublasSscal");
  auto api = std::make_shared<CublasSscal>(a, handle, n,
                                           alpha, x, incx);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  

  const cublasStatus_t ret =
      real_cublasSscal(handle, n, alpha, x, incx);

  finalize_api(profiler());
  return ret;
}

typedef cublasStatus_t (*cublasSdotFunc)(cublasHandle_t handle, int n,
                                         const float *x, int incx,
                                         const float *y, int incy,
                                         float *result);
extern "C" cublasStatus_t cublasSdot(cublasHandle_t handle, int n,
                                     const float *x, int incx, const float *y,
                                     int incy, float *result) {

  CUBLAS_DLSYM_BOILERPLATE(cublasSdot);

  auto a = make_cublas_this_thread_now("cublasSdot");
  auto api = std::make_shared<CublasSdot>(a, handle, n,
                                          x, incx, y, incy,
                                          result);
  profiler().driver().this_thread().api_enter(api);
//  profiler().driver().this_thread().configured_call().start();  

  auto ret = real_cublasSdot(handle, n, x, incx, y,
                               incy, result);

  finalize_api(profiler());
  return ret;
}
