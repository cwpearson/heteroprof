#ifndef CUDNN_UTIL_HPP
#define CUDNN_UTIL_HPP

#include <cstdlib>
#include <ostream>

#include <cudnn.h>

#define CUDNN_CHECK(ans, err)                                                  \
  { cudnnAssert((ans), __FILE__, __LINE__, (err)); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line,
                        std::ostream &err, bool abort = true) {
  if (code != CUDNN_STATUS_SUCCESS) {
    const char *errstr = cudnnGetErrorString(code);
    err << "CUDNN_CHECK: " << errstr << " " << file << " " << line << std::endl;
    if (abort)
      exit(code);
  }
}

size_t tensorSize(const cudnnTensorDescriptor_t tensorDesc);
#endif