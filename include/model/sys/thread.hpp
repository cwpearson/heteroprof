#ifndef CPROF_MODEL_THREAD_HPP
#define CPROF_MODEL_THREAD_HPP

#include <sys/types.h>

namespace model {
namespace sys {

typedef pid_t tid_t;

tid_t get_thread_id();

} // namespace sys
} // namespace model

#endif