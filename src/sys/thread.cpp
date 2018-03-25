#include <sys/syscall.h>
#include <unistd.h>

#include "sys/thread.hpp"

namespace sys {

typedef pid_t tid_t;

tid_t get_thread_id() { return syscall(SYS_gettid); }
} // namespace sys
