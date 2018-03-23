#include <sys/syscall.h>
#include <unistd.h>

#include "model/sys/thread.hpp"

namespace model {
namespace sys {

typedef pid_t tid_t;

tid_t get_thread_id() { return syscall(SYS_gettid); }
} // namespace sys
} // namespace model
