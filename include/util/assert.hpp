#ifndef UTIL_ASSERT_HPP
#define UTIL_ASSERT_HPP

#include <cassert>
#include <iostream>

#include "util/backtrace.hpp"

#define ASSERT(expr) { \
if (!(expr)) { \
  print_backtrace(std::cerr); \
} \
 \
  assert(expr); \
 \
}

#endif
