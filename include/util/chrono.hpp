#ifndef UTIL_CHRONO_HPP
#define UTIL_CHRONO_HPP

#include <chrono>

uint64_t epoch_nanos(const std::chrono::high_resolution_clock::time_point &t) {
  auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(t);
  auto epoch = now_ns.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
  return value.count();
}

#endif