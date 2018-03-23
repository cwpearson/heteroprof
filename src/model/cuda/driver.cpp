#include <cassert>

#include "model/cuda/api.hpp"
#include "model/cuda/driver.hpp"
#include "util_cupti.hpp"

using model::cuda::Api;
using model::cuda::ThreadState;

void ThreadState::api_enter(Api *a) { apiStack_.push_back(a); }

Api *ThreadState::api_exit() {
  assert(!apiStack_.empty());
  const auto current = apiStack_.back();
  apiStack_.pop_back();
  return current;
}

const Api *ThreadState::parent_api() const {
  assert(apiStack_.size() >= 2);
  return apiStack_[apiStack_.size() - 2];
}

Api *ThreadState::current_api() {
  assert(!apiStack_.empty());
  return apiStack_.back();
}

void ThreadState::pause_cupti_callbacks() {
  assert(cuptiCallbacksEnabled_);
  cuptiCallbacksEnabled_ = false;
}
void ThreadState::resume_cupti_callbacks() {
  assert(!cuptiCallbacksEnabled_);
  cuptiCallbacksEnabled_ = true;
}