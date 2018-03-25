#include <cassert>

#include "cuda/api.hpp"
#include "cuda/driver.hpp"

using cuda::Api;
using cuda::ThreadState;
using ApiRef = ThreadState::ApiRef;

void ThreadState::api_enter(ApiRef a) { apiStack_.push_back(a); }

ApiRef ThreadState::api_exit() {
  assert(!apiStack_.empty());
  const auto current = apiStack_.back();
  apiStack_.pop_back();
  return current;
}

const ApiRef ThreadState::parent_api() const {
  assert(apiStack_.size() >= 2);
  return apiStack_[apiStack_.size() - 2];
}

ApiRef ThreadState::current_api() {
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