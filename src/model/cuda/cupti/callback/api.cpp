#include <cassert>

#include <nlohmann/json.hpp>

#include "model/cuda/api.hpp"
#include "model/cuda/cupti/callback/api.hpp"
#include "model/sys/thread.hpp"

using json = nlohmann::json;
using namespace model::cuda::cupti::callback;
using tid_t = model::sys::tid_t;

Api::Api(const tid_t callingThread, const CUpti_CallbackData *cbdata)
    : model::cuda::Api(callingThread, cbdata->functionName),
      contextUid_(cbdata->contextUid), correlationId_(cbdata->correlationId) {}

json Api::to_json() const {
  json j = model::cuda::Api::to_json();
  j["context_uid"] = contextUid_;
  j["correlation_id"] = correlationId_;
  j["symbol_name"] = symbolName_;
  return j;
}
