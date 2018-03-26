#include <cassert>

#include <nlohmann/json.hpp>

#include "cuda/api.hpp"
#include "cuda/cupti/callback/api.hpp"
#include "sys/thread.hpp"

using json = nlohmann::json;
using namespace cuda::cupti::callback;
using tid_t = sys::tid_t;

Api::Api(const tid_t callingThread, const CUpti_CallbackData *cbdata,
         const CUpti_CallbackDomain domain)
    : cuda::Api(callingThread, cbdata->functionName),
      contextUid_(cbdata->contextUid), correlationId_(cbdata->correlationId),
      domain_(domain) {}

json Api::to_json() const {
  json j = cuda::Api::to_json();
  j["context_uid"] = contextUid_;
  j["correlation_id"] = correlationId_;
  j["symbol_name"] = symbolName_;
  return j;
}
