#ifndef CPROF_ALLOCATION_HPP
#define CPROF_ALLOCATION_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>

#include <nlohmann/json.hpp>

#include "address_space.hpp"
#include "model/location.hpp"
#include "model/memory.hpp"
#include "model/thread.hpp"
#include "util/interval_set.hpp"
#include "util/logging.hpp"

class Allocation {

public:
  uintptr_t lower_;
  uintptr_t upper_;
  AddressSpace address_space_;
  cprof::model::Memory memory_;
  cprof::model::tid_t thread_id_;
  cprof::model::Location location_;

  typedef uintptr_t pos_type;
  pos_type lower() const noexcept { return lower_; }
  pos_type upper() const noexcept { return upper_; }
  void set_lower(const pos_type &p) { lower_ = p; }
  void set_upper(const pos_type &p) { upper_ = p; }

  Allocation(uintptr_t pos, size_t size, const AddressSpace &as,
             const cprof::model::Memory &mem,
             const cprof::model::Location &location)
      : val_(next_val_++), val_size_(size), val_initialized_(false),
        lower_(pos), upper_(pos + size), address_space_(as), memory_(mem),
        location_(location), freed_(false) {}
  Allocation(const uintptr_t pos, const size_t size)
      : Allocation(pos, size, AddressSpace::Host(),
                   cprof::model::Memory::Unknown,
                   cprof::model::Location::Unknown()) {}
  Allocation() : Allocation(0, 0) {}

  nlohmann::json to_json() const;
  std::string to_json_string() const;

  pos_type pos() const noexcept { return lower_; }
  const AddressSpace &address_space() const { return address_space_; }
};

#endif
