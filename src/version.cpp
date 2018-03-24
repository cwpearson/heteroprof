#include "version.hpp"

const std::string v = "0.1.0";
const std::string dirty = GIT_DIRTY;
const std::string buildDate = BUILD_DATE;

std::string version_full() {
  auto s = v + "-" + GIT_DIRTY + "-" + BUILD_DATE;
  return s;
}

std::string version() { return v; }

std::string version_git() { return dirty; }

std::string version_build() { return buildDate; }