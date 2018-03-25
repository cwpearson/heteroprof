#include "version.hpp"

#include <iostream>

const char *v = "0.1.0";
const char *dirty = GIT_DIRTY;
const char *buildDate = BUILD_DATE;

std::string version_full() {
  auto s = v + std::string("-") + GIT_DIRTY + "-" + BUILD_DATE;
  return s;
}

const char *version() { return v; }

const char *version_git() { return dirty; }

const char *version_build() { return buildDate; }