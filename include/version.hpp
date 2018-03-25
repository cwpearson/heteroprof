#ifndef VERSION_HPP
#define VERSION_HPP

#include <string>

std::string version_full();
const char *version();
const char *version_build();
const char *version_git();

#endif