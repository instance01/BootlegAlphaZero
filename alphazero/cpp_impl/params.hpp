#ifndef PARAMS_HEADER
#define PARAMS_HEADER
#include <map>
#include <string>
#include <variant>
#include <vector>

using Params = std::map<std::string, std::variant<double, std::string, std::vector<int>, int>>;
#endif
