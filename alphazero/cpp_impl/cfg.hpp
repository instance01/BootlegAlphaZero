#ifndef JSON_HEADER
#define JSON_HEADER
#include <nlohmann/json.hpp>

using json = nlohmann::json;

json load_cfg(std::string param_num);
#endif
