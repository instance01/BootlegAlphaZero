#ifndef TENSORBOARD_UTIL_HEADER
#define TENSORBOARD_UTIL_HEADER
#include <string>

std::string gen_log_filename(std::string game, std::string key);
void init_logger(std::string log_file);
#endif
