#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <unistd.h>
#include <limits.h>
#include <random>

#include <sys/types.h>
#include <sys/stat.h>

#include "tensorboard_logger.h"


std::string gen_log_filename(std::string game, std::string key) {
  // hostname
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);

  // random number, for safety
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<int> dist(1, 1000);

  // time
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  // combine it all
  std::ostringstream oss;
  oss << "runs/"
    << std::put_time(&tm, "%b%d-%H:%M:%S-")
    << dist(generator) << "-"
    << std::string(hostname) << "-"
    << game << "-"
    << key;

  mkdir(oss.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  return oss.str() + "/event.tfevents";
}


void init_logger(std::string log_file) {
  TensorBoardLogger logger(log_file.c_str());
}
