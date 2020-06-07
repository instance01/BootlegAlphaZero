#include "tensorboard_logger.h"


void init_logger(std::string log_file) {
  TensorBoardLogger logger(log_file.c_str());
}
