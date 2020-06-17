#include "lr_scheduler.hpp"


LRScheduler::LRScheduler() {};
LRScheduler::~LRScheduler() {};


float
StepScheduler::step(float lr_before, int eps) {
  for (auto step_down : step_downs) {
    if (eps >= step_down) {
      step_downs.erase(step_downs.begin());
      return lr_before * factor;
    }
  }
  return lr_before;
}

float
ExponentialScheduler::step(float lr_before, int eps) {
  return lr_before * factor;
}
