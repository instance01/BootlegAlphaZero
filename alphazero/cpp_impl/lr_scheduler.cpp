#include "lr_scheduler.hpp"


LRScheduler::LRScheduler() {};
LRScheduler::~LRScheduler() {};


float
StepScheduler::step(float lr_before, int eps, double eval_reward) {
  for (auto step_down : step_downs) {
    if (eps >= step_down) {
      step_downs.erase(step_downs.begin());
      return std::max(min_lr, lr_before * factor);
    }
  }
  return std::max(min_lr, lr_before);
}

float
ExponentialScheduler::step(float lr_before, int eps, double eval_reward) {
  return std::max(min_lr, lr_before * factor);
}

float
ReduceOnGoodEval::step(float lr_before, int eps, double eval_reward) {
  if (eval_reward > min_good_eval)
    n_good_evals += 1;
  else if (consecutive)
    n_good_evals = 0;

  if (n_good_evals > min_n_good_evals) {
    n_good_evals = 0;
    return std::max(min_lr, lr_before * factor);
  }
  return std::max(min_lr, lr_before);
}
