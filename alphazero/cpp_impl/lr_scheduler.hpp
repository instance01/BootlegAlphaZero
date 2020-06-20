#ifndef SCHEDULER_HEADER
#define SCHEDULER_HEADER
#include <vector>

class LRScheduler {
  public:
    LRScheduler();
    ~LRScheduler();

    virtual float step(float lr_before, int eps, double eval_reward) {return 0;};
};

class ExponentialScheduler : public LRScheduler {
  public:
    float factor;

    ExponentialScheduler(float factor) : factor(factor) {};
    ~ExponentialScheduler() {};

    float step(float lr_before, int eps, double eval_reward);
};

class StepScheduler : public LRScheduler {
  public:
    std::vector<int> step_downs;
    float factor;

    StepScheduler(std::vector<int> step_downs, float factor) : step_downs(step_downs), factor(factor) {};
    ~StepScheduler() {};

    float step(float lr_before, int eps, double eval_reward);
};

class ReduceOnGoodEval : public LRScheduler {
  public:
    double min_good_eval;
    int min_n_good_evals;
    float factor;

    int n_good_evals;

    ReduceOnGoodEval(float factor, double min_good_eval, int min_n_good_evals)
      : min_good_eval(min_good_eval), min_n_good_evals(min_n_good_evals), factor(factor) {};
    ~ReduceOnGoodEval() {};

    float step(float lr_before, int eps, double eval_reward);
};
#endif
