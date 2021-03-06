#ifndef SCHEDULER_HEADER
#define SCHEDULER_HEADER
#include <vector>

class LRScheduler {
  public:
    LRScheduler();
    virtual ~LRScheduler();

    virtual float step(float lr_before, int eps, double eval_reward) {return 0;};
};

class ExponentialScheduler : public LRScheduler {
  public:
    float factor;
    float min_lr;

    ExponentialScheduler(float factor, float min_lr) : factor(factor), min_lr(min_lr) {};
    ~ExponentialScheduler() {};

    float step(float lr_before, int eps, double eval_reward);
};

class StepScheduler : public LRScheduler {
  public:
    std::vector<int> step_downs;
    float factor;
    float min_lr;

    StepScheduler(std::vector<int> step_downs, float factor, float min_lr)
      : step_downs(step_downs), factor(factor), min_lr(min_lr) {};
    ~StepScheduler() {};

    float step(float lr_before, int eps, double eval_reward);
};

class ReduceOnGoodEval : public LRScheduler {
  public:
    double min_good_eval = -100.;
    int min_n_good_evals = 10;
    float factor = 0.5;
    float min_lr = 0.000001;
    bool consecutive = false;

    int n_good_evals = 0;

    ReduceOnGoodEval(float factor, double min_good_eval, int min_n_good_evals, float min_lr, bool consecutive)
      : min_good_eval(min_good_eval), min_n_good_evals(min_n_good_evals), factor(factor), min_lr(min_lr), consecutive(consecutive) {};
    ~ReduceOnGoodEval() {};

    float step(float lr_before, int eps, double eval_reward);
};
#endif
