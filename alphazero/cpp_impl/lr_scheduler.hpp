#ifndef SCHEDULER_HEADER
#define SCHEDULER_HEADER
#include <vector>

class LRScheduler {
  public:
    LRScheduler();
    ~LRScheduler();

    virtual float step(float lr_before, int eps) {return 0;};
};

class ExponentialScheduler : public LRScheduler {
  public:
    float factor;

    ExponentialScheduler(float factor) : factor(factor) {};
    ~ExponentialScheduler() {};

    float step(float lr_before, int eps);
};

class StepScheduler : public LRScheduler {
  public:
    std::vector<int> step_downs;
    float factor;

    StepScheduler(std::vector<int> step_downs, float factor) : step_downs(step_downs), factor(factor) {};
    ~StepScheduler() {};

    float step(float lr_before, int eps);
};
#endif
