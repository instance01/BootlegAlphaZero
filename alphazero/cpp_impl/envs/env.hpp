#ifndef ENV_HEADER
#define ENV_HEADER
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <mutex>
#include <memory>


class Env {
  public:
    Env();
    ~Env();

    // For A2C. Just needs to be rough.
    std::vector<float> expected_mean;
    std::vector<float> expected_stddev;

    virtual std::tuple<std::vector<float>, double, bool> step(int action) {return {};};
    virtual std::vector<float> reset() {return {};};
};
#endif
