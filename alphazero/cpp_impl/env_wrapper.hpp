#ifndef ENV_W_HEADER
#define ENV_W_HEADER
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <mutex>
#include <memory>

#include "envs/env.hpp"
#include "cfg.hpp"


class EnvWrapper {
  public:
    std::shared_ptr<Env> env;
    json params;
    std::string game;

    int reward_exponent;

    EnvWrapper() {};
    ~EnvWrapper() {};

    void init(std::string game, json params);

    std::tuple<std::vector<float>, double, bool> step(int action);
    std::vector<float> reset();
    std::unique_ptr<EnvWrapper> clone();
};
#endif
