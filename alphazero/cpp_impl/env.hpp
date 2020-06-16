#ifndef ENV_HEADER
#define ENV_HEADER
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <mutex>
#include <memory>

#include "envs/gridworld.hpp"
#include "cfg.hpp"


class Env {
  public:
    GridWorldEnv grid_world;

    int reward_exponent;

    Env(){};
    ~Env(){};

    void init(std::string game, json params);

    std::tuple<std::vector<int>, double, bool> step(int action);
    std::vector<int> reset();
    std::unique_ptr<Env> clone();
};
#endif
