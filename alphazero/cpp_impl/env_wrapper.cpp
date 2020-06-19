#include <cmath>
#include <iostream>
#include "cfg.hpp"
#include "env_wrapper.hpp"
#include "envs/mountain_car.hpp"
#include "envs/gridworld.hpp"


// TODO This should be templates at some point.


void
EnvWrapper::init(std::string game, json params) {
  this->game = game;
  this->params = params;
  this->reward_exponent = params["reward_exponent"];

  if (game == "mtcar") {
    env = std::make_shared<MtCarEnv>(0.);
  } else {
    std::set<std::pair<int, int>> blocks;
    int width = 3;
    if (game == "5x5")
      width = 5;
    if (game == "8x8")
      width = 8;
    if (game == "16x16")
      width = 16;
    for (int i = 0; i < width; ++i) {
      blocks.insert({0, i});
      blocks.insert({width - 1, i});
      blocks.insert({i, 0});
      blocks.insert({i, width - 1});
    }
    env = std::make_shared<GridWorldEnv>(width, width, blocks);
  }
}

std::tuple<std::vector<float>, double, bool>
EnvWrapper::step(int action) {
  std::tuple<std::vector<float>, double, bool> ret = env->step(action);
  double& reward = std::get<1>(ret);
  if (reward != 0 && params["prioritized_sampling"])
    reward = std::pow(reward, reward_exponent);
  return ret;
}

std::vector<float>
EnvWrapper::reset() {
  return env->reset();
}

std::unique_ptr<EnvWrapper>
EnvWrapper::clone() {
  auto env_ = std::make_unique<EnvWrapper>();
  // TODO Could be better, eg move these things into a constructor.
  // TODO is this init needed?
  env_->init(game, params);

  if (game == "mtcar") {
    auto mtcar_env = std::static_pointer_cast<MtCarEnv>(env);
    env_->env = std::make_shared<MtCarEnv>(*mtcar_env);
  } else {
    auto grid_env = std::static_pointer_cast<GridWorldEnv>(env);
    env_->env = std::make_shared<GridWorldEnv>(*grid_env);
  }

  env_->reward_exponent = reward_exponent;
  return env_;
}
