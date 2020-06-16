#include <cmath>
#include "env.hpp"
#include "cfg.hpp"
#include <iostream>

//
// TODO At some point this will need templating for all envs.
//

void
Env::init(std::string game, json params) {
  reward_exponent = params["reward_exponent"];

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
  grid_world = GridWorldEnv(width, width, blocks);
}

std::tuple<std::vector<int>, double, bool>
Env::step(int action) {
  std::tuple<std::vector<int>, double, bool> ret = grid_world.step(action);
  double& reward = std::get<1>(ret);
  if (reward != 0)
    reward = std::pow(reward, reward_exponent);
  return ret;
}

std::vector<int>
Env::reset() {
  return grid_world.reset();
}

std::unique_ptr<Env>
Env::clone() {
  auto env = std::make_unique<Env>();
  // TODO Could be better, eg move these things into a constructor.
  env->grid_world = grid_world;
  env->reward_exponent = reward_exponent;
  return env;
}
