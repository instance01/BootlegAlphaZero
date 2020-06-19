#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <algorithm>
#include "mountain_car.hpp"


MtCarEnv::MtCarEnv(MtCarEnv &other) {
  max_steps = other.max_steps;
  goal_velocity = other.goal_velocity;
  steps = other.steps;
  state = other.state;
  generator = other.generator;
}

MtCarEnv::MtCarEnv(float goal_velocity) : goal_velocity(goal_velocity) {
  std::random_device rd;
  generator = std::mt19937(rd());

  reset();

  // State is already normalized well enough..
  expected_mean = {0., 0.};
  expected_stddev = {1., 1.};
}

std::vector<float>
MtCarEnv::reset() {
  steps = 0;
  std::uniform_real_distribution<float> distribution(-.6, -.4);
  state = {
    distribution(generator),
    0.
  };
  return state;
}

std::tuple<std::vector<float>, double, bool>
MtCarEnv::step(int action) {
  steps += 1;
  if (steps >= max_steps) {
    return std::make_tuple(state, 0.0, true);
  }

  float position = state[0];
  float velocity = state[1];

  velocity += (action - 1) * force + std::cos(3 * position) * (-gravity);
  velocity = std::clamp(velocity, -max_speed, max_speed);
  position += velocity;
  position = std::clamp(position, min_position, max_position);
  if (position == min_position && velocity < 0)
    velocity = 0;
  bool done = position >= goal_position && velocity >= goal_velocity;
  double reward = -1.0;

  // TODO Is there a better way ?
  state[0] = position;
  state[1] = velocity;

  return std::make_tuple(state, reward, done);
}
