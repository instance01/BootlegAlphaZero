#include <iostream>
#include <torch/torch.h>

#include "game.hpp"
#include "replay_buffer.hpp"
#include "a2c.hpp"
#include "alphazero.hpp"


void test_a2c() {
  Params params;
  params["n_input_features"] = 3;
  params["n_actions"] = 3;
  params["net_architecture"] = std::vector<int>{64, 64};
  params["alpha"] = 0.01;
  auto a2c_agent = A2CLearner(params);
  torch::Tensor policy;
  torch::Tensor value;
  std::tie(policy, value) = a2c_agent.predict_policy({{1, 2, 3}});
  std::cout << policy.detach() << " " << value.detach() << std::endl;
}

void test_replay_buffer() {
  ReplayBuffer replay_buffer(100);
  replay_buffer.add({{0, 0}, {1, 1}}, {0, 95}, {{0, 1, 0}, {1, 0, 0}});
  replay_buffer.add({{2, 2}, {3, 3}}, {0, 91}, {{0, 1, 0}, {1, 0, 0}});
  replay_buffer.add({{4, 4}, {5, 5}}, {0, 0}, {{0, 1, 0}, {1, 0, 0}});
  Game game = replay_buffer.sample();
  game = replay_buffer.sample();
  game = replay_buffer.sample();
  game = replay_buffer.sample();
}

void test_env_evaluate() {
  Params params;
  params["n_input_features"] = 3;
  params["n_actions"] = 3;
  params["net_architecture"] = std::vector<int>{64, 64};
  params["alpha"] = 0.01;
  params["reward_exponent"] = 1;
  Env env = Env();
  env.init("5x5", params);

  std::vector<double> obs = env.reset();
  std::cout << obs[0] << " " << obs[1] << " " << obs[2] << std::endl;

  double reward;
  bool done;
  std::tie(obs, reward, done) = env.step(2);
  std::cout << obs[0] << " " << obs[1] << " " << obs[2] << std::endl;

  auto a2c_agent = A2CLearner(params);
  evaluate(env, params, a2c_agent);

  std::tie(obs, reward, done) = env.step(0);
  std::cout << obs[0] << " " << obs[1] << " " << obs[2] << std::endl;

  env.cleanup();
  py_finalize();
}

void test_all() {
  test_a2c();
  test_replay_buffer();
  test_env_evaluate();
}
