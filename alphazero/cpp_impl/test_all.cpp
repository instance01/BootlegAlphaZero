#include <iostream>
#include <torch/torch.h>
#include "tensorboard_logger.h"

#include "game.hpp"
#include "replay_buffer.hpp"
#include "a2c.hpp"
#include "alphazero.hpp"
#include "simple_thread_pool.hpp"
#include "mcts.hpp"


std::tuple<Params, Env, MCTS, A2CLearner> setup_all() {
  Params params;
  params["n_input_features"] = 3;
  params["n_actions"] = 3;
  params["net_architecture"] = std::vector<int>{64, 64};
  params["alpha"] = 0.01;
  params["reward_exponent"] = 1;
  params["dirichlet_alpha"] = 0.1;
  params["dirichlet_frac"] = 0.25;
  params["pb_c_base"] = 500.0;
  params["pb_c_init"] = 0.1;
  params["simulations"] = 10;
  Env env = Env();
  env.init("5x5", params);
  auto a2c_agent = A2CLearner(params);
  auto mcts_agent = MCTS(env, a2c_agent, params);
  return std::make_tuple(params, env, mcts_agent, a2c_agent);
}


void test_a2c() {
  Params params;
  Env env;
  A2CLearner a2c_agent;
  MCTS mcts_agent;
  std::tie(params, env, mcts_agent, a2c_agent) = setup_all();

  torch::Tensor policy;
  torch::Tensor value;
  std::tie(policy, value) = a2c_agent.predict_policy({{1, 2, 3}});
  std::cout << policy.detach() << " " << value.detach() << std::endl;

  env.cleanup();
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
  Env env;
  A2CLearner a2c_agent;
  MCTS mcts_agent;
  std::tie(params, env, mcts_agent, a2c_agent) = setup_all();

  std::vector<double> obs = env.reset();
  std::cout << obs[0] << " " << obs[1] << " " << obs[2] << std::endl;

  double reward;
  bool done;
  std::tie(obs, reward, done) = env.step(2);
  std::cout << obs[0] << " " << obs[1] << " " << obs[2] << std::endl;

  evaluate(env, params, a2c_agent);

  std::tie(obs, reward, done) = env.step(0);
  std::cout << obs[0] << " " << obs[1] << " " << obs[2] << std::endl;

  // ATTENTION!
  // Env gets cloned in evaluate() and at the end is cleaned up.
  // However, the clone is a *shallow* one ! The PyObject references are simply copied.
  // So, if one env gets cleaned up, all derived (=cloned) ones get too.
  // TODO: That sucks, fix this.
  //env.cleanup();
}

void test_tensorboard_logger() {
  // TODO
  std::string log_file = "";
  TensorBoardLogger logger(log_file.c_str());
}

void test_thread_pool() {
  Params params;
  Env env;
  A2CLearner a2c_agent;
  MCTS mcts_agent;
  std::tie(params, env, mcts_agent, a2c_agent) = setup_all();

  auto pool = SimpleThreadPool(2);
  auto lambda = [env, params, mcts_agent, a2c_agent]() -> std::shared_ptr<Game> {
    return run_actor(env, params, mcts_agent, a2c_agent);
  };
  Task *task = new Task(lambda);
  Task *task2 = new Task(lambda);
  Task *task3 = new Task(lambda);
  pool.add_task(task);
  pool.add_task(task2);
  pool.add_task(task3);
  std::vector<std::shared_ptr<Game>> games = pool.join();
  std::cout << "# Games: " << games.size() << std::endl;

  env.cleanup();
}

void test_mcts() {
  Params params;
  Env env;
  A2CLearner a2c_agent;
  MCTS mcts_agent;
  std::tie(params, env, mcts_agent, a2c_agent) = setup_all();

  std::vector<double> obs = env.reset();

  std::vector<double> probs = mcts_agent.policy(env, obs);
  std::cout << probs[0] << " " << probs[1] << " " << probs[2] << std::endl;
}

void test_all() {
  test_a2c();
  test_replay_buffer();
  test_env_evaluate();
  test_env_evaluate();
  // test_tensorboard_logger();
  test_thread_pool();
  test_mcts();

  py_finalize();
}
