#include <iostream>
#include <torch/torch.h>
#include "alphazero.hpp"
#include "replay_buffer.hpp"
#include "test_all.hpp"  // TODO: Get rid of at some point


std::pair<int, double> evaluate(Env env, Params params, A2CLearner a2c_agent) {
  env = *env.clone();

  std::vector<double> state = env.reset();

  bool done = false;
  double total_reward = 0.;
  std::string actions = "";

  while (!done) {
    torch::Tensor action_probs;
    torch::Tensor val;
    std::tie(action_probs, val) = a2c_agent.predict_policy({state});

    int action = action_probs.argmax().item<int>();

    double reward;
    std::tie(state, reward, done) = env.step(action);
    total_reward += reward;
    actions += std::to_string(action);
  }

  env.cleanup();

  std::cout << "EVAL " << actions << " " << total_reward << std::endl;
  return std::make_pair(actions.length(), total_reward);
}

std::shared_ptr<Game> run_actor(Env env, Params params, MCTS mcts_agent, A2CLearner a2c_agent) {
  // TODO
  std::vector<std::vector<double>> states = {{}};
  std::vector<double> rewards = {};
  std::vector<std::vector<double>> mcts_actions = {{}};

  return std::make_shared<Game>(states, rewards, mcts_actions);
}

int main() {
  test_all();
}
