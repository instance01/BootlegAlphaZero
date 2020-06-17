#include <string>
#include <map>
#include <any>
#include <variant>
#include <iostream>

#include <torch/torch.h>
#include <torch/types.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/functional.h>
#include <torch/optim/adam.h>
#include <c10/core/DeviceType.h>

#include "a2c.hpp"
#include "util.hpp"


namespace F = torch::nn::functional;


A2CNetImpl::A2CNetImpl(
    int n_input_features, int n_actions, std::vector<int> net_architecture
) : n_input_features(n_input_features), n_actions(n_actions), net_architecture(net_architecture) {
  seq = register_module("seq", torch::nn::Sequential());

  int n_features_before = n_input_features;

  int i = 0;
  for (int layer_features : net_architecture) {
    auto linear = register_module(
        "l" + std::to_string(i),
        torch::nn::Linear(n_features_before, layer_features)
    );
    linear->reset_parameters();
    auto relu = register_module("r" + std::to_string(i + 1), torch::nn::ReLU());
    seq->push_back(linear);
    seq->push_back(relu);
    n_features_before = layer_features;
    i += 2;
  }

  action_head = register_module("a", torch::nn::Linear(n_features_before, 3));
  value_head = register_module("v", torch::nn::Linear(n_features_before, 1));

  action_head->reset_parameters();
  value_head->reset_parameters();
}

void A2CNetImpl::reset() {
  action_head->reset_parameters();
  value_head->reset_parameters();
}

std::pair<torch::Tensor, torch::Tensor>
A2CNetImpl::forward(torch::Tensor input) {
  auto x = input.view({input.size(0), -1});
  x = seq->forward(x);
  auto policy = F::softmax(action_head(x), F::SoftmaxFuncOptions(-1));
  auto value = value_head(x);
  return std::make_pair(policy, value);
}

A2CLearner::A2CLearner(json params, Env &env) : params(params) {
  policy_net = A2CNet(
    params["n_input_features"],
    params["n_actions"],
    params["net_architecture"]
  );

  double lr = params["alpha"];
  auto opt = torch::optim::AdamOptions(lr);
  policy_optimizer = std::make_shared<torch::optim::Adam>(
      policy_net->parameters(),
      opt
  );

  expected_mean_tensor = torch::from_blob(
      env.grid_world.expected_mean.data(),
      {env.grid_world.expected_mean.size()},
      torch::TensorOptions().dtype(torch::kFloat32)
  );
  expected_stddev_tensor = torch::from_blob(
      env.grid_world.expected_stddev.data(),
      {env.grid_world.expected_stddev.size()},
      torch::TensorOptions().dtype(torch::kFloat32)
  );
}

torch::Tensor
A2CLearner::normalize(torch::Tensor x) {
  return (x - expected_mean_tensor) / expected_stddev_tensor;
}

torch::Tensor
A2CLearner::_calc_normalized_rewards(std::vector<double> rewards) {
  // TODO Consider improving this function.
  double gamma = params["gamma"];
  std::vector<double> discounted_rewards;
  double R = 0;
  std::reverse(rewards.begin(), rewards.end());
  for (double reward : rewards) {
    R = reward + gamma * R;
    discounted_rewards.push_back(R);
  }
  std::reverse(discounted_rewards.begin(), discounted_rewards.end());

  std::vector<float> flt(discounted_rewards.begin(), discounted_rewards.end());

  auto discounted_rewards_tensor = torch::from_blob(
      flt.data(),
      flt.size(),
      torch::TensorOptions().dtype(torch::kFloat32)
  );

  discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean());
  auto std = discounted_rewards_tensor.std();
  if (std.item<float>() != 0) {
    discounted_rewards_tensor /= std;
  }

  return discounted_rewards_tensor;
}

std::pair<torch::Tensor, torch::Tensor>
A2CLearner::predict_policy(torch::Tensor samples_) {
  torch::Tensor samples = normalize(samples_.to(torch::kFloat32));
  return policy_net->forward(samples);
}

std::pair<torch::Tensor, torch::Tensor>
A2CLearner::predict_policy(std::vector<std::vector<int>> states) {
  auto flattened_states = flatten_as_float(states);
  auto samples_tensor = vec_2d_as_tensor(
      flattened_states, torch::kFloat32, states.size(), states[0].size()
  );

  torch::Tensor samples = normalize(samples_tensor);
  return policy_net->forward(samples);
}


torch::Tensor
A2CLearner::update(std::shared_ptr<Game> game) {
  policy_net->train();

  // Prepare data.
  auto flattened_states = flatten_as_float(game->states);
  auto samples_tensor = vec_2d_as_tensor(
      flattened_states, torch::kFloat32, game->states.size(), game->states[0].size()
  );
  torch::Tensor samples = normalize(samples_tensor);

  auto normalized_returns = _calc_normalized_rewards(game->rewards);

  auto flattened_mcts = flatten_as_float(game->mcts_actions);
  auto mcts_actions = vec_2d_as_tensor(
      flattened_mcts, torch::kFloat32, game->mcts_actions.size(), game->mcts_actions[0].size()
  );

  // Forward.
  torch::Tensor action_probs;
  torch::Tensor values;
  std::tie(action_probs, values) = policy_net->forward(samples);

  // Calcluate losses
  auto ff = -(torch::log(action_probs) * mcts_actions).sum({1});
  torch::Tensor cross_entropy = (ff).sum({0});

  torch::Tensor value_loss = F::smooth_l1_loss(
      values.reshape(-1),
      normalized_returns,
      torch::nn::SmoothL1LossOptions(torch::kSum)
  );

  torch::Tensor loss = cross_entropy + value_loss;

  policy_optimizer->zero_grad();
  loss.backward();
  policy_optimizer->step();

  // TODO Remove. Hacky way of updating weights.
  //for (auto layer : policy_net->layers) {
  //  layer->weight = layer->weight.sub(.01 * layer->named_parameters()["weight"].grad());
  //  layer->bias = layer->bias.sub(.01 * layer->named_parameters()["bias"].grad());
  //}
  //policy_net->action_head->weight = policy_net->action_head->weight.sub(.01 * policy_net->action_head->named_parameters()["weight"].grad());
  //policy_net->action_head->bias = policy_net->action_head->bias.sub(.01 * policy_net->action_head->named_parameters()["bias"].grad());

  policy_net->eval();
  return loss;
}
