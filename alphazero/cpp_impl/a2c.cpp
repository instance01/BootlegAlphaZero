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


namespace F = torch::nn::functional;


void A2CNetImpl::reset() {
  int n_features_before = n_input_features;

  this->fc_net = torch::nn::Sequential();

  int i = 0;
  for (int layer_features : net_architecture) {
    auto linear = register_module(
        std::to_string(i),
        torch::nn::Linear(n_features_before, layer_features)
    );
    auto relu = register_module(
        std::to_string(i + 1),
        torch::nn::ReLU()
    );
    fc_net->push_back(linear);
    fc_net->push_back(relu);
    n_features_before = layer_features;
    i += 2;
  }

  action_head = register_module(
      std::to_string(i + 1),
      torch::nn::Linear(n_features_before, n_actions)
  );
  value_head = register_module(
      std::to_string(i + 2),
      torch::nn::Linear(n_features_before, 1)
  );
}

std::pair<torch::Tensor, torch::Tensor>
A2CNetImpl::forward(torch::Tensor input) {
  auto x = input.view({input.size(0), -1});
  x = fc_net->forward(x);
  return std::make_pair(
      F::softmax(action_head(x), F::SoftmaxFuncOptions(-1)),
      value_head(x)
  );
}

A2CLearner::A2CLearner(
    Params params
) {
  this->params = params;
  this->policy_net = A2CNet(
      *std::get_if<int>(&params["n_input_features"]),
      *std::get_if<int>(&params["n_actions"]),
      *std::get_if<std::vector<int>>(&params["net_architecture"])
  );
  policy_net->to(torch::Device(c10::DeviceType::CPU));
  this->policy_optimizer = std::make_shared<torch::optim::Adam>(
      policy_net->parameters(),
      *std::get_if<double>(&params["alpha"])
  );

  // params["schedule_alpha"]
  // Add lr scheduler.
}

// TODO For now support 2D only.
std::pair<torch::Tensor, torch::Tensor>
A2CLearner::predict_policy_single(std::vector<double> sample) {
  // TODO
  // This could've been used to make a 2d array out of a single sample
  // x.view(-1, 3)  // where 3 is sample size
  return predict_policy({sample});
}

std::pair<torch::Tensor, torch::Tensor>
A2CLearner::predict_policy(std::vector<std::vector<double>> samples) {
  // TODO This requires at least one sample. So add an assert here that samples is not empty.
  auto samples_tensor = torch::from_blob(
      samples.data(),
      {static_cast<long long>(samples.size()), static_cast<long long>(samples[0].size())}
  );
  return policy_net(samples_tensor);
}

torch::Tensor
A2CLearner::_calc_normalized_rewards(std::vector<double> rewards) {
  // TODO Consider improving this function.
  double gamma = *std::get_if<double>(&params["gamma"]);
  std::vector<double> discounted_rewards;
  double R = 0;
  for (double reward : rewards) {
    R = reward + gamma * R;
    discounted_rewards.push_back(R);
  }
  std::reverse(discounted_rewards.begin(), discounted_rewards.end());

  auto discounted_rewards_tensor = torch::from_blob(
      discounted_rewards.data(),
      discounted_rewards.size(),
      torch::TensorOptions().dtype(torch::kFloat64)
  );

  discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean());
  discounted_rewards_tensor /= discounted_rewards_tensor.std();

  return discounted_rewards_tensor;
}

torch::Tensor
A2CLearner::update(Game game) {
  // Calculate and normalize discounted returns.
  auto normalized_returns = _calc_normalized_rewards(game.rewards);
  // Calculate losses of policy and value function.
  // Minimize cross entropy loss between softmax of net and mcts action
  // (one-hot encoded).
  torch::Tensor action_probs;
  torch::Tensor state_values;
  std::tie(action_probs, state_values) = predict_policy(game.states);
  torch::Tensor mcts_actions = torch::from_blob(
      game.mcts_actions.data(),
      game.mcts_actions.size()
  );

  torch::Tensor cross_entropy = (
      -(torch::log(action_probs) * mcts_actions).sum({1})
  ).sum({0});

  torch::Tensor value_losses = F::smooth_l1_loss(
      state_values.reshape(-1),
      normalized_returns,
      torch::nn::SmoothL1LossOptions(torch::kSum)
  );

  torch::Tensor loss = cross_entropy + value_losses;
  // Optimize joint loss.
  policy_optimizer->zero_grad();
  loss.backward();
  policy_optimizer->step();

  return loss;
}
