#ifndef A2C_HEADER
#define A2C_HEADER
#include <string>
#include <any>
#include <iostream>

#include <torch/torch.h>
#include <torch/types.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/functional.h>
#include <torch/optim/adam.h>
#include <c10/core/DeviceType.h>

#include "game.hpp"
#include "cfg.hpp"
#include "env.hpp"


struct A2CNetImpl : public torch::nn::Cloneable<A2CNetImpl> {
  public:
    torch::nn::Sequential seq{nullptr};
    torch::nn::Linear action_head{nullptr};
    torch::nn::Linear value_head{nullptr};

    int n_input_features;
    int n_actions;
    std::vector<int> net_architecture;

    A2CNetImpl() {};
    ~A2CNetImpl() {};
    A2CNetImpl(int n_input_features, int n_actions, std::vector<int> net_architecture);

    void reset() override;
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
};

TORCH_MODULE(A2CNet);

class A2CLearner {
  public:
    torch::Tensor expected_mean_tensor;
    torch::Tensor expected_stddev_tensor;

    json params;
    A2CNet policy_net;
    std::shared_ptr<torch::optim::Adam> policy_optimizer;

    A2CLearner() {};
    A2CLearner(json params, Env &env);
    ~A2CLearner() {};

    torch::Tensor normalize(torch::Tensor x);
    std::pair<torch::Tensor, torch::Tensor> predict_policy(torch::Tensor samples_);
    std::pair<torch::Tensor, torch::Tensor> predict_policy(std::vector<std::vector<int>> states);
    torch::Tensor _calc_normalized_rewards(std::vector<double> rewards);
    torch::Tensor update(std::shared_ptr<Game> game);
};
#endif
