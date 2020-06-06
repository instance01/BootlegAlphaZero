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
#include "params.hpp"


struct A2CNetImpl : public torch::nn::Cloneable<A2CNetImpl> {
  public:
    torch::nn::Sequential fc_net{nullptr};
    torch::nn::Linear action_head{nullptr};
    torch::nn::Linear value_head{nullptr};

    int n_input_features;
    int n_actions;
    std::vector<int> net_architecture;

    A2CNetImpl() {};
    ~A2CNetImpl() {};

    A2CNetImpl(int n_input_features, int n_actions, std::vector<int> net_architecture)
      : n_input_features(n_input_features), n_actions(n_actions), net_architecture(net_architecture)
    { reset(); }

    void reset() override;

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
};

TORCH_MODULE(A2CNet);


class A2CLearner {
  public:
    Params params;
    A2CNet policy_net;
    std::shared_ptr<torch::optim::Adam> policy_optimizer;

    A2CLearner() {};
    ~A2CLearner() {};

    A2CLearner(
        Params params
    );

    std::pair<torch::Tensor, torch::Tensor> predict_policy_single(std::vector<double> sample);

    std::pair<torch::Tensor, torch::Tensor> predict_policy(std::vector<std::vector<double>> samples);

    torch::Tensor _calc_normalized_rewards(std::vector<double> rewards);

    torch::Tensor update(Game game);
};
#endif
