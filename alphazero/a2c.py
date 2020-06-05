import copy
from pydoc import locate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class A2CNet(nn.Module):
    def __init__(self, n_input_features, n_actions, net_architecture):
        """Initialize a fully connected network based on a given architecture.

        Params:
            n_input_features: Size of input.
            n_actions: Number of actions.
            net_architecture: Layer sizes of a fully connected network, e.g.
                [64, 64, 32, 32].
        """
        super(A2CNet, self).__init__()
        net = []
        n_features_before = n_input_features
        for layer_features in net_architecture:
            net.append(nn.Linear(n_features_before, layer_features))
            net.append(nn.ReLU())
            n_features_before = layer_features

        self.fc_net = nn.Sequential(*net)
        self.action_head = nn.Linear(n_features_before, n_actions)
        self.value_head = nn.Linear(n_features_before, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)


class A2CLearner:
    """Autonomous agent using Synchronous Actor-Critic.
    """
    def __init__(self, params):
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = params["gamma"]
        self.n_actions = params["n_actions"]
        self.n_input_features = params["n_input_features"]
        self.device = torch.device("cpu")
        self.policy_net = A2CNet(
            self.n_input_features,
            self.n_actions,
            params["net_architecture"]
        ).to(self.device)
        # self.policy_net_copy = copy.deepcopy(self.policy_net)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=params["alpha"]
        )
        if params["schedule_alpha"]:
            scheduler = locate(
                'torch.optim.lr_scheduler.%s' % params["scheduler_class"]
            )
            self.scheduler = scheduler(
                self.policy_optimizer, params["scheduler_gamma"]
            )

    def policy(self, state):
        """Samples a new action using the policy network.
        """
        action_probs, _ = self.predict_policy([state])
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()

    def predict_policy(self, states):
        """Predicts the action probabilities.
        """
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        # TODO Unused for now.
        # return self.policy_net_copy(states)
        return self.policy_net(states)

    def set_nn_to_next_gen(self):
        self.policy_net_copy = copy.deepcopy(self.policy_net)

    def _calc_normalized_rewards(self, rewards):
        discounted_returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()
        discounted_returns = torch.tensor(
            discounted_returns, device=self.device, dtype=torch.float
        ).detach()
        normalized_returns = (discounted_returns - discounted_returns.mean())
        normalized_returns /= (discounted_returns.std() + self.eps)
        return normalized_returns

    def update(self, game):
        """Performs a learning update of the currently learnt policy and
        value function.
        """
        states, rewards, mcts_actions = zip(*game)

        # Calculate and normalize discounted returns.
        normalized_returns = self._calc_normalized_rewards(rewards)

        # Calculate losses of policy and value function.
        # Minimize cross entropy loss between softmax of net and mcts action
        # (one-hot encoded).
        action_probs, state_values = self.predict_policy(states)

        mcts_actions = torch.tensor(
            mcts_actions, device=self.device, dtype=torch.float
        )

        cross_entropy = (
            -(torch.log(action_probs) * mcts_actions).sum(axis=1)
        ).sum(axis=0)
        value_losses = F.smooth_l1_loss(
            state_values.reshape(-1),
            normalized_returns,
            reduction='sum'
        )
        loss = cross_entropy + value_losses

        # Optimize joint loss.
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss
