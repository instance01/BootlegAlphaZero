import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class A2CNet(nn.Module):
    def __init__(self, n_input_features, n_actions):
        super(A2CNet, self).__init__()
        n_hidden_units = 64
        self.fc_net = nn.Sequential(
            nn.Linear(n_input_features, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU()
        )
        self.action_head = nn.Linear(n_hidden_units, n_actions)
        self.value_head = nn.Linear(n_hidden_units, 1)

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
        self.transitions = []
        self.device = torch.device("cpu")
        self.policy_net = A2CNet(
            self.n_input_features, self.n_actions
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=params["alpha"]
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
        return self.policy_net(states)

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

    def update(self, state, action, reward, next_state, done, mcts_action):
        """Performs a learning update of the currently learned policy and
        value function.
        """
        self.transitions.append(
            (state, action, reward, next_state, done, mcts_action)
        )
        loss = None
        if done:
            states, actions, rewards, next_states, _, mcts_actions = tuple(
                zip(*self.transitions)
            )

            # Calculate and normalize discounted returns.
            normalized_returns = self._calc_normalized_rewards(rewards)

            # Calculate losses of policy and value function.
            # Minimize cross entropy loss between softmax of net and mcts action
            # (one-hot encoded)
            # -y * log(^y)
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            action_probs, state_values = self.predict_policy(states)
            policy_losses = []
            value_losses = []
            zipped = zip(
                action_probs,
                actions,
                state_values,
                normalized_returns,
                mcts_actions
            )
            for probs, action, value, R, mcts_action in zipped:
                cross_entropy = -sum([
                    torch.log(p) * a for p, a in zip(probs, mcts_action)
                ])
                policy_losses.append(cross_entropy)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            # Optimize joint loss.
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            # Delete all experiences afterwards.
            self.transitions.clear()
        return loss
