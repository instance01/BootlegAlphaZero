import random
import copy

import numpy as np


class Node:
    def __init__(self):
        self._id = str(random.random()) + str(random.random())
        self.is_fully_expanded = False
        self.is_terminal = False
        self.reward = 0
        self.parent = None
        self.children = []
        self.action = None
        self.Q = 0.

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return '[%s|-c:%d-|-r:%f-%s-|-%d-%d-]' % (
            self._id,
            len(self.children),
            self.reward,
            str(self.action),
            self.env.agent_pos[0],
            self.env.agent_pos[1]
            # self.env.was_right
        )

    def __eq__(self, other):
        return self._id == other._id


class MCTS:
    def __init__(
            self,
            env,
            a2c_agent,
            params):
        obs = env.reset()
        self.env = copy.copy(env)

        self.root_node = Node()
        self.root_node.env = copy.deepcopy(env)
        self.root_node.Q = 0
        self.root_node.visits = 0
        self.root_node.state = obs

        self.params = params
        self.gamma = params.get("gamma", .99)
        self.c = params.get("c", .2)
        self.n_iter = params["simulations"]
        self.max_actions = params["horizon"]
        self.dirichlet_alpha = params["dirichlet_alpha"]
        self.dirichlet_frac = params["dirichlet_frac"]
        self.pb_c_base = params["pb_c_base"]
        self.pb_c_init = params["pb_c_init"]
        self.a2c_agent = a2c_agent

        self.total_depth = 0
        self.policy_net_cache = {}

    def _ucb(self, parent_node, child_node, action_probs):
        # Polynomial UCT just like in the original AlphaZero.
        mean_q = child_node.Q / child_node.visits
        action = child_node.action

        base = self.pb_c_base
        pb_c = np.log((parent_node.visits + base + 1) / base) + self.pb_c_init
        pb_c *= np.sqrt(parent_node.visits) / (child_node.visits + 1)

        prior_score = pb_c * action_probs[action]
        return mean_q + prior_score

    def _gen_children_nodes(self, parent_node):
        for action in list(range(self.params['n_actions'])):
            env = copy.copy(parent_node.env)
            obs, reward, done, _ = env.step(action)
            node = Node()
            node.state = obs
            node.env = env
            node.action = action
            node.reward = reward
            node.is_terminal = done
            node.parent = parent_node
            node.Q = 0.
            node.visits = 0
            parent_node.children.append(node)

    def _expand(self, parent_node):
        """Pick the first action that was not visited yet.
        """
        if len(parent_node) == 0:
            self._gen_children_nodes(parent_node)

        for i, child_node in enumerate(parent_node):
            if child_node.visits == 0:
                return i, child_node

        parent_node.is_fully_expanded = True

        return 0, None

    def _get_best_node(self, parent_node):
        # Apparently prediction using torch the way we are currently doing it
        # is slow.
        # So to make things faster we cache policy predictions.
        key = tuple(parent_node.state)
        action_probs = self.policy_net_cache.get(key, None)
        if action_probs is None:
            action_probs, _ = self.a2c_agent.predict_policy(
                [parent_node.state]
            )
            self.policy_net_cache[key] = action_probs
        action_probs = action_probs[0]

        # UCB1
        ucb_vals = [
            self._ucb(parent_node, child_node, action_probs)
            for child_node in parent_node
        ]
        max_ucb = max(ucb_vals)
        children = []
        for child_node, ucb_child in zip(parent_node, ucb_vals):
            if ucb_child >= max_ucb:
                children.append(child_node)

        # Here an error might happen: children may be empty.
        # Long story short: This happens when we have gradient explosion.
        # We then have nan's all over the place which fuck things up.

        return children[np.random.choice(len(children))]

    def select_expand(self):
        """Select best node until finding a node that is not fully expanded.
        Expand it and return the expanded node (together with length of path
        for gamma).
        """
        path_len = 0

        curr_node = self.root_node
        while True:
            if curr_node.is_terminal:
                break
            if curr_node.is_fully_expanded:
                curr_node = self._get_best_node(curr_node)
                path_len += 1
            else:
                _, node = self._expand(curr_node)
                if node is not None:
                    path_len += 1
                    return node, path_len
        return curr_node, path_len

    def backup(self, curr_node, q_val, total_path_len):
        while curr_node is not None:
            # AlphaZero does not have discounting.
            # https://www.biostat.wisc.edu/~craven/cs760/lectures/AlphaZero.pdf
            curr_node.Q += q_val
            curr_node.visits += 1
            curr_node = curr_node.parent

    def reset_policy_cache(self):
        self.policy_net_cache = {}

    def policy(self, env=None, obs=None, ret_node=False):
        if env:
            self.env = copy.deepcopy(env)
            self.root_node = Node()
            self.root_node.env = copy.deepcopy(env)
            self.root_node.Q = 0
            self.root_node.visits = 0
            self.root_node.state = obs

            # Add Dirichlet noise to root
            key = tuple(self.root_node.state)
            action_probs, _ = self.a2c_agent.predict_policy(
                [self.root_node.state]
            )
            action_probs = action_probs[0]
            alpha = self.dirichlet_alpha
            frac = self.dirichlet_frac
            noise = np.random.gamma(alpha, 1, self.params['n_actions'])
            actions = list(range(self.params['n_actions']))
            for a, n in zip(actions, noise):
                action_probs[a] = action_probs[a] * (1 - frac) + n * frac
            self.policy_net_cache[key] = [action_probs]

        for i in range(self.n_iter):
            node, path_len = self.select_expand()
            _, state_values = self.a2c_agent.predict_policy(
                [node.state]
            )
            q_val = state_values[0][0].item()
            # If gradients explode, q_val will be nan.
            self.backup(node, q_val, path_len + self.total_depth)
        self.total_depth += 1

        return [n.visits / self.n_iter for n in self.root_node]
