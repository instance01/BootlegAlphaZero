#!/usr/local/bin/python3
import random
import time
import copy

import gym
import numpy as np
import matplotlib.pyplot as plt
import mini_discrete_env  # noqa: F401


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
        return '[%s|-c:%d-|-r:%f-%s-|-%d-%d:%d-]' % (
            self._id,
            len(self.children),
            self.reward,
            str(self.action),
            self.env.agent_pos[0],
            self.env.agent_pos[1],
            self.env.agent_dir
        )

    def __eq__(self, other):
        return self._id == other._id


class MCTS:
    def __init__(self, env, gamma=.99, c=.2, n_iter=1000, max_actions=100):
        env.reset()
        self.env = copy.copy(env)

        self.root_node = Node()
        self.root_node.env = copy.deepcopy(env)
        self.root_node.Q = 0
        self.root_node.visits = 0

        self.gamma = gamma
        self.c = c
        self.n_iter = n_iter
        self.max_actions = max_actions

        self.total_depth = 0

    def _ucb(self, parent_node, child_node):
        mean_q = child_node.Q / child_node.visits
        expl = 2 * np.log(parent_node.visits) / child_node.visits
        return mean_q + self.c * expl ** .5

    def _gen_children_nodes(self, parent_node):
        for action in parent_node.env.actions:
            env = copy.copy(parent_node.env)
            obs, reward, done, _ = env.step(action)
            node = Node()
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
        # TODO Support not only UCB, but also eps greedy and Boltzmann.
        children = []
        max_ucb = max(
            self._ucb(parent_node, child_node) for child_node in parent_node
        )
        for i, child_node in enumerate(parent_node):
            ucb_child = self._ucb(parent_node, child_node)
            if ucb_child >= max_ucb:
                children.append((i, child_node))
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
                _, curr_node = self._get_best_node(curr_node)
                path_len += 1
            else:
                _, node = self._expand(curr_node)
                if node is not None:
                    path_len += 1
                    return node, path_len
        return curr_node, path_len

    def simulate(self, curr_node, depth=1):
        if curr_node.is_terminal:
            return curr_node.reward
        env = copy.copy(curr_node.env)
        q_val = 0.
        i = depth
        while True:
            action = np.random.choice(env.actions)
            obs, reward, done, _ = env.step(action)

            q_val += self.gamma ** i * reward
            i += 1
            if done:
                break
        return q_val

    def backup(self, curr_node, q_val, total_path_len):
        while curr_node is not None:
            total_path_len -= 1
            discount = self.gamma ** total_path_len
            q_val += curr_node.reward * discount

            curr_node.Q = curr_node.visits * curr_node.Q + q_val
            curr_node.Q /= (curr_node.visits + 1)

            curr_node.visits += 1
            curr_node = curr_node.parent

    def policy(self, env=None, ret_node=False):
        if env:
            self.env = copy.deepcopy(env)
            self.root_node = Node()
            self.root_node.env = copy.deepcopy(env)
            self.root_node.Q = 0
            self.root_node.visits = 0

        for i in range(self.n_iter):
            node, path_len = self.select_expand()
            q_val = self.simulate(node, path_len + self.total_depth)
            self.backup(node, q_val, path_len + self.total_depth)
        self.total_depth += 1

        curr_node = self.root_node.children[
            np.argmax([n.Q for n in self.root_node])
        ]
        # action = curr_node.action
        # _, reward, done, _ = self.env.step(action)

        curr_node.parent = None
        self.root_node = curr_node
        if ret_node:
            return curr_node
        return curr_node.action

    def run(self):
        actions = []
        start_time = time.time()
        for j in range(self.max_actions):
            curr_node = self.policy(ret_node=True)
            actions.append(curr_node.action)

            if curr_node.is_terminal:
                break
        return actions, time.time() - start_time


if __name__ == '__main__':
    env = gym.make('MiniDiscreteEnv-v0')
    env.reset()
    lens = []
    for _ in range(100):
        mcts_obj = MCTS(env, gamma=.99, c=1.)
        path, timing = mcts_obj.run(200)
        print(path)
        lens.append(len(path))
    plt.hist(lens)
    plt.show()
