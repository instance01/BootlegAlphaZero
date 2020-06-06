import copy
from enum import Enum

import numpy as np
import gym
import gym_minigrid


class Action(Enum):
    left = 0
    right = 1
    forward = 2


class EnvWrapper:
    def __init__(self, env, params, actions):
        self.env = env
        self.actions = actions
        self.params = params

    def step(self, action):
        action = self.actions[action]
        obs, reward, done, _ = self.env.step(action)
        return (
            np.append(self.env.agent_pos, self.env.agent_dir),
            reward ** self.params["reward_exponent"],
            done,
            None
        )

    def reset(self):
        self.env.reset()
        return np.append(self.env.agent_pos, self.env.agent_dir)


def prepare_minigrid(game, params):
    env = gym.make('MiniGrid-Empty-%s-v0' % game)
    actions = {
        0: Action.left,
        1: Action.right,
        2: Action.forward
    }
    env.actions = Action
    return EnvWrapper(env, params, actions)


def init(game, params):
    return prepare_minigrid(game, params)


def step(env, action):
    obs, reward, done, _ = env.step(action)
    return list(obs), reward, done


def reset(env):
    return list(env.reset())


def clone(env):
    return copy.deepcopy(env)
