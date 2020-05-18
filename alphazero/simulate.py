#!/usr/local/bin/python3
import sys
import time
import copy
from enum import Enum

import gym

import mini_discrete_env  # noqa: F401
import gym_minigrid  # noqa: F401
import alphazero
from torch.utils.tensorboard import SummaryWriter


def get_params(env):
    params1 = {
        "n_actions": 2,
        "n_input_features": 2,
        "env": env,

        # MCTS
        "gamma": .99,
        "c": 1.,  # .001  # TODO Using puct now
        "simulations": 50,  # 1000
        "horizon": 100,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .25,
        "pb_c_base": 50,

        # A2C
        "alpha": .01,  # .01 (AlphaZero best); .001 (Imitation best)

        # AlphaZero
        "memory_capacity": 1000,
        "prioritized_sampling": True,
        "episodes": 100,
        "n_actors": 20,  # 5000
        "train_steps": 2000,  # 700000

        # TODO unused right now
        "epsilon": .1,
        "epsilon_linear_decay": 1. / 10000,  # 10000 is memory_capacity
        "epsilon_min": 0.01
    }

    params2 = copy.deepcopy(params1)
    params2.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 50,
        "train_steps": 7000,
    })

    params3 = copy.deepcopy(params1)
    params3.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 50,
        "train_steps": 7000,
    })

    params4 = copy.deepcopy(params1)
    params4.update({
        "alpha": .1,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 50,
        "train_steps": 7000,
    })

    params5 = copy.deepcopy(params1)
    params5.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 20,
        "train_steps": 7000,
    })

    params6 = copy.deepcopy(params1)
    params6.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 50,
        "train_steps": 7000,
    })

# NEW
    params7 = copy.deepcopy(params1)
    params7.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 50,
        "train_steps": 7000,
    })

    params8 = copy.deepcopy(params1)
    params8.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 7000,
    })

    params9 = copy.deepcopy(params1)
    params9.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
    })

    params10 = copy.deepcopy(params1)
    params10.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
    })

    params11 = copy.deepcopy(params1)
    params11.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
    })

    params12 = copy.deepcopy(params1)
    params12.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
    })

    params13 = copy.deepcopy(params1)
    params13.update({
        "alpha": .0001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
    })

    params14 = copy.deepcopy(params1)
    params14.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
    })

    params15 = copy.deepcopy(params1)
    params15.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "dirichlet_alpha": .03,
        "dirichlet_frac": .25
    })

    params16 = copy.deepcopy(params1)
    params16.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "dirichlet_alpha": .6,
        "dirichlet_frac": .25
    })

    params17 = copy.deepcopy(params1)
    params17.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "dirichlet_alpha": .9,
        "dirichlet_frac": .25
    })

    params18 = copy.deepcopy(params1)
    params18.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "dirichlet_alpha": .4,
        "dirichlet_frac": .25
    })

    params19 = copy.deepcopy(params1)
    params19.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params20 = copy.deepcopy(params1)
    params20.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "pb_c_base": 10
    })

    params21 = copy.deepcopy(params1)
    params21.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "pb_c_base": 100
    })

    params22 = copy.deepcopy(params1)
    params22.update({
        "alpha": .0001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 1000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params23 = copy.deepcopy(params1)
    params23.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 1000,
        "dirichlet_alpha": .2,
        "dirichlet_frac": .4
    })

    params24 = copy.deepcopy(params1)
    params24.update({
        "alpha": .00005,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 2000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params25 = copy.deepcopy(params1)
    params25.update({
        "alpha": .00005,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 1000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params26 = copy.deepcopy(params1)
    params26.update({
        "alpha": .00005,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 3000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params27 = copy.deepcopy(params1)
    params27.update({
        "alpha": .0002,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 3000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params28 = copy.deepcopy(params1)
    params28.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 3000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    return {
        "1": params1,
        "2": params2,
        "3": params3,
        "4": params4,
        "5": params5,
        "6": params6,
        "7": params7,
        "8": params8,
        "9": params9,
        "10": params10,
        "11": params11,
        "12": params12,
        "13": params13,
        "14": params14,
        "15": params15,
        "16": params16,
        "17": params17,
        "18": params18,
        "19": params19,
        "20": params20,
        "21": params21,
        "22": params22,
        "23": params23,
        "24": params24,
        "25": params25,
        "26": params26,
        "27": params27,
        "28": params28,
    }


"""
Gradient explosion:
(Pdb) print(self.a2c_agent.policy_net.fc_net[2].weight)
Parameter containing:
tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], requires_grad=True)
--

Params1:
    [2, 4, 0, 8, 3, 99, 1, 99, 7, 1]
    Minutes: 700.5658185561498

Params2:
    Crashed, gradient explosion
    Running (restarted)

Params3:
    [7, 8, 7, 8, 3, 3, 6, 1, 1, 0]
    Minutes: 602.0077656269074

Params4:
    Crashed, gradient explosion
    Running (restarted)
    +multiprocessing
    Crashed again, gradient explosion

Params5:
    Crashed, gradient explosion

Params6:
    [4, 99, 12, 4, 99, 30, 3, 2, 7, 99]
    Minutes: 2090.946610148748

Params7:
    [4, 13, 8, 1, 1, 4, 0, 33, 4, 19]
    Minutes: 562.2411169131597

Params8:
    Crashed, gradient explosion

Params9:
    [8, 6, 13, 4, 1, 2, 99, 2, 3, 1]
    Minutes: 250.394251592954

Params10:
    [13, 1, 10, 20, 3, 4, 1, 6, 3, 1]
    Minutes: 86.78793809811275
    +multiprocessing

Params11: (same as Params9)
    +multiprocessing
    +pb_c base is now 50.
    Crashed, pool leaking fds
    +fixed pool
    [4, 11, 29, 59, 10, 2, 6, 12, 24, 8]
    Minutes: 141.41895825068156

Params12:
    [37, 4, 11, 4, 13, 21, 6, 3, 6, 3]
    Minutes: 126.497244437535

Params13:
    [16, 99, 1, 11, 87, 1, 16, 5, 3, 1]
    Minutes: 278.8917894800504

Params14:
    [99, 61, 2, 14, 9, 2, 9, 11, 3, 10]
    Minutes: 248.89004709323248

Params12:  # Since it's the best one so far. (12_1)
    +goal_pos=[3,-3]
    +borders=[7,-7]
    +max_steps=100
    [34, 35, 64, 10, 3, 13, 99, 17, 7, 14]
    Minutes: 255.10599056879678

Params15:
    [63, 11, 11, 6, 38, 2, 99, 31, 99, 99]
    Minutes: 624.9840517878532

Params16:
    [2, 33, 11, 1, 22, 29, 1, 40, 99, 8]
    Minutes: 376.4243238766988

Params17:
    [4, 8, 7, 46, 3, 51, 62, 33, 1, 62]
    Minutes: 398.19128265380857

Params18:
    [7, 35, 13, 9, 5, 99, 99, 54, 99, 7]
    Minutes: 442.9775447527567

Params19:
    [18, 6, 29, 9, 99, 8, 5, 11, 18, 52]
    Minutes: 267.74789813756945

Params20:
    [32, 34, 99, 67, 1, 32, 99, 99, 2, 14]
    Minutes: 463.37765492598214

Params21:
    [3, 26, 99, 56, 2, 1, 4, 20, 18, 91]
    Minutes: 321.5921831091245

Params12_again:
    +multiprocessing numpy seed fix
    [37, 99, 9, 10, 13, 6, 11, 21, 7, 17]
    Minutes: 145.70517488718033

Params19_again:
    +multiprocessing numpy seed fix
    [5, 17, 10, 2, 2, 3, 1, 9, 10, 2]
    Minutes: 57.76622135241826

Params22:
    just testing Gridworld

Params23:
    just testing Gridworld

Params24:
    EVAL 1202212 0.7708295649609999
    Minutes: 76.87422110637029
    +dont quadruple rewards
    +do 10 runs (from now on charts)
    Running

Params25:
    EVAL 1202212 0.7708295649609999
    Minutes: 59.53636395931244
    Running

Params26:
    EVAL 1202212 0.7708295649609999
    Minutes: 120.59210259517035
    Running

Params27:
    EVAL 1202212 0.7708295649609999
    Minutes: 134.39488005638123
    Running

Params28:
    EVAL 122022 0.8008746470559999
    Minutes: 65.5352343996366
    Running

"""


def simulate_many_minidiscrete():
    env = gym.make('MiniDiscreteEnv-v0')
    env.goal_pos = [3, -3]  # [2,-2]
    env.borders = [7, -7]  # [5,-5]
    env.max_steps = 100  # 50
    desired_len = 9

    key = sys.argv[1]
    params = get_params(env)[key]

    start_time = time.time()
    episodes = [alphazero.run(env, params, desired_len, i) for i in range(10)]
    print(episodes)
    print("Minutes:", (time.time() - start_time) / 60.)


# TODO This should not be global, but it's needed for multiprocessing.
class Action(Enum):
    left = 0
    right = 1
    forward = 2


def simulate_many_minigrid():
    start_time = time.time()
    env = gym.make('MiniGrid-Empty-5x5-v0')
    desired_len = 8

    # Monkey patch actions, step and reset.
    # TODO Seems a bit hacky.
    actions = {
        0: Action.left,
        1: Action.right,
        2: Action.forward
    }
    env.actions = Action

    def step(cls, action):
        action = actions[action]
        obs, reward, done, _ = cls._step(action)
        # TODO An idea was to quadruple reward to make good rewards more
        # important.
        return obs['image'].flatten(), reward, done, None
    env._step = env.step
    env.__class__._step = env.__class__.step
    env.step = step.__get__(env, env.__class__)
    env.__class__.step = step.__get__(env, env.__class__)

    def reset(cls):
        obs = cls._reset()
        return obs['image'].flatten()
    env._reset = env.reset
    env.__class__._reset = env.__class__.reset
    env.reset = reset.__get__(env, env.__class__)
    env.__class__.reset = reset.__get__(env, env.__class__)

    # Load params and run AlphaZero.
    key = sys.argv[1]
    params = get_params(env)[key]
    params["key"] = key
    params["n_actions"] = 3
    params["n_input_features"] = 147

    writer = SummaryWriter()
    [alphazero.run(env, params, desired_len, i, writer) for i in range(10)]
    print("Minutes:", (time.time() - start_time) / 60.)


# TODO Refactor simulate* functions.


if __name__ == '__main__':
    simulate_many_minigrid()
    # simulate_many_minidiscrete()
