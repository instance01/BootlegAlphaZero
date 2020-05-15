#!/usr/local/bin/python3
import sys
import time
import copy

import gym

import mini_discrete_env  # noqa: F401
import alphazero


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
Running

Params16:
Running

"""


if __name__ == '__main__':
    env = gym.make('MiniDiscreteEnv-v0')
    env.goal_pos = [3, -3]  # [2,-2]
    env.borders = [7, -7]  # [5,-5]
    env.max_steps = 100  # 50
    desired_len = 9

    key = sys.argv[1]
    params = get_params(env)[key]

    start_time = time.time()
    lens = [alphazero.run(env, params, desired_len) for _ in range(10)]
    print(lens)
    print("Minutes:", (time.time() - start_time) / 60.)
