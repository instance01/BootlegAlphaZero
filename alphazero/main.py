#!/usr/local/bin/python3
import time
import gym
import alphazero
import mini_discrete_env  # noqa: F401


env = gym.make('MiniDiscreteEnv-v0')

env.goal_pos = [2, -2]
env.borders = [5, -5]
env.max_steps = 50

params = {
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
    "pb_c_base": 50,  # TODO I think this should be based on simulations.
    # base was 19652. But with just 100 simulations (and thus visits only
    # getting to 100 at max) visits don't matter.. At base=50, hell yes !

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

if __name__ == '__main__':
    start_time = time.time()
    lens = [alphazero.run(env, params, 6) for _ in range(10)]
    print(lens)
    print("Minutes:", (time.time() - start_time) / 60.)
