#!/usr/local/bin/python3
import time
import gym
import alphazero
import mini_discrete_env  # noqa: F401
from simulate import get_params


env = gym.make('MiniDiscreteEnv-v0')

env.goal_pos = [2, -2]
env.borders = [5, -5]
env.max_steps = 50
params = get_params()
params["env"] = env


if __name__ == '__main__':
    start_time = time.time()
    lens = [alphazero.run(env, params, 6) for _ in range(10)]
    print(lens)
    print("Minutes:", (time.time() - start_time) / 60.)
