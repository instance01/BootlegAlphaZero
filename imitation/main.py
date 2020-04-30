#!/usr/local/bin/python3
import sys
sys.path.append("..")
import copy
import a2c
import matplotlib.pyplot as plot
import numpy
import gym
from mcts.mcts import MCTS


def episode(env, mcts_agent, a2c_agent, n_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    actions = []
    while not done:
        action2_ = mcts_agent.policy(env)
        if action2_ == -1:
            action2_ = 0
        action2 = [0., 0.]
        action2[action2_] = 1.
        action = a2c_agent.policy(state)

        orig_env = copy.deepcopy(env)
        next_state2, reward2, done2, _ = orig_env.step(action2)
        next_state, reward, done, _ = env.step(action)

        loss = a2c_agent.update(state, action, reward, next_state, done, action2)

        state = next_state
        discounted_return += reward * discount_factor ** time_step
        time_step += 1

        actions.append(action)

    if loss:
        sys.stdout.write(
            str(n_episode) +
            ' ' +
            str(loss.item()).ljust(22) +
            ' |' +
            str(discounted_return) +
            '\r'
        )
        sys.stdout.flush()
    return actions


params = {}
import mini_discrete_env  # noqa: F401
env = gym.make('MiniDiscreteEnv-v0')
env.reset()
params["n_actions"] = 2
params["n_input_features"] = numpy.prod(env.observation_space.shape)
params["env"] = env
params["gamma"] = 0.99

# Planning/MCTS Hyperparameters
params["horizon"] = 10
params["simulations"] = 100  # 1000

# Deep RL Hyperparameters
params["alpha"] = 0.0005  # 0.001
params["epsilon"] = 0.1
params["memory_capacity"] = 10000
params["warmup_phase"] = 1000
params["target_update_interval"] = 5000
params["minibatch_size"] = 64
params["epsilon_linear_decay"] = 1.0/params["memory_capacity"]
params["epsilon_min"] = 0.01
training_episodes = 1  # 2000


mcts_agent = MCTS(params["env"], params["gamma"], c=1., n_iter=params["simulations"])
a2c_agent = a2c.A2CLearner(params)
lens = [len(episode(env, mcts_agent, a2c_agent, i)) for i in range(500)]
actions = episode(env, mcts_agent, a2c_agent, 500)
print('-')
print(actions)
plot.plot(lens)
plot.show()
