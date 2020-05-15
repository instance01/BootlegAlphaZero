import time
import sys
import copy
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

import a2c
from mcts import MCTS


class ReplayBuffer:
    def __init__(self, window_size, prioritized_sampling=True):
        self.buffer = []
        self.window_size = window_size
        self.prioritized_sampling = prioritized_sampling

    def add(self, sample):
        """Add a game to the replay buffer.
        Consists of a list of tuples in the form of:
            (state, action, reward, next_state, done, mcts_action)
        """
        self.buffer.append(sample)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def get_rewards(self):
        # Sum up rewards.
        # See add() for more information on the format of the samples.
        p = np.array([
            sum(sample[2] for sample in samples)
            for samples in self.buffer
        ])
        return p

    def sample(self):
        if not self.prioritized_sampling:
            # Uniform sampling
            return self.buffer[np.random.choice(len(self.buffer))]

        p = self.get_rewards()
        # TODO EXPLAIN -> So negative rewards also get a small chance
        p *= p
        # p += abs(p.min() * 2)
        if p.sum() == 0:
            return self.buffer[np.random.choice(len(self.buffer))]
        p /= p.sum()
        return self.buffer[np.random.choice(len(self.buffer), p=p)]


def evaluate(env, a2c_agent):
    env = copy.deepcopy(env)
    state = env.reset()

    done = False
    total_reward = 0
    actions = ""
    while not done:
        action_probs, _ = a2c_agent.predict_policy([state])
        action = np.argmax(action_probs.tolist()[0])

        state, reward, done, _ = env.step(action)
        total_reward += reward
        actions += str(action)
    print("EVAL", actions, total_reward)
    return len(actions)


def one_hot_encode(action, actions_len):
    action_one_hot = [0. for _ in range(actions_len)]
    action_one_hot[action] = 1.
    return action_one_hot


def run_actor(env, params, mcts_agent, a2c_agent):
    # TODO REMOVE
    mcts_actions = ""

    state = env.reset()
    done = False
    game = []
    while not done:
        mcts_action = mcts_agent.policy(env, state)
        # TODO This samples. Recheck whether we should sample here or
        # exploit, ie argmax.
        action = a2c_agent.policy(state)

        # TODO Apparently AlphaZero simply goes by MCTS in self play..
        # hmm..
        #next_state, reward, done, _ = env.step(action)
        next_state, reward, done, _ = env.step(mcts_action)

        mcts_actions += str(mcts_action)

        mcts_action = one_hot_encode(mcts_action, params["n_actions"])
        sample = (state, action, reward, next_state, done, mcts_action)

        state = next_state
        game.append(sample)

    print(mcts_actions)

    return game


def episode(
            env,
            mcts_agent,
            a2c_agent,
            n_episode,
            replay_buffer,
            params,
            start_time
        ):
    n_actors = params["n_actors"]
    train_steps = params["train_steps"]

    # MCTS is caching policy net predictions, so it is important to reset that
    # cache here since the net is updated in the previous episode.
    mcts_agent.reset_policy_cache()

    # Run self play games in 4 parallel processes.
    pool = multiprocessing.Pool(processes=4)
    multiple_results = [
        pool.apply_async(run_actor, (env, params, mcts_agent, a2c_agent))
        for _ in range(n_actors)
    ]
    games = ([res.get() for res in multiple_results])
    for game in games:
        replay_buffer.add(game)
    pool.close()

    # Print debug information
    print(mcts_agent.policy_net_cache)
    for i in range(10):
        print(
            i-5,
            round(a2c_agent.predict_policy([[i-5, False]])[1].item(), 6),
            end=' | '
        )
    print('')
    for i in range(10):
        print(
            i-5,
            round(a2c_agent.predict_policy([[i-5, True]])[1].item(), 6),
            end=' | '
        )
    print('')
    print(replay_buffer.get_rewards()[-n_actors:])

    # Train network after self play.
    lens = 0
    losses = 0
    for i in range(train_steps):
        samples = replay_buffer.sample()
        for sample in samples:
            loss = a2c_agent.update(*sample)

        actions = [sample[1] for sample in samples]
        if i % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        lens += len(actions)
        losses += loss.item()

    print("")
    print(
        "AVG LENS",
        lens / train_steps,
        " |AVG LOSS",
        losses / train_steps,
        " |TIME", time.time() - start_time
    )
    return evaluate(env, a2c_agent)


def run(env, params, desired_eval_len=6):
    start_time = time.time()
    replay_buffer = ReplayBuffer(params["memory_capacity"], params["prioritized_sampling"])
    a2c_agent = a2c.A2CLearner(params)
    mcts_agent = MCTS(
        params["env"],
        a2c_agent,
        params
    )
    for i in range(params["episodes"]):
        eval_len = episode(
            env,
            mcts_agent,
            a2c_agent,
            i,
            replay_buffer,
            params,
            start_time
        )
        if eval_len == desired_eval_len:
            break
    return i
