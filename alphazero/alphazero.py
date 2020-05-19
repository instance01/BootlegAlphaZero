import os
import time
import sys
import copy
import multiprocessing
from collections import defaultdict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    return len(actions), total_reward


def one_hot_encode(action, actions_len):
    action_one_hot = [0. for _ in range(actions_len)]
    action_one_hot[action] = 1.
    return action_one_hot


def run_actor(env, params, mcts_agent, a2c_agent):
    # Since we're running in parallel.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    # TODO: Is that even needed?
    env = copy.deepcopy(env)
    mcts_agent = copy.deepcopy(mcts_agent)
    a2c_agent = copy.deepcopy(a2c_agent)

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
            writer,
            n_run,
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

    actor_lengths = []

    # Run self play games in 4 parallel processes.
    pool = multiprocessing.Pool(processes=4)
    multiple_results = [
        pool.apply_async(run_actor, (env, params, mcts_agent, a2c_agent))
        for _ in range(n_actors)
    ]
    games = ([res.get() for res in multiple_results])
    for game in games:
        replay_buffer.add(game)
        actor_lengths.append(len(game))
    pool.close()

    # Print debug information
    print('')
    print(replay_buffer.get_rewards()[-n_actors:])

    # Train network after self play.
    samples_used = defaultdict(int)  # TODO REMOVE
    sample_lens = []
    losses = 0
    for i in range(train_steps):
        game = replay_buffer.sample()
        loss = a2c_agent.update(game)

        actions = ''.join([str(np.argmax(sample[-1])) for sample in game])
        samples_used[actions] += 1
        if i % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        sample_lens.append(len(actions))
        losses += loss.item()
    print("")
    print(samples_used)

    print("")
    print(
        "AVG LENS",
        sum(sample_lens) / train_steps,
        " |AVG LOSS",
        losses / train_steps,
        " |TIME", time.time() - start_time
    )
    eval_length, total_reward = evaluate(env, a2c_agent)
    writer.add_scalar('Eval/Length/%d' % n_run, eval_length, n_episode)
    writer.add_scalar('Eval/Reward/%d' % n_run, total_reward, n_episode)
    writer.add_histogram('Actor/Sample_length/%d' % n_run, np.array(actor_lengths), n_episode)
    writer.add_histogram('Train/Samples/%d' % n_run, np.array(sample_lens), n_episode)
    return eval_length


def run(env, params, desired_eval_len, n_run, writer=None):
    if not writer:
        writer = SummaryWriter()
    writer.add_text('Info/params/%d' % n_run, str(params), 0)

    start_time = time.time()
    replay_buffer = ReplayBuffer(params["memory_capacity"], params["prioritized_sampling"])
    a2c_agent = a2c.A2CLearner(params)
    mcts_agent = MCTS(
        params["env"],
        a2c_agent,
        params
    )

    # Need to have less than or equal desired evaluation length, 5 times in a
    # row. So we want to have very good paths 3 times in a row.
    is_done_stably = 0

    for i in range(params["episodes"]):
        eval_len = episode(
            writer,
            n_run,
            env,
            mcts_agent,
            a2c_agent,
            i,
            replay_buffer,
            params,
            start_time
        )
        if eval_len <= desired_eval_len:
            is_done_stably += 1
        else:
            is_done_stably = 0
        if is_done_stably > 5:
            break
    return i, eval_len
