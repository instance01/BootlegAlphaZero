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
            sum(sample[1] for sample in samples)
            for samples in self.buffer
        ])
        return p

    def sample(self):
        if not self.prioritized_sampling:
            # Uniform sampling
            return self.buffer[np.random.choice(len(self.buffer))]

        p = self.get_rewards()
        # Favor high rewards.
        # Deeply negative rewards still get a high chance though.
        # TODO Reconsider this.
        p *= p
        if p.sum() == 0:
            return self.buffer[np.random.choice(len(self.buffer))]
        p /= p.sum()
        return self.buffer[np.random.choice(len(self.buffer), p=p)]


def evaluate_with_policy(env, params, a2c_agent, policy):
    state = env.reset()

    done = False
    total_reward = 0
    actions = ""
    i = 0
    while not done:
        action_probs, val = a2c_agent.predict_policy([state])

        vals = []
        for action in range(params['n_actions']):
            env_ = copy.deepcopy(env)
            state_, _, _, _ = env_.step(action)
            _, val = a2c_agent.predict_policy([state_])
            vals.append(val.detach())

        i += 1
        action = policy(i, state, action_probs, vals)

        state, reward, done, _ = env.step(action)
        total_reward += reward
        actions += str(action)

    return actions, total_reward


def evaluate(env, params, a2c_agent):
    env = copy.deepcopy(env)

    def action_policy(i, state, action_probs, vals):
        # Policy that takes the argmax of the action probabilities.
        if i < 10:
            print(state, action_probs.detach().tolist(), vals)
        return np.argmax(action_probs.tolist()[0])

    def val_policy(i, state, action_probs, vals):
        # Policy that takes the argmax of the value.
        if i < 10:
            print(state, vals)
        return np.argmax(vals)

    actions, total_reward = evaluate_with_policy(
        env, params, a2c_agent, action_policy
    )
    evaluate_with_policy(env, params, a2c_agent, val_policy)
    print("EVAL", actions, total_reward)
    return len(actions), total_reward


def run_actor(env, params, mcts_agent, a2c_agent):
    # Since we're running in parallel.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    env = copy.deepcopy(env)
    mcts_agent = copy.deepcopy(mcts_agent)
    mcts_agent.reset_policy_cache()

    # TODO REMOVE
    mcts_actions = ""

    state = env.reset()
    done = False
    game = []
    for _ in range(params["horizon"]):
        mcts_action = mcts_agent.policy(env, state)
        sampled_action = np.random.choice(
            list(range(params["n_actions"])),
            p=mcts_action
        )
        mcts_actions += str(sampled_action)

        action_probs, _ = a2c_agent.predict_policy([state])

        next_state, reward, done, _ = env.step(sampled_action)
        sample = (state, reward, mcts_action)
        state = next_state
        game.append(sample)

        if done:
            break

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

    # Run self play games in n_procs parallel processes.
    pool = multiprocessing.Pool(processes=params["n_procs"])
    multiple_results = [
        pool.apply_async(run_actor, (env, params, mcts_agent, a2c_agent))
        for _ in range(n_actors)
    ]
    games = ([res.get() for res in multiple_results])
    for game in games:
        replay_buffer.add(game)
        actor_lengths.append(len(game))
    pool.close()

    # Print debug information.
    print('')
    print(replay_buffer.get_rewards()[-n_actors:])

    # Print how confident the MCTS is.
    max_action_probs = [
        np.max(transaction[-1])
        for transaction in game
        for game in replay_buffer.buffer[-n_actors:]
    ]
    print("CONFIDENCE", np.mean(max_action_probs), np.median(max_action_probs))

    # Train network after self play.
    samples_used = defaultdict(int)
    sample_lens = []
    losses = []

    for i in range(train_steps):
        game = replay_buffer.sample()
        loss = a2c_agent.update(game)

        actions = ''.join([str(np.argmax(sample[-1])) for sample in game])
        samples_used[actions] += 1
        if i % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        sample_lens.append(len(actions))
        losses.append(loss.item())

    if params["schedule_alpha"]:
        a2c_agent.scheduler.step()

    print("")
    print(samples_used)

    print("")
    avg_loss = sum(losses) / train_steps
    print(
        "AVG LENS",
        sum(sample_lens) / train_steps,
        " |AVG LOSS",
        avg_loss,
        " |TIME", time.time() - start_time
    )
    eval_length, total_reward = evaluate(env, params, a2c_agent)
    writer.add_scalar('Eval/Length/%d' % n_run, eval_length, n_episode)
    writer.add_scalar('Eval/Reward/%d' % n_run, total_reward, n_episode)
    writer.add_histogram(
        'Actor/Sample_length/%d' % n_run, np.array(actor_lengths), n_episode
    )
    writer.add_histogram(
        'Train/Samples/%d' % n_run, np.array(sample_lens), n_episode
    )
    writer.add_histogram(
        'Train/Loss/%d' % n_run, np.array(losses), n_episode
    )
    # TODO The decision to only use the first param group might be dubious.
    # Keep that in mind. For now it is fine, I checked.
    writer.add_histogram(
        'Train/LearningRate/%d' % n_run,
        a2c_agent.policy_optimizer.param_groups[0]['lr'],
        n_episode
    )
    writer.add_scalar('Train/AvgLoss/%d' % n_run, avg_loss, n_episode)
    return eval_length, total_reward


def run(env, params, n_run, writer=None):
    if not writer:
        writer = SummaryWriter()
    writer.add_text('Info/params/%d' % n_run, str(params), 0)

    start_time = time.time()
    replay_buffer = ReplayBuffer(
        params["memory_capacity"],
        params["prioritized_sampling"]
    )
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
        eval_len, total_reward = episode(
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
        if eval_len <= params["desired_eval_len"]:
            is_done_stably += 1
        else:
            is_done_stably = 0
        if is_done_stably > params["n_desired_eval_len"]:
            break
    return i, eval_len, total_reward
