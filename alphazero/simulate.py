#!/usr/local/bin/python3
import sys
import time
import copy
import functools
from enum import Enum
from multiprocessing import set_start_method

import gym
import numpy as np

#import mini_envs.discrete_envs.mini_discrete_env  # noqa: F401
import gym_minigrid  # noqa: F401
import alphazero
from torch.utils.tensorboard import SummaryWriter


def get_params():
    params1 = {
        "n_actions": 2,
        "n_input_features": 2,

        # MCTS
        "gamma": .99,
        "c": 1.,  # .001  # Using puct now.
        "simulations": 50,  # 1000
        "horizon": 200,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .25,
        "pb_c_base": 50,  # TODO I think this should be based on simulations.
        # base was 19652. But with just 100 simulations (and thus visits only
        # getting to 100 at max) visits don't matter.. At base=50, hell yes !
        "pb_c_init": 1.25,

        # A2C
        "alpha": .01,  # .01 (AlphaZero best); .001 (Imitation best)
        "net_architecture": [64, 64],
        "schedule_alpha": False,
        "scheduler_class": "ExponentialLR",
        "scheduler_args": [.995],
        "scheduler_kwargs": {},

        # AlphaZero
        "memory_capacity": 1000,
        "prioritized_sampling": True,
        "episodes": 100,
        "n_procs": 4,
        "n_actors": 20,  # 5000
        "train_steps": 2000,  # 700000
        "desired_eval_len": 8,
        "n_desired_eval_len": 10,

        # Other
        "reward_exponent": 1,

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

    params29 = copy.deepcopy(params1)
    params29.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 3000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params30 = copy.deepcopy(params1)
    params30.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 20,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params31 = copy.deepcopy(params1)
    params31.update({
        "alpha": .005,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 20,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params32 = copy.deepcopy(params1)
    params32.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 20,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params33 = copy.deepcopy(params1)
    params33.update({
        "alpha": .05,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 20,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params34 = copy.deepcopy(params1)
    params34.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 1000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params35 = copy.deepcopy(params1)
    params35.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 1000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .2
    })

    params36 = copy.deepcopy(params1)
    params36.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 1000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .2,
        "pb_c_base": 100
    })

    paramsPROF = copy.deepcopy(params1)
    paramsPROF.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 2,
        "train_steps": 1000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "episodes": 2,
        # "horizon": 1024
    })

    params37 = copy.deepcopy(params1)
    params37.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 3000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params38 = copy.deepcopy(params1)
    params38.update({
        "alpha": .005,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 3000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params39 = copy.deepcopy(params1)
    params39.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    params40 = copy.deepcopy(params1)
    params40.update({
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 5000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5
    })

    # With paramsCURR I had a quickly iterating approach.
    # This was just temporary, I changed something quickly and checked whether
    # it works after watching output for one run.

    # value function sometimes is fine.. policy not learnt.
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        # "dirichlet_frac": .5
        "dirichlet_frac": .25  # both worked btw. and .25 was a bit better.
    })

    # Works well, with value
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .01,
        "simulations": 50,
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .25
    })

    # ... fucking shit
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 200,  # 50
        "prioritized_sampling": False,
        "n_actors": 10,
        "train_steps": 4000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .25,
        "pb_c_base": 10
    })

    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 400,  # 50
        "prioritized_sampling": False,
        "n_actors": 40,  # 10
        "train_steps": 4000,
        "dirichlet_alpha": .2,
        "dirichlet_frac": .15,
        "pb_c_base": 10
    })

    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 200,  # 50
        "prioritized_sampling": True,
        "n_actors": 40,  # 10
        "train_steps": 4000,
        "dirichlet_alpha": .2,
        "dirichlet_frac": .15,
        "pb_c_base": 10,
        "reward_exponent": 8,
        "memory_capacity": 80
    })

    # Sanity check: // works
    paramsCURR = copy.deepcopy(params19)
    paramsCURR.update({
        "reward_exponent": 8
    })

    # WORKS AMAZING! But: Need a bit more randomness.. Now it gets suboptimal
    # solutions too.
    # The thing that worked: pb_c_init. So now we learn policy too.
    # sodalith
    # May21_12-54-01_sodalith.cip.ifi.lmu.de
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 100,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 10000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 19000,
        "pb_c_init": .25,
        "horizon": 256,  # For 8x8
    })

    # beryll (PRETTY GOOD)
    # May21_15-44-50_beryll.cip.ifi.lmu.de
    # EVAL 00022222022222 0.95078125
    # Minutes: 1012.9252081076304
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 100,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 10000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 19000,
        "pb_c_init": .3,  # .25
        "horizon": 256,  # For 8x8
    })

    # amazonit, danburit
    # May21_22-25-31_danburit.cip.ifi.lmu.de
    # May21_22-45-40_amazonit.cip.ifi.lmu.de
    # Pretty bad.
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 100,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 10000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 10,
        "pb_c_init": .1,
        "horizon": 256,  # For 8x8
    })

    # original; danburit
    # May22_06-59-34_danburit.cip.ifi.lmu.de
    paramsCURR = copy.deepcopy(params1)
    paramsCURR.update({
        "alpha": .001,
        "simulations": 100,
        "prioritized_sampling": True,
        "n_actors": 10,
        "train_steps": 10000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 19000,
        "pb_c_init": .25,
        "horizon": 256,  # For 8x8
    })

    # sodalith (BEST), but slow.
    # May22_07-45-00_sodalith.cip.ifi.lmu.de
    # Minutes: 268.5126764456431
    params41 = copy.deepcopy(params1)
    params41.update({
        "alpha": .001,
        "simulations": 100,
        "prioritized_sampling": True,
        "n_actors": 20,
        "train_steps": 5000,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 19000,
        "pb_c_init": .3,
        "horizon": 256,  # For 8x8
        "episodes": 200,
        "desired_eval_len": 18,
    })

    params42 = copy.deepcopy(params41)
    params42.update({
        "dirichlet_alpha": .2,
        "dirichlet_frac": .25
    })

    params43 = copy.deepcopy(params41)
    params43.update({
        "prioritized_sampling": False
    })

    params44 = copy.deepcopy(params41)
    params44.update({
        "alpha": .005
    })

    params45 = copy.deepcopy(params41)
    params45.update({
        "prioritized_sampling": False,
        "n_actors": 40
    })

    params46 = copy.deepcopy(params41)
    params46.update({
        "pb_c_init": .334
    })

    params47 = copy.deepcopy(params41)
    params47.update({
        "pb_c_init": .4
    })

    params48 = copy.deepcopy(params41)
    params48.update({
        "dirichlet_alpha": .4,
        "dirichlet_frac": .5
    })

    params49 = copy.deepcopy(params41)
    params49.update({
        "alpha": .0005,
    })

    params50 = copy.deepcopy(params41)
    params50.update({
        "pb_c_base": 50
    })

    params51 = copy.deepcopy(params41)
    params51.update({
        "pb_c_init": .25
    })

    params52 = copy.deepcopy(params41)
    params52.update({
        "pb_c_base": 500
    })

    params53 = copy.deepcopy(params41)
    params53.update({
        "pb_c_base": 1
    })

    params54 = copy.deepcopy(params41)
    params54.update({
        "pb_c_base": 5000
    })

    params55 = copy.deepcopy(params41)
    params55.update({
        "simulations": 150,
        "horizon": 1024,  # For 16x16
        "n_procs": 8,
        "desired_eval_len": 40,
        "n_desired_eval_len": 15,
    })

    params56 = copy.deepcopy(params41)
    params56.update({
        "simulations": 150,
        "horizon": 1024,  # For 16x16
        "reward_exponent": 8,
        "n_procs": 8,
        "desired_eval_len": 40,
        "n_desired_eval_len": 15,
    })

    params57 = copy.deepcopy(params41)
    params57.update({
        # For 16x16
        "simulations": 150,
        "horizon": 400,  # We won't need longer paths anyways.
        "reward_exponent": 8,
        "n_actors": 40,
        "train_steps": 6000,
        "n_procs": 8,
        "desired_eval_len": 40,
        "n_desired_eval_len": 15,
    })

    params58 = copy.deepcopy(params41)
    params58.update({
        # For 16x16
        "simulations": 150,
        "horizon": 400,
        "reward_exponent": 8,
        "n_actors": 60,
        "train_steps": 8000,
        "n_procs": 8,
        "desired_eval_len": 40,
        "n_desired_eval_len": 15,
    })

    params59 = copy.deepcopy(params41)
    params59.update({
        "episodes": 300,
        "simulations": 150,
        "horizon": 1024,
        "n_actors": 30,
        "n_procs": 8,
        "desired_eval_len": 40,
        "n_desired_eval_len": 15,
    })

    params60 = copy.deepcopy(params57)
    params60.update({
        "horizon": 300,  # Bug: This had no effect !!
    })

    params61 = copy.deepcopy(params57)
    params61.update({
        "n_procs": 12,
        "reward_exponent": 16
    })

    params62 = copy.deepcopy(params57)
    params62.update({
        "n_procs": 12,
        "net_architecture": [128, 64]
    })

    params63 = copy.deepcopy(params57)
    params63.update({
        # "n_procs": 12,
        "n_procs": 8,
        "net_architecture": [128, 128, 64, 64]
    })

    params64 = copy.deepcopy(params57)
    params64.update({
        "n_procs": 12,
        "alpha": .002
    })

    params65 = copy.deepcopy(params57)
    params65.update({
        "n_procs": 12,
        "alpha": .005,
        "train_steps": 7000
    })

    params66 = copy.deepcopy(params57)
    params66.update({
        "n_procs": 12,
        "horizon": 200  # Bug: This had no effect !!
    })

    params67 = copy.deepcopy(params57)
    params67.update({
        "n_procs": 12,
        "alpha": .01,
        "train_steps": 8000,
        "schedule_alpha": True
    })

    params68 = copy.deepcopy(params57)
    params68.update({
        "n_procs": 12,
        "reward_exponent": 1,
        "train_steps": 8000
    })

    params69 = copy.deepcopy(params57)
    params69.update({
        "n_procs": 12,
        "alpha": .001,
        "train_steps": 8000,
        "schedule_alpha": True
    })

    params70 = copy.deepcopy(params57)
    params70.update({
        "n_procs": 12,
        "n_actors": 10,
        "simulations": 50,
        "train_steps": 1000,
        "episodes": 300,
    })

    params71 = copy.deepcopy(params57)
    params71.update({
        "n_procs": 12,
        "n_actors": 10,
        "simulations": 50,
        "train_steps": 2000,
        "episodes": 300,
    })

    params72 = copy.deepcopy(params69)
    params72.update({
        # Includes schedule_alpha=True
        "train_steps": 9000
    })

    params73 = copy.deepcopy(params69)
    params73.update({
        # Includes schedule_alpha=True
        "scheduler_gamma": .99
    })

    params74 = copy.deepcopy(params69)
    params74.update({
        # Includes schedule_alpha=True
        "prioritized_sampling": False
    })

    params75 = copy.deepcopy(params69)
    params75.update({
        # Includes schedule_alpha=True
        "prioritized_sampling": False,
        "simulations": 50,
        "n_actors": 30,
        "train_steps": 3000
    })

    params76 = copy.deepcopy(params69)
    params76.update({
        # Includes schedule_alpha=True
        "n_procs": 8,
        "net_architecture": [128, 128, 64, 64]
    })

    # basically merge of params71 and params69
    params77 = copy.deepcopy(params69)
    params77.update({
        # Includes schedule_alpha=True
        "n_procs": 12,
        "n_actors": 10,
        "simulations": 50,
        "train_steps": 2000,
        "episodes": 300,
    })

    # Basically params72 but let's explore further
    params78 = copy.deepcopy(params69)
    params78.update({
        # Includes schedule_alpha=True
        "n_procs": 8,
        "train_steps": 10000
    })

    params79 = copy.deepcopy(params75)
    params79.update({
        "n_procs": 8,
        "episodes": 500
    })

    params80 = copy.deepcopy(params68)
    params80.update({
        "n_procs": 8,
        "prioritized_sampling": False
    })

    params81 = copy.deepcopy(params79)
    params81.update({
        "n_procs": 8,
        "prioritized_sampling": False,
        "reward_exponent": 1,

        # Just for clarity. The params from parent configs.
        "episodes": 500,
        "simulations": 50,
        "n_actors": 30,
        "train_steps": 3000,
        "alpha": .001,
        "schedule_alpha": True,
        "horizon": 400,  # Keep in mind: maximum actions is 1024!
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 19000,
        "pb_c_init": .3,
        # For 16x16
        "desired_eval_len": 40,
        "n_desired_eval_len": 15
    })

    params82 = copy.deepcopy(params68)
    params82.update({
        "n_procs": 8,
        "simulations": 50,
        "n_actors": 50,
        "prioritized_sampling": False
    })

    params83 = copy.deepcopy(params75)
    params83.update({
        "n_procs": 8,
        "episodes": 500,
        "scheduler_gamma": .99,
        "train_steps": 4000
    })

    params84 = copy.deepcopy(params69)
    params84.update({
        "n_procs": 8,
        "train_steps": 10000,
        "simulations": 50
    })

    params85 = copy.deepcopy(params81)
    params85.update({
        "train_steps": 5000
    })

    # Sanity check
    params86 = copy.deepcopy(params81)
    params86.update({
        "schedule_alpha": False
    })

    params87 = copy.deepcopy(params81)
    params87.update({
        "scheduler_class": "MultiStepLR",  # AlphaZero uses step too.
        "scheduler_args": [[200, 400]]
        # "scheduler_class": "LambdaLR",
        # "scheduler_args": [lambda x: math.max(.0001, .995 ** x)]
    })

    # This is an alternative to params81 that also works.
    params88 = copy.deepcopy(params82)
    params88.update({
        "episodes": 500,

        # Just for clarity. The params from parent configs.
        "reward_exponent": 1,
        "train_steps": 8000,
        "horizon": 400,
        "alpha": .001,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "pb_c_base": 19000,
        "pb_c_init": .3,
        # For 16x16
        "desired_eval_len": 40,
        "n_desired_eval_len": 15,
    })

    params89 = copy.deepcopy(params83)
    params89.update({
        "alpha": .005  # instead of .001
    })

    params90 = copy.deepcopy(params84)
    params90.update({
        "episodes": 500
    })

    params91 = copy.deepcopy(params81)
    params91.update({
        # More randomness
        "dirichlet_alpha": .4,
        "dirichlet_frac": .5,
        "pb_c_init": .2,
    })

    # 88 has more actors and train steps
    params92 = copy.deepcopy(params88)
    params92.update({
        # More randomness
        "dirichlet_alpha": .4,
        "dirichlet_frac": .5,
        "pb_c_init": .2,
        "n_actors": 70  # 50 -> 70
    })

    params93 = copy.deepcopy(params92)
    params93.update({
        "alpha": 0.005  # Was: 0.001
    })

    params94 = copy.deepcopy(params92)
    params94.update({
        "alpha": 0.0002  # Was: 0.001
    })

    params95 = copy.deepcopy(params92)
    params95.update({
        "alpha": 0.01  # Was: 0.001
    })

    # Still, let's try same 4 as above but with 81 as base.
    params96 = copy.deepcopy(params88)
    params96.update({
        # More randomness
        "dirichlet_alpha": .4,
        "dirichlet_frac": .5,
        "pb_c_init": .2,
        "n_actors": 50  # 30 -> 50
    })

    params97 = copy.deepcopy(params96)
    params97.update({
        "alpha": 0.005  # Was: 0.001
    })

    params98 = copy.deepcopy(params96)
    params98.update({
        "alpha": 0.0002  # Was: 0.001
    })

    params99 = copy.deepcopy(params96)
    params99.update({
        "alpha": 0.01  # Was: 0.001
    })

    params100 = copy.deepcopy(params96)
    params100.update({
        "n_actors": 20  # 30 -> 20
    })

    paramsPROF = copy.deepcopy(params81)
    paramsPROF.update({
        "horizon": 100,#  1024,
        "n_procs": 1,
        "alpha": .001,
        "simulations": 50,
        "prioritized_sampling": True,
        "n_actors": 1,
        "train_steps": 1,
        "dirichlet_alpha": .3,
        "dirichlet_frac": .5,
        "episodes": 1,
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
        "29": params29,
        "30": params30,
        "31": params31,
        "32": params32,
        "33": params33,
        "34": params34,
        "35": params35,
        "36": params36,
        "PROF": paramsPROF,
        "37": params37,
        "38": params38,
        "39": params39,
        "40": params40,
        "CURR": paramsCURR,
        "41": params41,
        "42": params42,
        "43": params43,
        "44": params44,
        "45": params45,
        "46": params46,
        "47": params47,
        "48": params48,
        "49": params49,
        "50": params50,
        "51": params51,
        "52": params52,
        "53": params53,
        "54": params54,
        "55": params55,
        "56": params56,
        "57": params57,
        "58": params58,
        "59": params59,
        "60": params60,
        "61": params61,
        "62": params62,
        "63": params63,
        "64": params64,
        "65": params65,
        "66": params66,
        "67": params67,
        "68": params68,
        "69": params69,
        "70": params70,
        "71": params71,
        "72": params72,
        "73": params73,
        "74": params74,
        "75": params75,
        "76": params76,
        "77": params77,
        "78": params78,
        "79": params79,
        "80": params80,
        "81": params81,
        "82": params82,
        "83": params83,
        "84": params84,
        "85": params85,
        "86": params86,
        "87": params87,
        "88": params88,
        "89": params89,
        "90": params90,
        "91": params91,
        "92": params92,
        "93": params93,
        "94": params94,
        "95": params95,
        "96": params96,
        "97": params97,
        "98": params98,
        "99": params99,
        "100": params100,
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

Params3:
    [7, 8, 7, 8, 3, 3, 6, 1, 1, 0]
    Minutes: 602.0077656269074

Params4:
    Crashed, gradient explosion
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
    Rerun:
    +dont quadruple rewards
    +do 10 runs (from now on charts)
    Cancelled after 3 runs. Did not learn anything stable.
    May18_11-50-53_amazonit.cip.ifi.lmu.de

Params25:
    EVAL 1202212 0.7708295649609999
    Minutes: 59.53636395931244
    Rerun:
    +dont quadruple rewards
    +do 10 runs (from now on charts)
    Cancelled after 4 runs. Did not learn anything stable.
    May18_11-49-16_amazonit.cip.ifi.lmu.de

Params26:
    EVAL 1202212 0.7708295649609999
    Minutes: 120.59210259517035
    Rerun:
    +dont quadruple rewards
    +do 10 runs (from now on charts)
    Cancelled after 4 runs. Learnt something a bit stable once. All other
    times, did not learn anything. Very slow.
    May18_11-57-17_beryll.cip.ifi.lmu.de

Params27:
    EVAL 1202212 0.7708295649609999
    Minutes: 134.39488005638123
    Rerun:
    +dont quadruple rewards
    +do 10 runs (from now on charts)
    Cancelled after 3 runs. Did not learn anything stable.
    May18_14-01-25_beryll.cip.ifi.lmu.de

Params28: // btw this was POMDP version.
    EVAL 122022 0.8008746470559999
    Minutes: 65.5352343996366
    Rerun:
    +dont quadruple rewards
    +do 10 runs (from now on charts)
    May18_11-51-21_danburit.cip.ifi.lmu.de
    Minutes: 218.93244425058364
    9/10 learnt successfully, 1/10 did not learn.

Params29:
    Cancelled after 2 runs. Did not learn and much too slow.
    May18_17-03-06_danburit.cip.ifi.lmu.de

Params30:
    Cancelled after 1 run. Did not learn and much too slow.
    May18_17-07-36_danburit.cip.ifi.lmu.de

Params31:
    Cancelled after 1 run. Did not learn and much too slow.
    May18_18-24-02_danburit.cip.ifi.lmu.de

Params19_again:
    +Exponentiate rewards with 8
    Minutes: 77.49310787518819
    Learnt 10/10.
    May18_18-56-37_sodalith.cip.ifi.lmu.de

Params19_again:
    +8x8
    Cancelled after 2 runs, did not learn. Will first explore 5x5 some more.
    May18_19-26-47_sodalith.cip.ifi.lmu.de

Params19_again: // just to test summary tensorboard (justatest)
    -8x8
    -Exponentiate rewards with 8
    May18_22-36-39_sodalith.cip.ifi.lmu.de

Params28_again:
    Nothing

Params32:
    +Refactored reward_exponent, game and key
    Nothing

Params33:
    Nothing

Params34:
    May19_17-28-02_amazonit.cip.ifi.lmu.de
    Did not learn anything

Params35:
    May19_17-28-13_amazonit.cip.ifi.lmu.de
    Did not learn anything

Params36:
    May19_17-28-35_amazonit.cip.ifi.lmu.de
    Did not learn anything

Params19_again:
    +5x5 but no pomdp
    +no reward exponentiation
    Minutes: 193.41552900870641
    May19_07-38-31_sodalith.cip.ifi.lmu.de
    Learnt 4/10, didn't learn 6/10

Params28_again:
    +5x5 but no pomdp
    +no reward exponentiation
    Minutes: 182.95773999293644
    May19_07-49-15_sodalith.cip.ifi.lmu.de
    Learnt 7/10, didn't learn 3/10

Params28_again:
    May19_17-21-56_sodalith.cip.ifi.lmu.de
    Did not learn anything

Params37:
    May19_17-29-27_beryll.cip.ifi.lmu.de
    Did not learn anything

Params38:
    May19_17-29-37_beryll.cip.ifi.lmu.de
    Learnt 1/10, didn't learn 9/10.

Params39:
    May19_17-30-52_sodalith.cip.ifi.lmu.de
    Did not learn anything

Params40:
    May19_17-31-00_sodalith.cip.ifi.lmu.de
    Learnt 2/10. Did not learn 8/10.

Params41:
    May22_22-57-11_sodalith.cip.ifi.lmu.de
    Minutes: 207.7851839462916
    Learnt 10/10. Rather quick.
    Another run:
    May23_09-52-35_sodalith.cip.ifi.lmu.de
    Minutes: 240.90073643128076
    BUT: Suboptimal solutions.

Params42:
    May22_22-58-34_danburit.cip.ifi.lmu.de
    Minutes: ~Roughly a day
    Learnt 9/9. VERY slow.

Params43:
    May22_22-59-15_beryll.cip.ifi.lmu.de
    Minutes: 575.4137872378031
    Learnt 10/10. Slow.
    But this is without prioritized sampling!!

Params44:
    May22_22-59-54_amazonit.cip.ifi.lmu.de
    Minutes: ~Roughly 19 hours.
    Learnt 5/7. Did not learn 2/7. Very slow.

Params45:
    May23_11-58-18_beryll.cip.ifi.lmu.de
    Minutes: 472.1143970608711
    Learnt 10/10. Slow, but a bit better than Params43.

Params41_again: (5x5)
    May23_11-58-21_indigiolith.cip.ifi.lmu.de
    Minutes: 69.30242917537689
    Learnt 10/10.
    Good: Params generalize.
    BUT: Suboptimal solutions.

Params46:
    May23_14-20-16_indigiolith.cip.ifi.lmu.de
    Minutes: 253.66197155714036
    Learnt 10/10.

Params47:
    May23_19-55-29_beryll.cip.ifi.lmu.de
    Minutes: ~600
    Learnt 9/9.

Params48:
    May23_19-55-49_sodalith.cip.ifi.lmu.de
    Minutes: ~342
    Learnt 10/10.

Params49:
    May23_19-55-59_indigiolith.cip.ifi.lmu.de
    Minutes: ~278
    Learnt 10/10.

Params50:
    May24_09-33-49_sodalith.cip.ifi.lmu.de
    Cancelled. Too slow and did not learn good enough.
    Learnt 1/5. (but 12, good solution..)

Params51:
    May24_09-34-06_indigiolith.cip.ifi.lmu.de
    Minutes: 507.6325584053993
    Learnt 10/10. Slow.

Params52:
    For 5x5:
    May24_23-58-59_indigiolith.cip.ifi.lmu.de
    Minutes: 100.3497024377187
    Learnt 10/10. But a bit suboptimal paths.
    For 8x8:
    May25_09-06-02_indigiolith.cip.ifi.lmu.de
    Minutes: 360.7428737640381
    Learnt 10/10.

Params53:
    For 5x5:
    May25_00-00-59_sodalith.cip.ifi.lmu.de
    Cancelled. Too slow and did not learn good enough.
    Learnt 1/4.

Params54:
    For 5x5:
    May25_00-00-53_danburit.cip.ifi.lmu.de
    Minutes: 108.62775423924128
    Learnt 10/10. But a bit suboptimal paths.
    For 8x8:
    May25_09-45-27_danburit.cip.ifi.lmu.de
    Minutes: 722.0350600719452
    Learnt 10/10. Very Slow.

Params55: (16x16)
    May25_00-09-37_amazonit.cip.ifi.lmu.de
    Restarted.
    May26_08-54-04_amazonit.cip.ifi.lmu.de
    Minutes: 4302.391547409693
    Learnt 3/10. Did not learn 7/10.

Params56: (16x16)
    May25_21-39-12_sodalith.cip.ifi.lmu.de
    Restarted.
    May26_08-55-49_sodalith.cip.ifi.lmu.de
    Minutes: 2877.46658252875
    Learnt 4/10. Did not learn 6/10.

Params57: (16x16)
    May25_21-54-59_beryll.cip.ifi.lmu.de
    Restarted.
    May26_08-55-51_beryll.cip.ifi.lmu.de
    Minutes: 4159.658444670836
    Learnt 8/10. Did not learn 2/10.

Params58: (16x16)
    May25_22-00-59_danburit.cip.ifi.lmu.de
    Restarted.
    May26_08-55-55_danburit.cip.ifi.lmu.de
    Minutes: 5006.399745408694
    Learnt 7/10. Did not learn 3/10.

Params59: (16x16)
    May25_22-05-18_indigiolith.cip.ifi.lmu.de
    Restarted.
    May26_08-56-09_indigiolith.cip.ifi.lmu.de
    Minutes: 5517.272387822469
    Learnt 7/10. Did not learn 3/10.
    Once it straight up did not eval a good path even ONCE, for 300 episodes.

Params60: (16x16)
    Stopped early due to forced restart of the system.
    May28_23-41-36_sodalith.cip.ifi.lmu.de
    Learnt 6/8. Did not learn 2/8.
    Minutes: ~2700
    Doesnt seem to give much of an advantage. Need to try even crazier values.
    But it works.

Params61: (16x16)
    May31_06-48-23_amazonit.cip.ifi.lmu.de
    Minutes: 4144.575373788674
    Learnt 7/10. Did not learn 3/10. (but one more could be considered learnt)
    Increasing reward exponent to this amount has thus no effect.

Params62: (16x16)
    May31_06-48-23_beryll.cip.ifi.lmu.de
    Minutes: 3837.1865882476172
    Learnt 8/10. Did not learn 2/10. Often learnt quite quickly (in terms of
    episodes)
    Increasing net architecture might be successful. Unfortunately danburit is
    having issues, else I would already know.

Params63: (16x16)
    See further below.

Params64: (16x16)
    May31_06-48-42_euklas.cip.ifi.lmu.de
    Minutes: 3657.1515263160068
    Learnt 6/10. Did not learn 4/10.
    Increasing alpha on its own does not help much.

Params65: (16x16)
    Cancelled earlier due to bad performance.
    May31_06-48-30_indigiolith.cip.ifi.lmu.de
    Minutes: ~3780 (but for 9 simuls)
    Learnt 5/9. Did not learn 4/9.
    Increasing alpha and train steps does not help much.

Params66: (16x16)
    Cancelled earlier due to bad performance.
    May31_17-10-03_sodalith.cip.ifi.lmu.de
    Minutes: ~3840 (but for 9 simuls)
    Learnt 3/9. Did not learn 6/9. (but one more could be considered learnt)
    Horizon is a bad idea, at least with this config. Seems to work for MuZero
    though. Weird.

Params67: (16x16)
    Cancelled earlier due to bad performance.
    May31_17-10-09_peridot.cip.ifi.lmu.de
    Minutes: ~3660 (for 7 simuls only!)
    Learnt 1/7. Did not learn 6/7.
    Idea was to test scheduling alpha.
    However, alpha was set too high.
    For an experiment with current best alpha, see params69.

Params68: (16x16)
    See further below.

Params69: (16x16)
    May31_17-10-22_zirkon.cip.ifi.lmu.de
    Minutes: 4276.210630186399
    Learnt 9/10. Did not learn 1/10.
    Worked well, but is it much of an improvement to version without
    scheduling? Warrants some more experiments.

// Don't forget, the below are all horizon=400
Params70:
    Jun03_22-18-22_amazonit.cip.ifi.lmu.de
    Minutes: 1540.9452624837556
    Learnt 1/10. Did not learn 9/10. A few that could be considered nearly
    learnt.
    Was finished after just 22h!! So maybe just add more episodes?

Params71:
    Jun03_22-18-24_beryll.cip.ifi.lmu.de
    Minutes: 1955.9187905669212
    Learnt 3/10. Did not learn 7/10. A few times did not eval a single good
    path.
    With increasing train steps it becomes more extreme, which makes sense:
    We learn more than params70, but often we also don't learn anything at
    all.

Params72:
    Jun03_22-18-27_sodalith.cip.ifi.lmu.de
    Minutes: 3162.17279317379
    Learnt 9/10. Did not learn 1/10. But it was close.
    When it learns, it becomes stable extremely fast. Pretty good!

Params73:
    Jun03_22-18-29_indigiolith.cip.ifi.lmu.de
    Minutes: 3582.519363939762
    Learnt 9/10. Did not learn 1/10. But it was close.
    Another good candidate.

Params74:
    Jun03_22-18-32_euklas.cip.ifi.lmu.de
    Minutes: 4058.873200114568
    Learnt 8/10. Did not learn 2/10.
    Needs to be redone. Should've set reward_exponent to 1.

Params75:
    Jun03_22-18-34_peridot.cip.ifi.lmu.de
    Minutes: 1437.1275587677956
    Learnt 7/10. Did not learn 3/10. Some were close.
    Very fast, and we could do more episodes!

Params76:
    Jun10_16-31-40_rubin.cip.ifi.lmu.de
    Minutes: ~4500
    Only managed to learn 1 (!)
    Learnt 1/1. Second one did not learn anything, crashed before max episodes.
    Huge net architecture slows everything down insanely.

Params77:
    Jun03_22-18-44_goshenit.cip.ifi.lmu.de
    Minutes: 1817.216767668724
    Learnt 3/10. Did not learn 7/10.
    Makes sense, since params71 already had such issues.

// Moved here for visibiility
Params68: (16x16)
    May31_17-10-16_petalit.cip.ifi.lmu.de
    Crashed: Exploding gradients (?)
    Jun03_22-18-37_petalit.cip.ifi.lmu.de
    Minutes: ~3960
    Learnt 8/10. Did not learn 2/10.
    Amazing result. Apparently without reward exponent (only thing left: the
    square from prioritized_sampling=True) we still learn quite a lot.
    But very slow.

// moved here for visibility
Params63: (16x16)
    +set_start_method("spawn")
    Jun10_16-36-40_tansanit.cip.ifi.lmu.de
    Minutes: ~4740
    Learnt 0/1.
    Same problem as params76: Huge net architecture slows down.

+Python performance improvements (~2.5x)

Params78:
    Killed at the end.
    Jun07_05-46-02_amazonit.cip.ifi.lmu.de
    Minutes: ~4380
    Learnt 8/9. Did not learn 1/9. Last one wasn't doing that well either.

Params79:
    Jun07_05-46-02_beryll.cip.ifi.lmu.de
    Minutes: 1352.0987024943033
    Learnt 10/10. !!!!!!!!!!!!!! But one was a very close one (finished at
    episode 492/500). Fast.
    I think with 500 episodes learning rate scheduler should have a minimum.

Params80:
    Jun07_05-46-02_sodalith.cip.ifi.lmu.de
    Minutes: 4046.197966166337
    Learnt 8/10. Did not learn 2/10.

Params81:
    Jun07_05-46-02_indigiolith.cip.ifi.lmu.de
    Minutes: 1417.6139293591182
    Learnt 10/10. !!!!!!!!!!!!! Awesome result. Fast.
    It's based on 79 (but without prioritized sampling/reward exponent), so
    again: lr scheduler should have minimum.

Params82:
    Jun07_05-46-02_euklas.cip.ifi.lmu.de
    Minutes: 1187.9134648482004
    Learnt 7/10. Did not learn 3/10. Extremely fast, but probably due to low
    episodes. Need to increase.

Params83:
    Jun07_05-46-02_peridot.cip.ifi.lmu.de
    Minutes: 1036.8351935903231 Learned 9/10. Did not learn 1/10. Fast.
    The one time, it got close but then reverted and did not eval anything
    good for 300 episodes.
    -> From what I'm seeing scheduling is too much here. We have 500
    episodes. Decrease scheduling accordingly. Eg .99 -> .995 or .999
    Other than that, this seems to be the fastest good one.

Params84:
    Jun07_05-46-02_petalit.cip.ifi.lmu.de
    Minutes: 2271.3330056786535
    Learnt 9/10. Did not learn 1/10. Rather close. The 9 times it was mostly
    very fast, i.e. <100 episodes and done. Adding episodes=300 or so would
    make this one a 10/10.

Params85:
    Jun08_12-24-56_beryll.cip.ifi.lmu.de
    Minutes: 2312.261377151807
    Learnt 10/10. However often rather slow (>300 episodes).
    Not a good candidate.

Params86:
    Jun08_12-25-14_indigiolith.cip.ifi.lmu.de
    Minutes: 1805.1473118782044
    Learnt 8/10. Did not learn 2/10. Should be 3/10 because one was learnt very
    close. It seems not scheduling alpha is not so good, we have three cases
    with nearly or equal 500 episodes in this one.

Params87:
    Jun08_12-25-00_euklas.cip.ifi.lmu.de
    Minutes: 1096.2937799533208
    Learnt 9/10. Did not learn 1/10. MultiStep is probably not fit correctly.

Params88:
    Jun08_12-25-03_peridot.cip.ifi.lmu.de
    Minutes: 1171.3968317349752
    Learnt 10/10. Good alternative to params81.

Params89:
    Jun08_12-25-11_goshenit.cip.ifi.lmu.de
    Minutes: 2203.914812485377
    Learnt 10/10. Needs many episodes, rather slow.
    High lr (alpha) and aggressive scheduling (.99) leads to slowness.

Params81_again:
    just a sanity check whether it's really that fast.
    Jun10_16-18-27_zirkon.cip.ifi.lmu.de
    Minutes: ~1020
    Learnt 10/10.
    It's super fast. Nice.

Params90:  // 84 but with 500 episodes
    Jun10_16-22-54_saphir.cip.ifi.lmu.de
    Minutes: ~1620
    Learnt 10/10.

// From now on, MTCAR.
// Bad means no good rewards. Ok means a few ok rewards, could nearly be
// considered a success. Good means a success.

Params81_MTCAR:
    Jun10_16-25-10_diamant.cip.ifi.lmu.de
    Minutes: ~4300
    Bad: 7; Ok: 3; Good: 0;
    Bad ones were mostly flat lines. Ok ones become flat.

Params91:
    Jun10_16-27-06_goshenit.cip.ifi.lmu.de
    Minutes: ~4300
    Bad: 6; Ok: 4; Good: 0;
    Bad ones were mostly flat lines. Ok ones become flat.

Params92:
    Jun10_16-30-39_hackmannit.cip.ifi.lmu.de
    Minutes: ~3600
    Bad: 4; Ok: 1; Good: 0;
    Not enough time to finish.

Params93:
    Jun10_16-31-10_rubellit.cip.ifi.lmu.de
    Minutes: ~3600
    Bad: 4; Ok: 1; Good: 0;
    Not enough time to finish.

Params94:
    Jun10_16-45-18_thulit.cip.ifi.lmu.de
    Minutes: ~3600
    Bad: 3; Ok: 1; Good: 0;
    Not enough time to finish.

Params95:
    Jun10_16-17-09_hambergit.cip.ifi.lmu.de
    Minutes: ~3600
    Bad: 4; Ok: 1; Good: 0;
    Not enough time to finish.

Params96:
    Jun10_20-06-55_peridot.cip.ifi.lmu.demtcar.96
    Minutes: ~3600
    Bad: 4; Ok: 0; Good: (1);
    One that is nearly good. Not enough time to finish.

Params97:
    Jun10_20-08-05_pyrit.cip.ifi.lmu.de-mtcar.97
    Minutes: ~3600
    Bad: 5; Ok: 0; Good: 0;
    No good.

Params98:
    Jun10_20-08-05_rhodonit.cip.ifi.lmu.de-mtcar.98
    Minutes: ~3600
    Bad: 4; Ok: 1; Good: 0;
    Not enough time to finish.

Params99:
    Jun10_20-08-07_smaragd.cip.ifi.lmu.de-mtcar.99
    Minutes: ~3600
    Bad: 4; Ok: 1; Good: 0;
    Not enough time to finish.

Params100:
    Jun10_20-08-05_feuerachat.cip.ifi.lmu.de-mtcar.100
    Minutes: ~4080
    Bad: 9; Ok: 1; Good: 0;
    Pretty bad. But fast.

TODO:
    Explore more lr schedules
    Explore more net architectures

CURRENT BEST:
Params41 prio=True
Params45 prio=False
Params57 prio=True (8x8, 16x16)
Params69 prio=True (16x16)
Params79 prio=True (16x16)
Params81 prio=False (16x16)  // 10/10
Params88 prio=False (16x16)  // 10/10, alternative to 81

"""


def simulate_many_minidiscrete(game, key):
    env = gym.make('MiniDiscreteEnv-v0')
    env.goal_pos = [3, -3]
    env.borders = [7, -7]
    env.max_steps = 100

    params = get_params()[key]
    params["env"] = env
    params["game"] = game
    params["key"] = key

    start_time = time.time()
    episodes = [alphazero.run(env, params, i) for i in range(10)]
    print(episodes)
    print("Minutes:", (time.time() - start_time) / 60.)


# TODO This should not be global, but it's needed for multiprocessing.
class Action(Enum):
    left = 0
    right = 1
    forward = 2


class EnvWrapper:
    def __init__(self, env, params, actions, pomdp):
        self.env = env
        self.params = params
        self.actions = actions
        self.pomdp = pomdp

    def step(self, action):
        action = self.actions[action]
        obs, reward, done, _ = self.env.step(action)
        if self.pomdp:
            # TODO This is partially observable. Either return history or
            # addtionally the global state.
            return (
                obs['image'].flatten(),
                reward ** self.params["reward_exponent"],
                done,
                None
            )
        else:
            return (
                np.append(self.env.agent_pos, self.env.agent_dir),
                reward ** self.params["reward_exponent"],
                done,
                None
            )

    def reset(self):
        obs = self.env.reset()
        if self.pomdp:
            return obs['image'].flatten()
        else:
            return np.append(self.env.agent_pos, self.env.agent_dir)


def prepare_minigrid(game, params, pomdp):
    env = gym.make('MiniGrid-Empty-%s-v0' % game)

    # Monkey patch actions, step and reset.
    # TODO Seems a bit hacky.
    actions = {
        0: Action.left,
        1: Action.right,
        2: Action.forward
    }
    env.actions = Action
    return EnvWrapper(env, params, actions, pomdp)


def simulate_many_minigrid(game, key, pomdp=False, n_runs=10):
    start_time = time.time()

    # Load params and run AlphaZero.
    params = get_params()[key]
    params["game"] = game
    params["key"] = key
    params["n_actions"] = 3
    params["n_input_features"] = 3
    if pomdp:
        params["n_input_features"] = 147
    env = prepare_minigrid(game, params, pomdp)
    params["env"] = env

    writer = SummaryWriter(comment="-%s.%s" % (game, key))
    for i in range(n_runs):
        episodes, last_eval_len, last_tot_reward = alphazero.run(
            env, params, i, writer
        )
        writer.add_scalar('Summary/Length_All', last_eval_len, i)
        writer.add_scalar('Summary/Episodes_All', episodes, i)
        writer.add_scalar('Summary/Rewards_All', last_tot_reward, i)
    print("Minutes:", (time.time() - start_time) / 60.)


class EnvWrapperMtCar:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.actions = env.actions

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        reward **= self.params["reward_exponent"]
        return obs[0], reward, done, None

    def reset(self):
        obs = self.env.reset()
        return obs[0]


def simulate_many_mtcar(game, key, n_runs=10):
    env = gym.make('MountainCar-v0')
    env.actions = [0, 1, 2]

    start_time = time.time()

    # Load params and run AlphaZero.
    params = get_params()[key]
    params["game"] = game
    params["key"] = key
    params["n_actions"] = 3
    params["n_input_features"] = 2
    params["env"] = EnvWrapperMtCar(env, params)

    writer = SummaryWriter(comment="-%s.%s" % (game, key))
    for i in range(n_runs):
        episodes, last_eval_len, last_tot_reward = alphazero.run(
            env, params, i, writer
        )
        writer.add_scalar('Summary/Length_All', last_eval_len, i)
        writer.add_scalar('Summary/Episodes_All', episodes, i)
        writer.add_scalar('Summary/Rewards_All', last_tot_reward, i)
    print("Minutes:", (time.time() - start_time) / 60.)


if __name__ == '__main__':
    set_start_method("spawn")

    games = {
        'minidiscrete': (simulate_many_minidiscrete,),
        '5x5': (simulate_many_minigrid,),
        '8x8': (simulate_many_minigrid,),
        '16x16': (simulate_many_minigrid,),
        '5x5_pomdp': (
            functools.partial(simulate_many_minigrid, pomdp=True),
            '5x5'
        ),
        'mtcar': (simulate_many_mtcar,)
    }
    game = sys.argv[1]
    key = sys.argv[2]
    game_func = games[game]
    if len(game_func) > 1:
        game = game_func[1]
    print('Playing game', game, 'using func', game_func, '.')
    game_func[0](game, key)
