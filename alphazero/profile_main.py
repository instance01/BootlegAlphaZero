#!/usr/local/bin/python3
import cProfile
from simulate import simulate_many_minigrid


def run():
    # main.alphazero.run(main.env, main.params, 0, 0)
    # simulate_many_minigrid('5x5', 'PROF', n_runs=1)
    simulate_many_minigrid('16x16', 'PROF', n_runs=1)


# run()
cProfile.runctx("run()", globals(), locals(), "profiles/9.profile")
