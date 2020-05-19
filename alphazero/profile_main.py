#!/usr/local/bin/python3
import main
import cProfile
from simulate import simulate_many_minigrid


main.params["episodes"] = 1
main.params["n_actors"] = 1
main.params["train_steps"] = 1000


def run():
    # main.alphazero.run(main.env, main.params, 0, 0)
    simulate_many_minigrid('5x5', 'PROF', n_runs=1)


# run()
cProfile.runctx("run()", globals(), locals(), "profiles/3.profile")
