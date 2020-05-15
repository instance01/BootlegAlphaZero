#!/usr/local/bin/python3
import main
import cProfile


main.params["episodes"] = 1
main.params["n_actors"] = 10
main.params["train_steps"] = 100


def run():
    main.alphazero.run(main.env, main.params)


cProfile.runctx("run()", globals(), locals(), "profiles/2.profile")
