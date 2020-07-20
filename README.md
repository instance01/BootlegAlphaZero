This is a bootleg version of AlphaZero written in C++. It also includes some work leading up to it such as MCTS, imitation learning and a Python implementation (which I stopped working with due to performance). I tried a kind of bridge whereas the C++ code is the main program which only calls gym routines in Python using the Python C API, but unfortunately that turned out to be too slow. Thus, I went for a pure C++ version and included a few environments from [openai/gym](https://github.com/openai/gym/) rewritten in C++ (see the envs folder).

Below is the result of training 10 times on MountainCar using bootleg AlphaZero with parameter configuration 127. More current configurations can be seen [here](https://github.com/instance01/BootlegAlphaZero/blob/master/alphazero/cpp_impl/results.md).
<img src=".github/cpp_mtcar_127.png" /> 

Quite the variance, and takes ages to learn (roughly 6 hours to be more precise). Needs more work.
