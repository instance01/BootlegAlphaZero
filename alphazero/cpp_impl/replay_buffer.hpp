#ifndef REPLAY_BUFFER_HEADER
#define REPLAY_BUFFER_HEADER
#include "game.hpp"

class ReplayBuffer {
  public:
    std::vector<Game> buffer;
    int window_size;
    bool prioritized_sampling = true;

    ReplayBuffer() {};
    ~ReplayBuffer() {};

    ReplayBuffer(
        int window_size, bool prioritized_sampling = true
    ) : window_size(window_size), prioritized_sampling(prioritized_sampling) {};


    void add(
        std::vector<std::vector<double>> states,
        std::vector<double> rewards,
        std::vector<std::vector<double>> mcts_actions
    );
    std::vector<double> get_rewards();
    Game sample();
};
#endif
