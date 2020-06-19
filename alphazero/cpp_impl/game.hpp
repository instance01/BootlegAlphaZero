#ifndef GAME_HEADER
#define GAME_HEADER
#include <vector>

class Game {
  public:
    std::vector<std::vector<float>> states;
    std::vector<double> rewards;
    std::vector<std::vector<double>> mcts_actions;

    Game() {};
    Game(
      std::vector<std::vector<float>> states,
      std::vector<double> rewards,
      std::vector<std::vector<double>> mcts_actions
    ) : states(states), rewards(rewards), mcts_actions(mcts_actions) {};
    ~Game() {};
};
#endif
