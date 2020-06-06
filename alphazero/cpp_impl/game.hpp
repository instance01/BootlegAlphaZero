#ifndef GAME_HEADER
#define GAME_HEADER
#include <vector>

class Game {
  public:
    std::vector<std::vector<double>> states;
    std::vector<double> rewards;
    std::vector<std::vector<double>> mcts_actions;

    Game(
      std::vector<std::vector<double>> states,
      std::vector<double> rewards,
      std::vector<std::vector<double>> mcts_actions
    ) : states(states), rewards(rewards), mcts_actions(mcts_actions) {};
    ~Game() {};
};
#endif
