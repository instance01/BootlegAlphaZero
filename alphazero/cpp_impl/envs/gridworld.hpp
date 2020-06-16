#ifndef GRIDWORLD_HEADER
#define GRIDWORLD_HEADER
#include <tuple>
#include <utility>
#include <vector>
#include <set>


// TODO Polymorphism. Define step, reset
class GridWorldEnv {
  public:
    std::pair<int, int> start;
    std::pair<int, int> goal;
    std::pair<int, int> pos;
    int dir;

    int width;
    int height;
    int max_steps;
    int steps;

    // Due to predictably small size no need for hashing (unordered_set)
    std::set<std::pair<int, int>> blocks;

    GridWorldEnv() {};
    GridWorldEnv(int width, int height, std::set<std::pair<int, int>> blocks);
    ~GridWorldEnv() {};

    void move(int action);
    std::vector<int> reset();
    std::tuple<std::vector<int>, double, bool> step(int action);
};
#endif
