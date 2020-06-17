#ifndef GRIDWORLD_HEADER
#define GRIDWORLD_HEADER
#include <tuple>
#include <utility>
#include <vector>
#include <set>
#include <random>


// TODO Polymorphism. Define step, reset
class MtCarEnv {
  public:
    // Constants
    float min_position = -1.2;
    float max_position = 0.6;
    float max_speed = 0.07 ;
    float goal_position = 0.5;
    float force = 0.001;
    float gravity = 0.0025;
    int max_steps;

    // Variable
    float goal_velocity;

    // State
    int steps;
    std::vector<float> state; // contains position, velocity.

    // For A2C. Just needs to be rough.
    std::vector<float> expected_mean;
    std::vector<float> expected_stddev;

    // Due to predictably small size no need for hashing (unordered_set)
    std::set<std::pair<int, int>> blocks;

    std::mt19937 generator;

    MtCarEnv() {};
    MtCarEnv(float goal_velocity);
    ~MtCarEnv() {};

    std::vector<float> reset();
    std::tuple<std::vector<float>, double, bool> step(int action);
};
#endif
