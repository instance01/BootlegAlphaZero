#ifndef MTCAR_HEADER
#define MTCAR_HEADER
#include <tuple>
#include <utility>
#include <vector>
#include <set>
#include <random>
#include "env.hpp"


class MtCarEnv : public Env {
  public:
    // Constants
    float min_position = -1.2;
    float max_position = 0.6;
    float max_speed = 0.07 ;
    float goal_position = 0.5;
    float force = 0.001;
    float gravity = 0.0025;
    int max_steps = 200;

    // Variable
    float goal_velocity = 0;

    // State
    int steps = 0;
    std::vector<float> state; // contains position, velocity.

    std::mt19937 generator;

    MtCarEnv() {};
    MtCarEnv(MtCarEnv &other);
    MtCarEnv(float goal_velocity);
    ~MtCarEnv() {};

    std::vector<float> reset();
    std::tuple<std::vector<float>, double, bool> step(int action);
};
#endif
