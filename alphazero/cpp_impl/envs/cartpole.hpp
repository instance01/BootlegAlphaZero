#ifndef CARTPOLE_HEADER
#define CARTPOLE_HEADER
#include <tuple>
#include <utility>
#include <vector>
#include <set>
#include <random>
#include <cmath>
#include "env.hpp"


class CartPoleEnv : public Env {
  public:
    // Constants
    float gravity = 9.8;
    float masscart = 1.0;
    float masspole = 0.1;
    float total_mass = (masspole + masscart);
    float length = 0.5;
    float polemass_length = (masspole * length);
    float force_mag = 10.0;
    float tau = 0.02;
    float theta_threshold_radians = 12 * 2 * M_PI / 360;
    float x_threshold = 2.4;
    int max_steps = 200;

    // State
    int steps = 0;
    std::vector<float> state; // contains position, velocity, angle, angular velocity.

    std::mt19937 generator;

    CartPoleEnv();
    CartPoleEnv(CartPoleEnv &other);
    ~CartPoleEnv() {};

    std::vector<float> reset();
    std::tuple<std::vector<float>, double, bool> step(int action);
};
#endif
