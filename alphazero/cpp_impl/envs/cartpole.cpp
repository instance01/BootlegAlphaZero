#include "cartpole.hpp"


// Original source from Python.
// https://github.com/openai/gym

CartPoleEnv::CartPoleEnv() {
  std::random_device rd;
  generator = std::mt19937(rd());

  reset();

  // State is already normalized well enough..
  // TODO: stddev=2 for velocities is pulled out of my ass..
  expected_mean = {0., 0.};
  expected_stddev = {5., 2., 1., 2.};
}

CartPoleEnv::CartPoleEnv(CartPoleEnv &other) {
  max_steps = other.max_steps;
  steps = other.steps;
  state = other.state;
  generator = other.generator;
}

std::tuple<std::vector<float>, double, bool>
CartPoleEnv::step(int action) {
  steps += 1;
  if (steps >= max_steps)
    return {state, 0., true};

  float x = state[0];
  float x_dot = state[1];
  float theta = state[2];
  float theta_dot = state[3];
  float force = -force_mag;
  if (action == 1)
    force = force_mag;
  float costheta = cos(theta);
  float sintheta = sin(theta);

  float temp = (force + polemass_length * pow(theta_dot, 2) * sintheta) / total_mass;
  float thetaacc = (gravity * sintheta - costheta * temp) / (length * (4. / 3. - masspole * pow(costheta, 2) / total_mass));
  float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

  x = x + tau * x_dot;
  x_dot = x_dot + tau * xacc;
  theta = theta + tau * theta_dot;
  theta_dot = theta_dot + tau * thetaacc;

  state = {
    x, x_dot, theta, theta_dot
  };

  bool done = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;

  float reward = 1.;
  return {state, reward, done};
}

std::vector<float>
CartPoleEnv::reset() {
  steps = 0;
  std::uniform_real_distribution<float> distribution(-.05, .05);
  state = {
    distribution(generator),
    distribution(generator),
    distribution(generator),
    distribution(generator)
  };
  return state;
}
