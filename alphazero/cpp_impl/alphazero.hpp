#ifndef ALPHAZERO_HEADER
#define ALPHAZERO_HEADER
#include "env_wrapper.hpp"
#include "cfg.hpp"
#include "a2c.hpp"
#include "mcts.hpp"

std::pair<int, double> evaluate(EnvWrapper env, json params, A2CLearner a2c_agent);
std::shared_ptr<Game> run_actor(EnvWrapper env, json params, MCTS mcts_agent, A2CLearner a2c_agent);
#endif
