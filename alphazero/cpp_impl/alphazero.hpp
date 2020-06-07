#ifndef ALPHAZERO_HEADER
#define ALPHAZERO_HEADER
#include "env.hpp"
#include "params.hpp"
#include "a2c.hpp"
#include "mcts.hpp"

std::pair<int, double> evaluate(Env env, Params params, A2CLearner a2c_agent);
std::shared_ptr<Game> run_actor(Env env, Params params, MCTS mcts_agent, A2CLearner a2c_agent);
#endif
