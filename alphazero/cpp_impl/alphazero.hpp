#ifndef ALPHAZERO_HEADER
#define ALPHAZERO_HEADER
#include "env.hpp"
#include "params.hpp"
#include "a2c.hpp"

std::pair<int, double> evaluate(Env env, Params params, A2CLearner a2c_agent);
#endif
