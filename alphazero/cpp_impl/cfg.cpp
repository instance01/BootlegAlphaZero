#include <iostream>
#include <fstream>
#include "cfg.hpp"

using json = nlohmann::json;


json get_default(std::string base) {
  json params = {
    {"1", {
      {"n_actions", 3},
      {"n_input_features", 3},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"c", 1.},  // .001  # Using puct now.
      {"simulations", 50},
      {"horizon", 200},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .25},
      // TODO I think this should be based on simulations.
      // base was 19652. But with just 100 simulations (and thus visits only
      // getting to 100 at max) visits don't matter.. At base=50}, hell yes !
      {"pb_c_base", 50},
      {"pb_c_init", 1.25},
      {"tough_ce", false},

      // A2C
      {"alpha", .01},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", false},
      {"scheduler_class", "exp"},  // exp, step, reduce_eval
      {"scheduler_factor", .995},
      {"scheduler_min_lr", .000001},
      {"scheduler_steps", {100, 200}},
      {"scheduler_min_good_eval", -120.0},
      {"scheduler_min_n_good_evals", 15},
      {"scheduler_consecutive", true},
      {"use_weight_decay", false},
      {"weight_decay", .00001},

      // AlphaZero
      {"memory_capacity", 1000},
      {"prioritized_sampling", true},
      {"episodes", 100},
      {"n_procs", 4},
      {"n_actors", 20},  // 5000
      {"train_steps", 2000},  // 700000
      {"desired_eval_len", 8},
      {"n_desired_eval_len", 10},

      // Other
      {"reward_exponent", 1},

      // TODO unused right now
      {"epsilon", .1},
      {"epsilon_linear_decay", 1. / 10000},  // 10000 is memory_capacity
      {"epsilon_min", 0.01}
    }},
    {"81", {
      {"n_actions", 3},
      {"n_input_features", 2},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"simulations", 50},
      {"horizon", 400},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .5},
      {"pb_c_base", 19000},
      {"pb_c_init", .3},
      {"tough_ce", true},

      // A2C
      {"alpha", .001},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", true},
      {"scheduler_class", "exp"},  // exp, step, reduce_eval
      {"scheduler_factor", .995},
      {"scheduler_min_lr", .000001},
      {"scheduler_steps", {100, 200}},
      {"scheduler_min_good_eval", -120.0},
      {"scheduler_min_n_good_evals", 20},
      {"scheduler_consecutive", true},
      {"use_weight_decay", false},
      {"weight_decay", .00001},

      // AlphaZero
      {"memory_capacity", 1000},
      {"prioritized_sampling", false},
      {"episodes", 500},
      {"n_procs", 8},
      {"n_actors", 40},  // 5000
      {"train_steps", 4000},  // 700000
      {"desired_eval_len", 8},
      {"n_desired_eval_len", 10},

      // Other
      {"reward_exponent", 1},
    }},
    {"120", {
      {"n_actions", 3},
      {"n_input_features", 2},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"simulations", 50},
      {"horizon", 400},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .5},
      {"pb_c_base", 1000},
      {"pb_c_init", .1},
      {"tough_ce", true},

      // A2C
      {"alpha", .001},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", true},
      {"scheduler_class", "reduce_eval"},  // exp, step, reduce_eval
      {"scheduler_factor", .5},
      {"scheduler_min_lr", .000001},
      {"scheduler_steps", {100, 200}},
      {"scheduler_min_good_eval", -100.0},
      {"scheduler_min_n_good_evals", 10},
      {"scheduler_consecutive", false},
      {"use_weight_decay", false},
      {"weight_decay", .00001},

      // AlphaZero
      {"memory_capacity", 1000},
      {"prioritized_sampling", false},
      {"episodes", 1000},
      {"n_procs", 8},
      {"n_actors", 40},
      {"train_steps", 4000},
      {"desired_eval_len", 110},
      {"n_desired_eval_len", 100},

      // Other
      {"reward_exponent", 1},
    }}
  };

  return params[base];
}


json load_cfg(std::string param_num) {
  std::ifstream ifs("simulations.json");
  json jsondata = json::parse(ifs);
  std::string base = jsondata[param_num]["base"];
  json ret = get_default(base);
  ret.update(jsondata[param_num]);
  std::cout << ret.dump(2) << std::endl;
  return ret;
}
