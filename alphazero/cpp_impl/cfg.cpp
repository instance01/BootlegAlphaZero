#include <iostream>
#include <fstream>
#include "cfg.hpp"

using json = nlohmann::json;


json get_default() {
  json params1 = {
    {"n_actions", 3},
    {"n_input_features", 3},

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

    // A2C
    {"alpha", .01},
    {"net_architecture", {64, 64}},
    {"schedule_alpha", false},
    {"scheduler_class", "ExponentialLR"},
    {"scheduler_args", {.995}},
    {"scheduler_kwargs", {}},

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
  };

  return params1;
}


json load_cfg(std::string param_num) {
  std::ifstream ifs("simulations.json");
  json jsondata = json::parse(ifs);
  json ret = get_default();
  ret.update(jsondata[param_num]);
  std::cout << ret.dump(2) << std::endl;
  return ret;
}
