#include "mcts.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include "util.hpp"


MCTS::MCTS(Env env, A2CLearner a2c_agent, json params) : params(params), a2c_agent(a2c_agent) {
  auto obs = env.reset();
  this->env = std::move(env.clone());

  this->root_node = std::make_shared<Node>();
  root_node->env = std::move(env.clone());
  root_node->state = obs;
}

double
MCTS::_ucb(std::shared_ptr<Node> parent_node, std::shared_ptr<Node> child_node, torch::Tensor action_probs) {
  double mean_q = child_node->Q / child_node->visits;
  double base = params["pb_c_base"];
  double pb_c_init = params["pb_c_init"];

  double pb_c = std::log((parent_node->visits + base + 1) / base) + pb_c_init;
  pb_c *= std::sqrt(parent_node->visits) / (child_node->visits + 1);

  double prior_score = pb_c * action_probs[0][child_node->action].item<double>();
  return mean_q + prior_score;
}

void
MCTS::_gen_children_nodes(std::shared_ptr<Node> parent_node) {
  int n_actions = params["n_actions"];
  for (int i = 0; i < n_actions; ++i) {
    auto env = this->env->clone();

    std::vector<int> obs;
    double reward;
    bool done;
    std::tie(obs, reward, done) = env->step(i);

    std::shared_ptr<Node> node = std::make_unique<Node>();
    node->state = obs;
    node->env = std::move(env);
    node->action = i;
    node->reward = reward;
    node->is_terminal = done;
    node->parent = std::weak_ptr(parent_node);

    node->torch_state = vec_1d_as_tensor(node->state, torch::kInt);

    parent_node->children.push_back(node);
  }
}

std::shared_ptr<Node>
MCTS::_expand(std::shared_ptr<Node> parent_node) {
  if (parent_node->children.empty())
    _gen_children_nodes(parent_node);

  for (auto child : parent_node->children) {
    if (child->visits == 0) {
      return child;
    }
  }

  parent_node->is_fully_expanded = true;
  return nullptr;
}

std::shared_ptr<Node>
MCTS::_get_best_node(std::shared_ptr<Node> parent_node) {
  // TODO Test whether prediction (forwards) need to be really cached.
  // In Python they do, since it's so slow.
  //
  // Hashing of a vector<double> is not trivial.
  // I opted for a red-black tree (map) for now.
  // TODO: Reconsider.
  auto cached_action_probs = policy_net_cache.find(parent_node->state);
  torch::Tensor action_probs;
  if (cached_action_probs == policy_net_cache.end()) {
    std::tie(action_probs, std::ignore) = a2c_agent.predict_policy(parent_node->torch_state);
    policy_net_cache[parent_node->state] = action_probs;
  } else {
    action_probs = cached_action_probs->second;
  }

  std::vector<double> ucb_vals;
  for (auto child : parent_node->children) {
    ucb_vals.push_back(
      _ucb(parent_node, child, action_probs)
    );
  }

  double max_ucb = ucb_vals[std::distance(
      ucb_vals.begin(),
      std::max_element(ucb_vals.begin(), ucb_vals.end())
  )];

  // TODO There must be a better way.

  std::vector<int> idx;
  for (size_t i = 0; i < ucb_vals.size(); ++i) {
    if (ucb_vals[i] >= max_ucb)
      idx.push_back(i);
  }
  if (idx.size() == 0) {
    std::cout << "Aborting after idx." << std::endl;
    abort();
  }
  return parent_node->children[idx[std::rand() % idx.size()]];
}

std::shared_ptr<Node>
MCTS::select_expand() {
  std::shared_ptr<Node> curr_node = root_node;

  while (true) {
    if (curr_node->is_terminal) {
      break;
    }

    if (curr_node->is_fully_expanded) {
      curr_node = _get_best_node(curr_node);
    } else {
      auto node = _expand(curr_node);
      if (node != nullptr)
        return node;
    }
  }

  return curr_node;
}

void
MCTS::backup(std::shared_ptr<Node> curr_node, double Q_val) {
  while (curr_node) {
    curr_node->Q += Q_val;
    curr_node->visits += 1;
    auto parent = curr_node->parent.lock();
    curr_node = parent;
  }
}

void
MCTS::reset_policy_cache() {
  policy_net_cache.clear();
}

std::vector<double>
MCTS::policy(Env env, std::vector<int> obs, bool ret_node) {
  int n_actions = params["n_actions"];
  int n_iter = params["simulations"];

  this->env = std::move(env.clone());
  root_node = std::make_shared<Node>();
  root_node->env = std::move(env.clone());
  root_node->state = obs;

  root_node->torch_state = vec_1d_as_tensor(root_node->state, torch::kInt);

  torch::Tensor action_probs;
  std::tie(action_probs, std::ignore) = a2c_agent.predict_policy(root_node->torch_state);
  double alpha = params["dirichlet_alpha"];
  double frac = params["dirichlet_frac"];

  std::random_device rd;
  std::mt19937 generator(rd());
  std::gamma_distribution<double> distribution(alpha, 1.);
  for (int i = 0; i < n_actions; ++i) {
    double noise = distribution(generator);
    action_probs[0][i] = action_probs[0][i] * (1 - frac) + noise * frac;
  }

  policy_net_cache[root_node->state] = action_probs;

  for (int i = 0; i < n_iter; ++i) {
    std::shared_ptr<Node> node = select_expand();
    torch::Tensor value;
    std::tie(std::ignore, value) = a2c_agent.predict_policy(node->torch_state);
    double q_val = value[0][0].item<double>();
    // Comment from Python: 'If gradients explode, q_val will be nan.'
    // TODO Update: Not so sure any longer whether that comment was a correct assessment.
    backup(node, q_val);
  }

  std::vector<double> ret;
  for (auto child : root_node->children) {
    ret.push_back(1.0 * child->visits / n_iter);
  }

  return ret;
}
