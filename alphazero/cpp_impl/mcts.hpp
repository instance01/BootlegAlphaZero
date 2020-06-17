#ifndef MCTS_HEADER
#define MCTS_HEADER
#include <cstdlib>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "env.hpp"
#include "a2c.hpp"
#include "cfg.hpp"


class Node {
  public:
    long _id;
    bool is_fully_expanded = false;
    bool is_terminal = false;
    double reward = 0.;
    std::weak_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children;
    int action;
    double Q = 0.;
    int visits = 0;
    std::vector<int> state;
    torch::Tensor torch_state;
    std::shared_ptr<Env> env;

    // TODO Increase that?
    Node() : _id(std::rand() % 2147483648) {};
    ~Node() {};

    bool operator==(Node& other) {
      return _id == other._id;
    }
};

class MCTS {
  public:
    json params;
    A2CLearner a2c_agent;
    // shared_ptr because unique_ptr makes this class uncopyable.
    // And I don't want to define a custom copy function.
    std::shared_ptr<Env> env;
    std::shared_ptr<Node> root_node;

    std::map<std::vector<int>, torch::Tensor> policy_net_cache;

    MCTS(Env env, A2CLearner a2c_agent, json params);
    MCTS() {};
    ~MCTS() {};

    double _ucb(std::shared_ptr<Node> parent_node, std::shared_ptr<Node> child_node, torch::Tensor action_probs);
    void _gen_children_nodes(std::shared_ptr<Node> parent_node);
    std::shared_ptr<Node> _expand(std::shared_ptr<Node> parent_node);
    std::shared_ptr<Node> _get_best_node(std::shared_ptr<Node> parent_node);
    std::shared_ptr<Node> select_expand();
    void backup(std::shared_ptr<Node> curr_node, double Q_val);
    void reset_policy_cache();
    std::vector<double> policy(Env env, std::vector<int> obs, bool ret_node=false);
};
#endif
