#include <iostream>
#include <algorithm>
#include <torch/torch.h>
#include "tensorboard_logger.h"
#include "alphazero.hpp"
#include "replay_buffer.hpp"
#include "tensorboard_util.hpp"
#include "simple_thread_pool.hpp"
#include "cfg.hpp"
#include "lr_scheduler.hpp"


std::pair<int, double> evaluate(EnvWrapper env, json params, A2CLearner a2c_agent) {
  env = *env.clone();

  std::vector<float> state = env.reset();

  bool done = false;
  double total_reward = 0.;
  std::string actions = "";

  while (!done) {
    torch::Tensor action_probs;
    torch::Tensor val;
    std::tie(action_probs, val) = a2c_agent.predict_policy({state});

    int action = action_probs.argmax().item<int>();

    // TODO Remove. This prints current policy and value in compact way.
    //float* xx = (float*)action_probs.data_ptr();
    //for (int i = 0; i < action_probs.sizes()[1]; ++i)
    //  std::cout << std::ceil(xx[i] * 100.0) / 100.0 << " ";
    //std::cout << "(" << std::ceil(val.item<float>() * 100.0) / 100.0 << ") # ";

    double reward;
    std::tie(state, reward, done) = env.step(action);
    total_reward += reward;
    actions += std::to_string(action);
  }
  //std::cout << std::endl;

  std::cout << "EVAL " << actions << " " << total_reward << std::endl;
  return std::make_pair(actions.length(), total_reward);
}

std::shared_ptr<Game> run_actor(EnvWrapper orig_env, json params, MCTS mcts_agent, A2CLearner a2c_agent) {
  EnvWrapper env = *orig_env.clone();
  auto state = env.reset();

  // TODO Make sure this is a good idea to init here.
  std::random_device rd;
  std::mt19937 generator(rd());

  int horizon = params["horizon"];

  std::string mcts_actions = "";

  bool done = false;
  std::shared_ptr<Game> game = std::make_shared<Game>();
  for (int i = 0; i < horizon; ++i) {
    auto mcts_action = mcts_agent.policy(env, state);

    std::discrete_distribution<int> distribution(mcts_action.begin(), mcts_action.end());
    int sampled_action = distribution(generator);
    mcts_actions += std::to_string(sampled_action);

    torch::Tensor action_probs;
    std::tie(action_probs, std::ignore) = a2c_agent.predict_policy({state});

    std::vector<float> next_state;
    double reward;
    std::tie(next_state, reward, done) = env.step(sampled_action);

    game->states.push_back(state);
    game->rewards.push_back(reward);
    game->mcts_actions.push_back(mcts_action);

    state = next_state;

    if (done)
      break;
  }

  std::cout << mcts_actions.size() << " " << std::flush;
  return game;
}

// TODO Move median und mean to util.

template<typename T>
T median(std::vector<T> &v)
{
  size_t n = v.size() / 2;
  nth_element(v.begin(), v.begin()+n, v.end());
  return v[n];
}

template<typename T>
T mean(std::vector<T> &v)
{
  return std::accumulate(v.begin(), v.end(), T{} * 0) / v.size();
}

std::pair<int, double> episode(
  TensorBoardLogger &writer,
  int n_run,
  EnvWrapper env,
  MCTS mcts_agent,
  A2CLearner a2c_agent,
  int n_episode,
  ReplayBuffer replay_buffer,
  json params,
  LRScheduler *lr_scheduler,
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time
) {
  a2c_agent.policy_net->eval();
  int n_actors = params["n_actors"];
  int train_steps = params["train_steps"];
  int n_procs = params["n_procs"];

  mcts_agent.reset_policy_cache();

  std::vector<int> actor_lengths;

  // Run self play games in n_procs parallel processes.
  auto pool = SimpleThreadPool(n_procs);
  auto lambda = [env, params, mcts_agent, a2c_agent]() -> std::shared_ptr<Game> {
    return run_actor(env, params, mcts_agent, a2c_agent);
  };
  for (int i = 0; i < n_actors; ++i) {
    Task *task = new Task(lambda);
    pool.add_task(task);
  }
  std::vector<std::shared_ptr<Game>> games = pool.join();
  std::cout << std::endl << "# Games: " << games.size() << std::endl;

  for(auto game : games) {
    actor_lengths.push_back(game->states.size());
    replay_buffer.add(std::move(game));
  }
  // TODO After this the games vector is unusable. We've moved it all (std::move).

  // Print debug information.
  std::cout << "REWARDS ";
  auto rewards = replay_buffer.get_rewards();
  int size_ = rewards.size() - n_actors - 1;
  for (int i = rewards.size() - 1; i > size_; --i) {
    std::cout << rewards[i] << " ";
  }
  std::cout << std::endl;

  std::vector<double> max_action_probs;
  size_ = replay_buffer.buffer.size() - n_actors - 1;
  for (int i = replay_buffer.buffer.size() - 1; i > size_; --i) {
    auto game = replay_buffer.buffer[i];
    for (auto mcts_action : game->mcts_actions) {
      auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
      max_action_probs.push_back(*max_el);
      //std::cout << *max_el << " ";
    }
  }
  //std::cout << std::endl;
  auto mcts_confidence_mean = mean<double>(max_action_probs);
  auto mcts_confidence_median = median<double>(max_action_probs);
  std::cout << "CONFIDENCE " << mcts_confidence_mean << " " << mcts_confidence_median << std::endl;
  writer.add_scalar("Eval/MCTS_Confidence/" + std::to_string(n_run), n_episode, (float) mcts_confidence_median);

  // Train network after self play.
  std::vector<int> sample_lens;
  std::vector<double> losses;

  a2c_agent.policy_net->train();
  for (int i = 0; i < train_steps; ++i) {
    auto game = replay_buffer.sample();
    auto loss = a2c_agent.update(game);
    std::string actions;
    for (auto mcts_action : game->mcts_actions) {
      auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
      actions += std::to_string(std::distance(mcts_action.begin(), max_el));
    }
    if (i % 10 == 0) {
      std::cout << "." << std::flush;
    }
    sample_lens.push_back(actions.size());
    losses.push_back(loss.item<double>());
  }

  std::cout << std::endl;

  a2c_agent.policy_net->eval();

  auto curr_time = std::chrono::high_resolution_clock::now();

  double avg_loss = mean<double>(losses);
  std::cout <<
      "AVG LENS " << mean<int>(sample_lens) <<
      " |AVG LOSS " << avg_loss <<
      " |TIME (min) " << std::chrono::duration_cast<std::chrono::minutes>(curr_time - start_time).count() <<
      std::endl;
  int eval_length;
  double total_reward;
  std::tie(eval_length, total_reward) = evaluate(env, params, a2c_agent);

  // The decision to only use the first param group might be dubious.
  // Keep that in mind. For now it is fine, I checked.
  auto& options = static_cast<torch::optim::AdamOptions&>(
      a2c_agent.policy_optimizer->param_groups()[0].options()
  );
  auto lr = options.lr();
  if (params["schedule_alpha"]) {
    options.lr(lr_scheduler->step(lr, n_episode, total_reward));
  }

  writer.add_scalar("Eval/Length/" + std::to_string(n_run), n_episode, (float) eval_length);
  writer.add_scalar("Eval/Reward/" + std::to_string(n_run), n_episode, total_reward);
  writer.add_histogram(
      "Actor/Sample_length/" + std::to_string(n_run), n_episode, actor_lengths
  );
  writer.add_histogram(
      "Train/Samples/" + std::to_string(n_run), n_episode, sample_lens
  );
  writer.add_histogram(
      "Train/Loss/" + std::to_string(n_run), n_episode, losses
  );

  std::cout << "LR " << lr << std::endl;

  writer.add_scalar("Train/LearningRate/" + std::to_string(n_run), n_episode, lr);
  writer.add_scalar("Train/AvgLoss/" + std::to_string(n_run), n_episode, avg_loss);

  std::cout << std::endl;

  return {eval_length, total_reward};
}

std::tuple<int, int, double> run(EnvWrapper env, json params, int n_run, TensorBoardLogger &writer) {
  std::ostringstream oss;
  oss << params.dump();
  writer.add_text("Info/params/" + std::to_string(n_run), 0, oss.str().c_str());

  auto start_time = std::chrono::high_resolution_clock::now();
  ReplayBuffer replay_buffer(
    params["memory_capacity"],
    params["prioritized_sampling"]
  );
  auto a2c_agent = A2CLearner(params, env);
  auto mcts_agent = MCTS(env, a2c_agent, params);
  LRScheduler *lr_scheduler;
  if (params["scheduler_class"] == "exp") {
    lr_scheduler = new ExponentialScheduler(
        params["scheduler_factor"],
        params["scheduler_min_lr"]
    );
  } else if (params["scheduler_class"] == "step") {
    lr_scheduler = new StepScheduler(
        params["scheduler_steps"],
        params["scheduler_factor"],
        params["scheduler_min_lr"]
    );
  } else if (params["scheduler_class"] == "reduce_eval") {
    lr_scheduler = new ReduceOnGoodEval(
        params["scheduler_factor"],
        params["scheduler_min_good_eval"],
        params["scheduler_min_n_good_evals"],
        params["scheduler_min_lr"]
    );
  }

  // Need to have less than or equal desired evaluation length, certain times in a row.
  int is_done_stably = 0;
  int desired_eval_len = params["desired_eval_len"];
  int n_desired_eval_len = params["n_desired_eval_len"];

  int eval_len;
  double total_reward;
  int i = 0;
  for (; i < params["episodes"]; ++i) {
    std::cout << "Episode " << i << std::endl;
    std::tie(eval_len, total_reward) = episode(
        writer,
        n_run,
        env,
        mcts_agent,
        a2c_agent,
        i,
        replay_buffer,
        params,
        lr_scheduler,
        start_time
    );

    if (eval_len <= desired_eval_len) {
        is_done_stably += 1;
    } else {
        is_done_stably = 0;
    }

    if (is_done_stably > n_desired_eval_len)
        break;
  }
  delete lr_scheduler;
  return std::make_tuple(i, eval_len, total_reward);
}

int main(int argc, char* argv[]) {
  std::string game(argv[1]);
  std::string param_num(argv[2]);
  std::cout << "Running with parameters: " << game << " " << param_num << std::endl;

  // TODO This is important. Maybe there's more such improvements?
  at::init_num_threads();

  auto params = load_cfg(param_num);

  EnvWrapper env = EnvWrapper();
  env.init(game, params);

  TensorBoardLogger writer(gen_log_filename(game, param_num).c_str());

  for (int i = 0; i < params["n_runs"]; ++i)
    run(env, params, i, writer);
}
