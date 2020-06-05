#include <numeric>
#include <random>

#include <iostream>

#include "game.hpp"
#include "replay_buffer.hpp"


void
ReplayBuffer::add(
    std::vector<std::vector<double>> states,
    std::vector<double> rewards,
    std::vector<std::vector<double>> mcts_actions
) {
  Game game(states, rewards, mcts_actions);
  buffer.push_back(game);
}

std::vector<double>
ReplayBuffer::get_rewards() {
  std::vector<double> tot_reward_per_game;
  for (Game game : buffer) {
    tot_reward_per_game.push_back(std::accumulate(game.rewards.begin(), game.rewards.end(), 0));
  }
  return tot_reward_per_game;
}

Game
ReplayBuffer::sample() {
  std::random_device rd;
  std::mt19937 generator(rd());
  auto rewards = get_rewards();
  std::discrete_distribution<double> distribution(rewards.begin(), rewards.end());
  int idx = distribution(generator);
  std::cout << "idx " << idx << std::endl;
  return buffer[idx];
}
