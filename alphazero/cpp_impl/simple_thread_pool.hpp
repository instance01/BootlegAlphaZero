#ifndef THREAD_POOL_HEADER
#define THREAD_POOL_HEADER
#include <mutex>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>
#include <utility>
#include "game.hpp"


// Semaphore copy-pasted from:
// https://stackoverflow.com/questions/4792449/c0x-has-no-semaphores-how-to-synchronize-threads
// Since I don't want to use C++20 yet.
class Semaphore {
  private:
    std::mutex mtx;
    std::condition_variable cv;

  public:
    int count;

    Semaphore (int count_ = 0) : count(count_) {};
    inline void notify();
    inline void wait();
};


class Task {
  public:
    std::function<std::shared_ptr<Game>()> func;
    std::shared_ptr<Game> result;
    bool finished;

    template<typename Function>
    Task(Function && function) : func(function) { };
    ~Task() {};
};


class SimpleThreadPool {
  private:
    std::vector<std::thread> threads;
    std::queue<Task*> task_queue;
    std::vector<Task*> tasks;
    std::mutex queue_mutex;
    std::condition_variable pool_notifier;
    bool should_stop_processing;
    Semaphore *semaphore;

  public:
    SimpleThreadPool(const std::size_t thread_count);
    ~SimpleThreadPool();

    void add_task(Task *task);
    void worker();
    std::vector<std::shared_ptr<Game>> join();
};
#endif
