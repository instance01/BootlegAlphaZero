#include <mutex>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>
#include "simple_thread_pool.hpp"
#include "mcts.hpp"

#include <mutex>
#include <condition_variable>


inline void Semaphore::notify()
{
  std::unique_lock<std::mutex> lock(mtx);
  count++;
  cv.notify_one();
}

inline void Semaphore::wait()
{
  std::unique_lock<std::mutex> lock(mtx);

  while (count == 0) {
      cv.wait(lock);
  }
  count--;
}

SimpleThreadPool::SimpleThreadPool(const std::size_t thread_count) : should_stop_processing(false) {
  threads.reserve(thread_count);
  for (std::size_t i = 0; i < thread_count; ++i)
    threads.emplace_back([this](){ worker(); });
  semaphore = new Semaphore(0);
}

SimpleThreadPool::~SimpleThreadPool() {
  {
    std::unique_lock<std::mutex> queue_lock(queue_mutex);
    should_stop_processing = true;
  }

  pool_notifier.notify_all();

  for (auto & task_thread: threads)
    task_thread.join();

  delete semaphore;
}

void SimpleThreadPool::add_task(Task *task) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    task_queue.emplace(task);
    tasks.push_back(task);
    semaphore->count += 1;
  }

  pool_notifier.notify_one();
}

std::vector<std::shared_ptr<Game>> SimpleThreadPool::join() {
  for (int i = 0; i < semaphore->count; ++i) {
    semaphore->wait();
  }

  std::vector<std::shared_ptr<Game>> ret;
  for (Task *t : tasks) {
    ret.push_back(t->result);
  }
  return ret;
}

void SimpleThreadPool::worker() {
  while (true) {
    Task *task;

    {
      std::unique_lock<std::mutex> queue_lock(queue_mutex);

      pool_notifier.wait(
          queue_lock,
          [this]() { return !task_queue.empty() || should_stop_processing; }
      );

      if (task_queue.empty() && should_stop_processing)
        return;

      task = task_queue.front();
      task_queue.pop();
      semaphore->notify();
    }

    std::shared_ptr<Game> result = task->func();
    task->result = result;
    task->finished = true;
  }
}
