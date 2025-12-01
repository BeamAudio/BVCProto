#pragma once
#include <mutex>
#include <condition_variable>
#include <queue>
#include <optional>

namespace BVC {
    template<typename T>
    class BoundedQueue {
    private:
        std::queue<T> queue_;
        mutable std::mutex mutex_;
        std::condition_variable not_empty_;
        std::condition_variable not_full_;
        size_t capacity_;
        bool finished_ = false;

    public:
        explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

        void push(T item) {
            std::unique_lock<std::mutex> lock(mutex_);
            not_full_.wait(lock, [this] { return queue_.size() < capacity_ || finished_; });
            if (finished_) return; // Reject push if closed
            queue_.push(std::move(item));
            not_empty_.notify_one();
        }

        std::optional<T> pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });
            
            if (queue_.empty()) return std::nullopt; // Finished
            
            T item = std::move(queue_.front());
            queue_.pop();
            not_full_.notify_one();
            return item;
        }

        void finish() {
            std::unique_lock<std::mutex> lock(mutex_);
            finished_ = true;
            not_empty_.notify_all();
            not_full_.notify_all();
        }
    };
}
