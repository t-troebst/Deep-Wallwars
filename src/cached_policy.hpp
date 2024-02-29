#pragma once

#include <folly/Synchronized.h>
#include <folly/container/EvictingCacheMap.h>

#include <atomic>
#include <thread>

#include "mcts.hpp"

class CachedPolicy {
public:
    CachedPolicy(EvaluationFunction evaluate, std::size_t capacity,
                 std::size_t shards = std::thread::hardware_concurrency());

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn,
                                             std::optional<Cell> previous_position);

    int cache_hits() const;
    int cache_misses() const;

private:
    struct EvaluationCache {
        EvaluationFunction evaluate;
        std::vector<folly::Synchronized<folly::EvictingCacheMap<std::uint64_t, Evaluation>>> shards;

        std::atomic<int> cache_hits = 0;
        std::atomic<int> cache_misses = 0;
    };

    std::shared_ptr<EvaluationCache> m_cache;
};
