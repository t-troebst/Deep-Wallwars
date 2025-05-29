#pragma once

#include <folly/Synchronized.h>
#include <folly/ThreadLocal.h>
#include <folly/container/EvictingCacheMap.h>
#include <folly/container/HeterogeneousAccess.h>

#include <atomic>

#include "mcts.hpp"
#include "batched_model.hpp"
#include "batched_model_policy.hpp"

struct CacheEntry {
    Board board;
    Turn turn;
    std::optional<Cell> previous_position;
};

struct CacheEntryView {
    Board const& board;
    Turn const& turn;
    std::optional<Cell> const& previous_position;
};

template <>
struct folly::HeterogeneousAccessHash<CacheEntry> {
    using is_transparent = void;
    using folly_is_avalanching = std::true_type;

    std::uint64_t operator()(CacheEntry const& cache_entry) const;
    std::uint64_t operator()(CacheEntryView ce_view) const;
};

template <>
struct folly::HeterogeneousAccessEqualTo<CacheEntry> {
    using is_transparent = void;

    bool operator()(CacheEntry const& lhs, CacheEntry const& rhs) const;
    bool operator()(CacheEntry const& lhs, CacheEntryView rhs) const;
    bool operator()(CacheEntryView lhs, CacheEntry const& rhs) const;
};

class CachedPolicy {
public:
    CachedPolicy(EvaluationFunction evaluate, std::size_t capacity,
                 unsigned shards = std::thread::hardware_concurrency());

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn,
                                             std::optional<Cell> previous_position);

    int cache_hits() const;
    int cache_misses() const;

    // Returns a reference to the underlying policy
    EvaluationFunction const& underlying_policy() const {
        return m_cache->evaluate;
    }

private:
    using LRU = folly::EvictingCacheMap<CacheEntry, Evaluation>;

    struct EvaluationCache {
        EvaluationFunction evaluate;
        // LRUs are sharded instead of thread-local because computing an evaluation is a co-routine
        // that might hop threads.
        std::vector<folly::Synchronized<LRU>> lrus;

        std::atomic<int> cache_hits = 0;
        std::atomic<int> cache_misses = 0;
    };

    std::shared_ptr<EvaluationCache> m_cache;
};
