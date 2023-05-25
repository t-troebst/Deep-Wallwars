#pragma once

#include <folly/experimental/coro/SharedMutex.h>
#include <folly/concurrency/ConcurrentHashMap.h>

#include <unordered_map>
#include <atomic>

#include "mcts.hpp"

class CachedPolicy : public MCTSPolicy {
public:
    CachedPolicy(std::shared_ptr<MCTSPolicy> policy);

    folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn,
                                                    TreeNode const* parent) override;
    void snapshot(TreeNode const& current_root) override;

    int cache_hits() const;
    int cache_misses() const;

private:
    std::shared_ptr<MCTSPolicy> m_policy;
    // We don't care about collisions, if there are any - so be it :)
    // TODO: use some kind of LRU cache instead? hopefully we don't run out of memory...
    folly::ConcurrentHashMap<std::uint64_t, Evaluation> m_cache;
    std::atomic<int> m_cache_hits = 0;
    std::atomic<int> m_cache_misses = 0;
};
