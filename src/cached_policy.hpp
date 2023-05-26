#pragma once

#include <folly/Synchronized.h>
#include <folly/container/EvictingCacheMap.h>

#include <atomic>
#include <thread>

#include "mcts.hpp"

class CachedPolicy : public MCTSPolicy {
public:
    CachedPolicy(std::shared_ptr<MCTSPolicy> policy, std::size_t capacity,
                 std::size_t shards = std::thread::hardware_concurrency());

    folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn) override;
    void snapshot(NodeInfo const& node_info, std::optional<Player> winner) override;

    int cache_hits() const;
    int cache_misses() const;

private:
    std::shared_ptr<MCTSPolicy> m_policy;
    std::vector<folly::Synchronized<folly::EvictingCacheMap<std::uint64_t, Evaluation>>> m_shards;

    std::atomic<int> m_cache_hits = 0;
    std::atomic<int> m_cache_misses = 0;
};
