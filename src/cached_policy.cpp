#include "cached_policy.hpp"

#include <folly/Hash.h>

std::uint64_t position_hash(Board const& board, Turn turn, TreeNode const*) {
    return folly::hash::hash_128_to_64(board.hash_from_pov(turn.player), turn.action);
}

CachedPolicy::CachedPolicy(std::shared_ptr<MCTSPolicy> policy)
    : m_policy{std::move(policy)} {}

int CachedPolicy::cache_hits() const {
    return m_cache_hits;
}

int CachedPolicy::cache_misses() const {
    return m_cache_misses;
}

folly::coro::Task<MCTSPolicy::Evaluation> CachedPolicy::evaluate_position(Board const& board,
                                                                          Turn turn,
                                                                          TreeNode const* parent) {
    // TODO: exploit symmetry
    auto const hash = position_hash(board, turn, parent);
    auto const it = m_cache.find(hash);

    if (it != m_cache.end()) {
        ++m_cache_hits;
        co_return it->second;
    }

    ++m_cache_misses;
    Evaluation const eval = co_await m_policy->evaluate_position(board, turn, parent);
    m_cache.insert_or_assign(hash, eval);
    co_return eval;
}

void CachedPolicy::snapshot(TreeNode const& current_root) {
    m_policy->snapshot(current_root);
}
