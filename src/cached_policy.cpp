#include "cached_policy.hpp"

#include <folly/Hash.h>
#include <folly/Overload.h>

std::uint64_t position_hash(Board const& board, Turn turn, TreeNode const*, bool flip_horizontal) {
    return folly::hash::hash_128_to_64(board.hash_from_pov(turn.player, flip_horizontal),
                                       turn.action);
}

CachedPolicy::CachedPolicy(std::shared_ptr<MCTSPolicy> policy) : m_policy{std::move(policy)} {}

int CachedPolicy::cache_hits() const {
    return m_cache_hits;
}

int CachedPolicy::cache_misses() const {
    return m_cache_misses;
}

void flip_evaluation(Board const& board, MCTSPolicy::Evaluation& eval) {
    for (TreeEdge& edge : eval.edges) {
        folly::variant_match(
            edge.action, [&](Direction dir) { edge.action = flip_horizontal(dir); },
            [&](Wall wall) { edge.action = board.flip_horizontal(wall); });
    }
}

folly::coro::Task<MCTSPolicy::Evaluation> CachedPolicy::evaluate_position(Board const& board,
                                                                          Turn turn,
                                                                          TreeNode const* parent) {
    auto const hash = position_hash(board, turn, parent, false);
    auto const it = m_cache.find(hash);

    if (it != m_cache.end()) {
        ++m_cache_hits;
        co_return it->second;
    }

    auto const flipped_hash = position_hash(board, turn, parent, true);
    auto const flipped_it = m_cache.find(flipped_hash);

    if (flipped_it != m_cache.end()) {
        ++m_cache_hits;
        Evaluation flipped = flipped_it->second;
        flip_evaluation(board, flipped);
        co_return flipped;
    }

    ++m_cache_misses;
    Evaluation const eval = co_await m_policy->evaluate_position(board, turn, parent);
    m_cache.insert_or_assign(hash, eval);
    co_return eval;
}

void CachedPolicy::snapshot(TreeNode const& current_root) {
    m_policy->snapshot(current_root);
}
