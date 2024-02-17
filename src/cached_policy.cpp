#include "cached_policy.hpp"

#include <folly/Hash.h>
#include <folly/Overload.h>

std::uint64_t position_hash(Board const& board, Turn turn, bool flip_horizontal) {
    return folly::hash::hash_128_to_64(board.hash_from_pov(turn.player, flip_horizontal),
                                       turn.action);
}

CachedPolicy::CachedPolicy(EvaluationFunction evaluate, std::size_t capacity, std::size_t shards)
    : m_cache{std::make_shared<EvaluationCache>(std::move(evaluate))} {
    // Can't figure out how to do this in the constructor initializer list...
    for (std::size_t i = 0; i < shards; ++i) {
        m_cache->shards.emplace_back(
            folly::in_place_t(),
            folly::EvictingCacheMap<std::uint64_t, Evaluation>(capacity / shards));
    }
}

int CachedPolicy::cache_hits() const {
    return m_cache->cache_hits;
}

int CachedPolicy::cache_misses() const {
    return m_cache->cache_misses;
}

void flip_evaluation(Board const& board, Evaluation& eval) {
    for (TreeEdge& edge : eval.edges) {
        folly::variant_match(
            edge.action, [&](Direction dir) { edge.action = flip_horizontal(dir); },
            [&](Wall wall) { edge.action = board.flip_horizontal(wall); });
    }
}

folly::coro::Task<Evaluation> CachedPolicy::operator()(Board const& board, Turn turn) {
    auto const hash = position_hash(board, turn, false);
    auto& lru = m_cache->shards[hash % m_cache->shards.size()];

    {
        auto lock = lru.wlock();
        if (auto const eval_it = lock->find(hash); eval_it != lock->end()) {
            ++m_cache->cache_hits;
            co_return eval_it->second;
        }
    }

    auto const flipped_hash = position_hash(board, turn, true);
    auto& flipped_lru = m_cache->shards[flipped_hash % m_cache->shards.size()];
    {
        auto lock = flipped_lru.wlock();
        if (auto const eval_it = lock->find(flipped_hash); eval_it != lock->end()) {
            ++m_cache->cache_hits;
            Evaluation eval = eval_it->second;
            flip_evaluation(board, eval);
            co_return eval;
        }
    }

    ++m_cache->cache_misses;
    Evaluation eval = co_await m_cache->evaluate(board, turn);
    lru.wlock()->insert(hash, eval);
    co_return eval;
}
