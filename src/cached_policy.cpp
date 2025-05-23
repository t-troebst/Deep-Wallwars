#include "cached_policy.hpp"

#include <folly/Hash.h>
#include <folly/Overload.h>
#include <folly/container/EvictingCacheMap.h>
#include <folly/container/HeterogeneousAccess.h>
#include <folly/detail/ThreadLocalDetail.h>

#include <utility>

std::size_t folly::HeterogeneousAccessHash<CacheEntry>::operator()(
    CacheEntry const& cache_entry) const {
    return operator()(
        CacheEntryView{cache_entry.board, cache_entry.turn, cache_entry.previous_position});
}

std::size_t folly::HeterogeneousAccessHash<CacheEntry>::operator()(CacheEntryView ce_view) const {
    return folly::hash::hash_combine(ce_view.board, ce_view.turn.action, ce_view.turn.player,
                                     ce_view.previous_position);
}

bool folly::HeterogeneousAccessEqualTo<CacheEntry>::operator()(CacheEntry const& lhs,
                                                               CacheEntry const& rhs) const {
    return operator()(CacheEntryView{lhs.board, lhs.turn, lhs.previous_position}, rhs);
}

bool folly::HeterogeneousAccessEqualTo<CacheEntry>::operator()(CacheEntryView lhs,
                                                               CacheEntry const& rhs) const {
    if (lhs.turn != rhs.turn) {
        return false;
    }

    if (lhs.turn.player != rhs.turn.player) {
        return false;
    }

    if (lhs.previous_position != rhs.previous_position) {
        return false;
    }

    return lhs.board == rhs.board;
}

bool folly::HeterogeneousAccessEqualTo<CacheEntry>::operator()(CacheEntry const& lhs,
                                                               CacheEntryView rhs) const {
    return operator()(rhs, lhs);
}

CachedPolicy::CachedPolicy(EvaluationFunction evaluate, std::size_t capacity, unsigned shards)
    : m_cache{std::make_shared<EvaluationCache>(std::move(evaluate))} {
    for (unsigned i = 0; i < shards; ++i) {
        m_cache->lrus.emplace_back(std::in_place, capacity / shards);
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

folly::coro::Task<Evaluation> CachedPolicy::operator()(Board const& board, Turn turn,
                                                       std::optional<Cell> previous_position) {
    CacheEntryView ce_view{board, turn, previous_position};

    // TODO: its a bit annoying that we always compute the hash twice
    auto hash = folly::HeterogeneousAccessHash<CacheEntry>{}(ce_view);
    auto& lru = m_cache->lrus[hash % m_cache->lrus.size()];

    {
        auto locked_lru = lru.wlock();
        auto existing_entry = locked_lru->find(CacheEntryView{board, turn, previous_position});
        if (existing_entry != locked_lru->end()) {
            ++m_cache->cache_hits;
            co_return existing_entry->second;
        }
    }

    Evaluation eval = co_await m_cache->evaluate(board, turn, previous_position);
    lru.wlock()->insert(CacheEntry{board, turn, previous_position}, eval);
    ++m_cache->cache_misses;
    co_return eval;
}
