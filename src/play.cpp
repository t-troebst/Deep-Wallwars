#include "play.hpp"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>

#include <algorithm>
#include <random>
#include <ranges>

#include "mcts.hpp"

namespace views = std::ranges::views;

folly::coro::Task<Player> computer_play_single(const Board& board,
                                               std::shared_ptr<MCTSPolicy> policy1,
                                               std::shared_ptr<MCTSPolicy> policy2,
                                               std::uint32_t seed,
                                               ComputerPlayOptions const& opts) {
    MCTS mcts1{policy1, board, {.seed = opts.seed}};
    MCTS mcts2{policy2, board, {.seed = opts.seed}};

    while (true) {
        co_await mcts1.sample(opts.samples);
        Move move = mcts1.commit_to_move(opts.temperature);

        if (mcts1.current_board().winner()) {
            co_return Player::Red;
        }

        mcts2.force_move(move);
        co_await mcts2.sample(opts.samples);

        if (mcts2.current_board().winner()) {
            co_return Player::Blue;
        }
    }
}

folly::coro::Task<double> computer_play(const Board& board, std::shared_ptr<MCTSPolicy> policy1,
                                        std::shared_ptr<MCTSPolicy> policy2, int games,
                                        ComputerPlayOptions const& opts) {
    std::mt19937_64 twister{opts.seed};
    std::uniform_int_distribution<std::uint32_t> dist;

    auto game_tasks = views::iota(0, games) | views::transform([&](int) {
                          return computer_play_single(board, policy1, policy2, dist(twister), opts);
                      });

    auto results = co_await folly::coro::collectAllRange(game_tasks);
    int red_wins = 0;

    for (Player player : results) {
        if (player == Player::Red) {
            ++red_wins;
        }
    }

    co_return double(red_wins) / games;
}
