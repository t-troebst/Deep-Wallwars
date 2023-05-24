#include "play.hpp"

#include <folly/Executor.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <algorithm>
#include <random>
#include <ranges>

#include "mcts.hpp"

namespace views = std::ranges::views;

folly::coro::Task<Player> computer_play_single(const Board& board,
                                               std::shared_ptr<MCTSPolicy> policy1,
                                               std::shared_ptr<MCTSPolicy> policy2,
                                               std::uint32_t index,
                                               ComputerPlayOptions const& opts) {
    MCTS mcts1{policy1, board, {.seed = index}};
    MCTS mcts2{policy2, board, {.seed = index}};

    XLOGF(INFO, "Starting game {}.", index);
    int num_moves = 0;

    while (true) {
        for (int i = 0; i < 2; ++i) {
            co_await mcts1.sample(opts.samples);
            Action action = mcts1.commit_to_action(opts.temperature);

            if (mcts1.current_board().winner()) {
                XLOGF(INFO, "Red player won game {}.", index);
                co_return Player::Red;
            }

            mcts2.force_action(action);
        }

        for (int i = 0; i < 2; ++i) {
            co_await mcts2.sample(opts.samples);
            Action action = mcts2.commit_to_action(opts.temperature);

            if (mcts2.current_board().winner()) {
                XLOGF(INFO, "Blue player won game {}.", index);
                co_return Player::Blue;
            }
            mcts1.force_action(action);
        }

        num_moves += 2;

        XLOGF(INFO, "Game {} hit {} moves with {} samples the roots.", index, num_moves,
              mcts1.root_samples() + mcts2.root_samples());
    }
}

folly::coro::Task<double> computer_play(Board board, std::shared_ptr<MCTSPolicy> policy1,
                                        std::shared_ptr<MCTSPolicy> policy2, int games,
                                        ComputerPlayOptions opts) {
    folly::Executor* executor = co_await folly::coro::co_current_executor;

    auto game_tasks =
        views::iota(1, games + 1) | views::transform([&](int i) {
            return computer_play_single(board, policy1, policy2, i, opts).scheduleOn(executor);
        });

    auto results = co_await folly::coro::collectAllRange(game_tasks);
    int red_wins = 0;

    for (Player player : results) {
        if (player == Player::Red) {
            ++red_wins;
        }
    }

    XLOGF(INFO, "{} games have finished and red won {} of them.", games, red_wins);

    co_return double(red_wins) / games;
}
