#include "play.hpp"

#include <folly/Executor.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <ranges>

#include "mcts.hpp"

namespace views = std::ranges::views;

struct GameResult {
    std::optional<Player> winner;
    int wasted_inferences;
};

folly::coro::Task<GameResult> computer_play_single(const Board& board, EvaluationFunction evaluate1,
                                                   EvaluationFunction evaluate2,
                                                   std::uint32_t index,
                                                   ComputerPlayOptions const& opts) {
    MCTS mcts1{evaluate1, board, {.max_parallelism = opts.max_parallel_samples, .seed = index}};
    MCTS mcts2{evaluate2, board, {.max_parallelism = opts.max_parallel_samples, .seed = index}};

    XLOGF(INFO, "Starting game {}.", index);

    for (int num_moves = 1; opts.move_limit == 0 || num_moves <= opts.move_limit; ++num_moves) {
        for (int i = 0; i < 2; ++i) {
            co_await mcts1.sample(opts.samples);
            Action action = mcts1.commit_to_action(opts.temperature);
            mcts2.force_action(action);

            if (mcts1.current_board().winner()) {
                XLOGF(INFO, "Red player won game {} in {} moves.", index, num_moves);
                opts.on_complete(mcts1);
                opts.on_complete(mcts2);
                co_return {Player::Red, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }
        }

        for (int i = 0; i < 2; ++i) {
            co_await mcts2.sample(opts.samples);
            Action action = mcts2.commit_to_action(opts.temperature);
            mcts1.force_action(action);

            if (mcts2.current_board().winner()) {
                XLOGF(INFO, "Blue player won game {} in {} moves.", index, num_moves);
                opts.on_complete(mcts1);
                opts.on_complete(mcts2);
                co_return {Player::Blue, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }
        }
    }

    XLOGF(INFO, "Game {} was ended because it hit the move limit of {}", index, opts.move_limit);
    opts.on_complete(mcts1);
    opts.on_complete(mcts2);
    co_return {{}, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
}

folly::coro::Task<double> computer_play(Board board, EvaluationFunction evaluate1,
                                        EvaluationFunction evaluate2, int games,
                                        ComputerPlayOptions opts) {
    folly::Executor* executor = co_await folly::coro::co_current_executor;

    auto game_tasks =
        views::iota(1, games + 1) | views::transform([&](int i) {
            return computer_play_single(board, evaluate1, evaluate2, i, opts).scheduleOn(executor);
        });

    auto results = co_await folly::coro::collectAllWindowed(game_tasks, opts.max_parallel_games);
    int red_wins = 0;
    int timeouts = 0;
    int wasted_inferences = 0;

    for (GameResult result : results) {
        if (!result.winner) {
            ++timeouts;
        } else if (result.winner == Player::Red) {
            ++red_wins;
        }

        wasted_inferences += result.wasted_inferences;
    }

    XLOGF(INFO, "{}/{} games have finished and red won {} of them.", games - timeouts, games,
          red_wins);
    XLOGF(INFO, "{} inferences were wasted.", wasted_inferences);

    co_return double(red_wins) / games;
}
