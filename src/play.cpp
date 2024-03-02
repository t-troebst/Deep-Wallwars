#include "play.hpp"

#include <folly/Executor.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <iostream>
#include <ranges>

#include "mcts.hpp"

namespace views = std::ranges::views;

struct GameResult {
    Winner winner;
    int wasted_inferences;
};

folly::coro::Task<> human_play(Board board, EvaluationFunction model,
                               HumanPlayOptions const& opts) {
    std::cout << "Do you want to go first? (y/n): ";
    char first;
    std::cin >> first;

    bool human_goes_first = first == 'y';
    auto human_player = human_goes_first ? Player::Red : Player::Blue;
    auto ai_player = other_player(human_player);

    GameRecorder recorder(board, human_goes_first ? "Player" : "Deep Wallwars",
                          human_goes_first ? "Deep Wallwars" : "Player");
    MCTS mcts{
        model, std::move(board), {.max_parallelism = opts.max_parallel_samples, .seed = opts.seed}};

    bool take_turn = !human_goes_first;
    while (true) {
        if (take_turn) {
            Cell ai_cell = mcts.current_board().position(ai_player);
            auto ai_move = co_await mcts.sample_and_commit_to_move(opts.samples);
            if (ai_move) {
                recorder.record_move(ai_player, *ai_move);
            } else {
                recorder.record_winner(winner_from_player(human_player));
                break;
            }

            std::cout << ai_move->standard_notation(ai_cell) << "\n";
            if (auto winner = mcts.current_board().winner(); winner != Winner::Undecided) {
                recorder.record_winner(winner);
                break;
            }
        } else {
            take_turn = true;
        }

        // TODO: read in standard notation instead of this ad hoc solution
        std::array<Action, 2> actions;
        for (int i = 0; i < 2; ++i) {
            std::string action_type;
            std::cin >> action_type;

            if (action_type == "step") {
                Direction dir;
                std::cin >> dir;
                actions[i] = dir;
            } else if (action_type == "wall") {
                Wall wall;
                std::cin >> wall;
                actions[i] = wall;
            }
        }

        Move human_move{actions[0], actions[1]};
        mcts.force_move(human_move);
        recorder.record_move(human_player, human_move);

        if (auto winner = mcts.current_board().winner(); winner != Winner::Undecided) {
            recorder.record_winner(winner);
            break;
        }
    }

    XLOGF(INFO, "Game finished, json string: {}", recorder.to_json());
}

folly::coro::Task<GameResult> computer_play_single(const Board& board, EvaluationFunction evaluate1,
                                                   EvaluationFunction evaluate2, int index,
                                                   ComputerPlayOptions const& opts) {
    MCTS mcts1{
        evaluate1,
        board,
        {.max_parallelism = opts.max_parallel_samples, .seed = static_cast<std::uint32_t>(index)}};

    MCTS mcts2{
        evaluate2,
        board,
        {.max_parallelism = opts.max_parallel_samples, .seed = static_cast<std::uint32_t>(index)}};

    XLOGF(INFO, "Starting game {}.", index);

    for (int num_moves = 1; opts.move_limit == 0 || num_moves <= opts.move_limit; ++num_moves) {
        for (int i = 0; i < 2; ++i) {
            co_await mcts1.sample(opts.samples);
            auto action = mcts1.commit_to_action(opts.temperature);
            if (!action) {
                XLOGF(INFO, "Blue player won game {} in {} moves.", index, num_moves);
                opts.on_complete(mcts1, index);
                opts.on_complete(mcts2, index);
                co_return {Winner::Blue, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }

            mcts2.force_action(*action);

            if (Winner winner = mcts1.current_board().winner(); winner != Winner::Undecided) {
                if (winner == Winner::Red) {
                    XLOGF(INFO, "Red player won game {} in {} moves.", index, num_moves);
                } else {
                    XLOGF(INFO, "Red player drew game {} in {} moves.", index, num_moves);
                }

                opts.on_complete(mcts1, index);
                opts.on_complete(mcts2, index);
                co_return {winner, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }
        }

        for (int i = 0; i < 2; ++i) {
            co_await mcts2.sample(opts.samples);
            auto action = mcts2.commit_to_action(opts.temperature);
            if (!action) {
                XLOGF(INFO, "Red player won game {} in {} moves.", index, num_moves);
                opts.on_complete(mcts1, index);
                opts.on_complete(mcts2, index);
                co_return {Winner::Blue, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }
            mcts1.force_action(*action);

            if (Winner winner = mcts2.current_board().winner(); winner != Winner::Undecided) {
                XLOGF(INFO, "Blue player won game {} in {} moves.", index, num_moves);
                opts.on_complete(mcts1, index);
                opts.on_complete(mcts2, index);
                co_return {winner, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }
        }
    }

    XLOGF(INFO, "Game {} was ended because it hit the move limit of {}", index, opts.move_limit);
    opts.on_complete(mcts1, index);
    opts.on_complete(mcts2, index);
    co_return {Winner::Undecided, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
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
    int draws = 0;
    int blue_wins = 0;
    int wasted_inferences = 0;

    for (GameResult result : results) {
        if (result.winner == Winner::Red) {
            ++red_wins;
        } else if (result.winner == Winner::Draw) {
            ++draws;
        } else if (result.winner == Winner::Blue) {
            ++blue_wins;
        }

        wasted_inferences += result.wasted_inferences;
    }

    int total_games = red_wins + draws + blue_wins;

    XLOGF(INFO, "Red's W/L/D statistic over {}/{} games is: {} / {} / {}", total_games, games,
          red_wins, blue_wins, draws);
    XLOGF(INFO, "{} inferences were wasted.", wasted_inferences);

    co_return double(red_wins) / games;
}
