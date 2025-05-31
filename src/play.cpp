#include "play.hpp"

#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <folly/logging/xlog.h>

#include <iostream>
#include <ranges>

#include "game_recorder.hpp"
#include "mcts.hpp"

namespace views = std::ranges::views;

struct GameResult {
    Winner winner;
    int wasted_inferences;
};

folly::coro::Task<GameRecorder> interactive_play(Board board, InteractivePlayOptions opts) {
    std::cout << "Do you want to go first? (y/n): ";
    char first;
    std::cin >> first;

    bool human_goes_first = first == 'y';
    auto human_player = human_goes_first ? Player::Red : Player::Blue;
    auto ai_player = other_player(human_player);

    GameRecorder recorder(board, human_goes_first ? "Player" : "Deep Wallwars",
                          human_goes_first ? "Deep Wallwars" : "Player");
    MCTS mcts{opts.model,
              std::move(board),
              {.max_parallelism = opts.max_parallel_samples, .seed = opts.seed}};

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

        XLOGF(INFO, "AI thinks your position is worth {}", mcts.root_value());

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
    co_return recorder;
}

folly::coro::Task<GameResult> training_play_single(Board const& board, EvaluationFunction evaluate1,
                                                   EvaluationFunction evaluate2, int index,
                                                   TrainingPlayOptions opts) {
    MCTS mcts1{evaluate1,
               board,
               {.max_parallelism = opts.max_parallel_samples,
                .seed = opts.seed * static_cast<std::uint32_t>(index)}};

    MCTS mcts2{evaluate2,
               board,
               {.max_parallelism = opts.max_parallel_samples,
                .seed = opts.seed * static_cast<std::uint32_t>(index)}};

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
                    XLOGF(INFO, "Red player won game {} in {} moves.", index, 2 * num_moves - 1);
                } else {
                    XLOGF(INFO, "Red player drew game {} in {} moves.", index, 2 * num_moves - 1);
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
                XLOGF(INFO, "Red player won game {} in {} moves.", index, 2 * num_moves);
                opts.on_complete(mcts1, index);
                opts.on_complete(mcts2, index);
                co_return {Winner::Blue, mcts1.wasted_inferences() + mcts2.wasted_inferences()};
            }
            mcts1.force_action(*action);

            if (Winner winner = mcts2.current_board().winner(); winner != Winner::Undecided) {
                XLOGF(INFO, "Blue player won game {} in {} moves.", index, 2 * num_moves);
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

folly::coro::Task<GameRecorder> evaluation_play_single(Board const& board, int index,
                                                       EvaluationPlayOptions opts) {
    auto const& red = index % 2 == 0 ? opts.model1 : opts.model2;
    auto const& blue = index % 2 == 0 ? opts.model2 : opts.model1;

    MCTS mcts1{red.model,
               board,
               {.max_parallelism = opts.max_parallel_samples,
                .seed = opts.seed * static_cast<std::uint32_t>(index)}};

    MCTS mcts2{blue.model,
               board,
               {.max_parallelism = opts.max_parallel_samples,
                .seed = opts.seed * static_cast<std::uint32_t>(index)}};

    XLOGF(INFO, "Starting game {} with {} as red and {} as blue.", index, red.name, blue.name);
    GameRecorder recorder(board, red.name, blue.name);

    for (int num_moves = 1; opts.move_limit == 0 || num_moves <= opts.move_limit; ++num_moves) {
        auto move1 = co_await mcts1.sample_and_commit_to_move(opts.samples);
        if (!move1) {
            recorder.record_winner(Winner::Blue);
            break;
        }

        recorder.record_move(Player::Red, *move1);
        if (auto winner = mcts1.current_board().winner(); winner != Winner::Undecided) {
            recorder.record_winner(winner);
            break;
        }

        mcts2.force_move(*move1);
        auto move2 = co_await mcts2.sample_and_commit_to_move(opts.samples);

        if (!move2) {
            recorder.record_winner(Winner::Red);
            break;
        }

        recorder.record_move(Player::Blue, *move2);
        if (auto winner = mcts2.current_board().winner(); winner != Winner::Undecided) {
            recorder.record_winner(winner);
            break;
        }

        mcts1.force_move(*move2);
    }

    XLOGF(INFO, "Game {} has concluded.", index);
    co_return recorder;
}

folly::coro::Task<std::vector<GameRecorder>> evaluation_play(Board board, int games,
                                                             EvaluationPlayOptions opts) {
    auto* executor = co_await folly::coro::co_current_executor;
    auto game_tasks = views::iota(1, games + 1) | views::transform([&](int i) {
                          return evaluation_play_single(board, i, opts).scheduleOn(executor);
                      });

    auto results = co_await folly::coro::collectAllWindowed(game_tasks, opts.max_parallel_games);

    co_return results;
}

folly::coro::Task<> training_play(Board board, int games, TrainingPlayOptions opts) {
    auto* executor = co_await folly::coro::co_current_executor;
    auto game_tasks = views::iota(1, games + 1) | views::transform([&](int i) {
                          return training_play_single(board, opts.model1, opts.model2, i, opts)
                              .scheduleOn(executor);
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
}

folly::coro::Task<std::vector<GameRecorder>> ranking_play(Board board, int games_per_matchup,
                                                          RankingPlayOptions opts) {
    std::vector<folly::coro::Task<GameRecorder>> game_tasks;
    std::vector<GameRecorder> recorders;

    int game_index = 1;
    std::size_t start_model =
        opts.models_to_rank == 0 ? 0 : opts.models.size() - opts.models_to_rank;
    auto* executor = co_await folly::coro::co_current_executor;
    for (std::size_t i = start_model; i < opts.models.size(); ++i) {
        std::size_t rank_start_model = std::max(static_cast<int>(i) - opts.max_matchup_distance, 0);

        for (std::size_t j = rank_start_model; j < i; ++j) {
            EvaluationPlayOptions eval_opts{.model1 = opts.models[i],
                                            .model2 = opts.models[j],

                                            .samples = opts.samples,
                                            .max_parallel_samples = opts.max_parallel_samples,
                                            .move_limit = opts.move_limit,
                                            .seed = opts.seed};
            auto game_tasks =
                views::iota(0, games_per_matchup) | views::transform([&, game_index](int i) {
                    return evaluation_play_single(board, game_index + i, eval_opts)
                        .scheduleOn(executor);
                });
            game_index += games_per_matchup;
            auto matchup_recorders = co_await folly::coro::collectAllWindowed(
                std::move(game_tasks), opts.max_parallel_games);
            recorders.insert(recorders.end(), matchup_recorders.begin(), matchup_recorders.end());
        }
    }

    co_return recorders;
}
