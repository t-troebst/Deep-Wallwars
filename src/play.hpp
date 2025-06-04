#pragma once

#include <folly/experimental/coro/Task.h>

#include <cstdint>
#include <filesystem>

#include "game_recorder.hpp"
#include "gamestate.hpp"
#include "mcts.hpp"

// Called once for each game with the winner after the MCTS has finished. Can be used to output
// training data.
using CompletionCallback = std::function<void(MCTS const&, int)>;

struct NamedModel {
    EvaluationFunction model;
    std::string name;
};

struct InteractivePlayOptions {
    EvaluationFunction model;

    int samples = 1000;
    int max_parallel_samples = 256;

    std::uint32_t seed = 42;
};

struct TrainingPlayOptions {
    EvaluationFunction model1;
    EvaluationFunction model2;

    int samples = 1000;
    int max_parallel_games = 128;
    int max_parallel_samples = 16;
    int move_limit = 100;
    double temperature = 1;

    CompletionCallback on_complete = [](MCTS const&, int) {};

    std::uint32_t seed = 42;
};

struct EvaluationPlayOptions {
    NamedModel model1;
    NamedModel model2;

    int samples = 1000;
    int max_parallel_games = 128;
    int max_parallel_samples = 16;
    int move_limit = 100;

    std::uint32_t seed = 42;
};

struct RankingPlayOptions {
    std::vector<NamedModel> models;
    std::filesystem::path output_folder;

    int samples = 1000;
    int games_per_matchup = 10;
    int num_tournaments = 10;
    int max_parallel_games = 128;
    int max_parallel_samples = 32;
    int move_limit = 100;

    std::uint32_t seed = 42;
};

folly::coro::Task<GameRecorder> interactive_play(Board board, InteractivePlayOptions opts);

folly::coro::Task<> training_play(Board board, int games, TrainingPlayOptions opts);

folly::coro::Task<std::vector<GameRecorder>> evaluation_play(Board board, int games,
                                                             EvaluationPlayOptions opts);

// Generates random tournaments between the models to generate ranking games for bayeselo.
folly::coro::Task<std::vector<GameRecorder>> ranking_play(Board board, RankingPlayOptions opts);