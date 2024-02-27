#pragma once

#include <folly/experimental/coro/Task.h>

#include <cstdint>

#include "gamestate.hpp"
#include "mcts.hpp"

// Called once for each game with the winner after the MCTS has finished. Can be used to output
// training data.
using CompletionCallback = std::function<void(MCTS const&, int)>;

struct HumanPlayOptions {
    int threads = 4;
    int samples = 1000;
    int max_parallel_samples = 16;
    std::uint32_t seed = 42;
};

struct ComputerPlayOptions {
    int threads = 4;
    int samples = 1000;
    int max_parallel_games = 128;
    int max_parallel_samples = 16;
    int move_limit = 100;
    double temperature = 0.2;
    std::uint32_t seed = 42;
    CompletionCallback on_complete = [](MCTS const&, int) {};
};

folly::coro::Task<> human_play(Board board, EvaluationFunction model,
                               HumanPlayOptions const& opts = {});

folly::coro::Task<double> computer_play(Board board, EvaluationFunction evaluate1,
                                        EvaluationFunction evaluate2, int games,
                                        ComputerPlayOptions opts = {});
