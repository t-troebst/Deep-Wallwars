#pragma once

#include <folly/experimental/coro/Task.h>

#include <cstdint>
#include <memory>

#include "gamestate.hpp"

struct MCTSPolicy;

struct ComputerPlayOptions {
    int threads = 4;
    int samples = 1000;
    int max_parallel_games = 64;
    int move_limit = 100;
    double temperature = 0.2;
    std::uint32_t seed = 42;
};

folly::coro::Task<double> computer_play(Board board, std::shared_ptr<MCTSPolicy> policy1,
                                        std::shared_ptr<MCTSPolicy> policy2, int games,
                                        ComputerPlayOptions opts = {});
