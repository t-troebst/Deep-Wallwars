#include <iostream>

#include "simple_policy.hpp"
#include "mcts.hpp"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <gflags/gflags.h>
#include <folly/logging/xlog.h>

#include "play.hpp"

DEFINE_int32(columns, 5, "Number of columns");
DEFINE_int32(rows, 5, "Number of rows");

DEFINE_int32(games, 100, "Number of games to play");
DEFINE_int32(j, 8, "Number of threads");

DEFINE_double(move_prior_1, 0.3, "Move prior of agent 1");
DEFINE_double(good_move_1, 1.5, "Good move bias of agent 1");
DEFINE_double(bad_move_1, 0.75, "Bad move bias of agent 1");

DEFINE_double(move_prior_2, 0.3, "Move prior of agent 2");
DEFINE_double(good_move_2, 1.5, "Good move bias of agent 2");
DEFINE_double(bad_move_2, 0.75, "Bad move bias of agent 2");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto sp1 = std::make_shared<SimplePolicy>(FLAGS_move_prior_1, FLAGS_good_move_1, FLAGS_bad_move_1);
    auto sp2 = std::make_shared<SimplePolicy>(FLAGS_move_prior_2, FLAGS_good_move_2, FLAGS_bad_move_2);

    Board board{FLAGS_columns, FLAGS_rows};

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(computer_play(board, sp1, sp2, FLAGS_games).scheduleOn(&thread_pool));
}
