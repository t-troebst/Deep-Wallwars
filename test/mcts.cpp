#include "mcts.hpp"

#include <folly/Executor.h>
#include <folly/executors/GlobalExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Sleep.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "simple_policy.hpp"

// For testing, only generates moves downwards
struct DownPolicy {
    std::shared_ptr<int> samples = std::make_shared<int>(0);

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn, std::optional<Cell>) {
        ++*samples;
        if (board.is_blocked(Wall{board.position(turn.player), Direction::Down})) {
            co_return Evaluation{0, {}};
        }
        co_return Evaluation{0, {TreeEdge(Direction::Down, 1.0)}};
    };
};

TEST_CASE("Basic Initialization", "[MCTS]") {
    Board board{4, 4};
    MCTS mcts{SimplePolicy{1.0, 1.0, 1.0}, std::move(board)};

    CHECK(mcts.root_value() == 0.0);
    CHECK(mcts.root_samples() == 1);
    CHECK(mcts.wasted_inferences() == 0);
}

TEST_CASE("Single sample", "[MCTS]") {
    Board board{4, 4};
    MCTS mcts{SimplePolicy{1.0, 1.0, 1.0}, std::move(board)};

    folly::coro::blockingWait(mcts.sample(1));

    CHECK(mcts.root_value() > 0.0);
    CHECK(mcts.root_samples() == 2);
    CHECK(mcts.wasted_inferences() == 0);
}

TEST_CASE("Commit to action", "[MCTS]") {
    Board board{4, 4};
    MCTS mcts{DownPolicy{}, std::move(board)};

    CHECK_FALSE(mcts.commit_to_action());
    CHECK_FALSE(mcts.commit_to_action(0.2));

    folly::coro::blockingWait(mcts.sample(1));

    auto action = mcts.commit_to_action();
    CHECK(std::get<Direction>(*action) == Direction::Down);
    CHECK(mcts.current_board().position(Player::Red) == Cell{0, 1});
}

TEST_CASE("Force action", "[MCTS]") {
    Board board{4, 4};
    MCTS mcts{DownPolicy{}, std::move(board)};

    SECTION("No previous sample") {}
    SECTION("Previous sample") {
        folly::coro::blockingWait(mcts.sample(1));
    }

    mcts.force_action(Direction::Down);
    CHECK(mcts.root_samples() == 1);
    CHECK(mcts.current_board().position(Player::Red) == Cell{0, 1});
}

TEST_CASE("Sample many", "[MCTS]") {
    Board board{4, 4};
    DownPolicy policy;
    MCTS mcts{policy, std::move(board)};

    folly::coro::blockingWait(mcts.sample(1000));

    CHECK(mcts.wasted_inferences() == 0);
    CHECK(mcts.root_samples() == 1001);
    CHECK(*policy.samples == 6);
}

struct SlowDownPolicy {
    std::shared_ptr<std::atomic<int>> samples = std::make_shared<std::atomic<int>>(0);

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn, std::optional<Cell>) {
        ++*samples;
        co_await folly::coro::sleep(std::chrono::milliseconds{250});
        Evaluation result;

        if (board.is_blocked(Wall{board.position(turn.player), Direction::Down})) {
            result = Evaluation{0, {}};
        } else {
            result = Evaluation{0, {TreeEdge(Direction::Down, 1.0)}};
        }
        co_return result;
    };
};

TEST_CASE("Sample slow in parallel", "[MCTS]") {
    Board board{4, 4};
    SlowDownPolicy policy;
    MCTS mcts{policy, std::move(board), {.max_parallelism = 5}};

    folly::coro::blockingWait(mcts.sample(16));

    CHECK(mcts.wasted_inferences() == 12);
    CHECK(mcts.root_samples() == 17);
    CHECK(*policy.samples > 3);
}
