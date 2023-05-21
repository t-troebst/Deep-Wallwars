#include "mcts.hpp"
#include "simple_policy.hpp"
#include <memory>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Check that sample terminates") {
    std::unique_ptr<SimplePolicy> policy = std::make_unique<SimplePolicy>(0.5, 2, 0.5);
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    MCTS mcts{board, {Player::Red, Turn::First}, std::move(policy)};

    REQUIRE(mcts.sample() != 0);
}

TEST_CASE("Check that full run terminates") {
    std::unique_ptr<SimplePolicy> policy = std::make_unique<SimplePolicy>(0.5, 2, 0.5);
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    MCTS mcts{board, {Player::Red, Turn::First}, std::move(policy)};

    REQUIRE(mcts.sample(1600) != 0);
}

TEST_CASE("Check trivial self-play") {
    std::unique_ptr<SimplePolicy> policy_1 = std::make_unique<SimplePolicy>(0.5, 2, 0.5);
    std::unique_ptr<SimplePolicy> policy_2 = std::make_unique<SimplePolicy>(0.5, 2, 0.5);
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};

    MCTS mcts_1{board, {Player::Red, Turn::First}, std::move(policy_1)};
    MCTS mcts_2{board, {Player::Red, Turn::First}, std::move(policy_2)};

    mcts_1.sample(100);
    Move m1 = mcts_1.commit_to_move();
    mcts_1.sample(100);
    Move m2 = mcts_1.commit_to_move();
    mcts_2.force_move(m1);
    mcts_2.force_move(m2);

    REQUIRE(mcts_1.current_turn() == mcts_2.current_turn());
}
