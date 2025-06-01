#include "gamestate.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

TEST_CASE("Legal walls", "[Game State]") {
    Board tiny{2, 2};

    REQUIRE(tiny.legal_walls().size() == 4);
    tiny.place_wall(Player::Blue, {{0, 0}, Direction::Right});
    REQUIRE(tiny.legal_walls().size() == 0);
}

TEST_CASE("Empty board", "[Game State]") {
    Board board{3, 3};

    REQUIRE(board.position(Player::Red) == Cell{0, 0});
    REQUIRE(board.position(Player::Blue) == Cell{2, 0});

    REQUIRE(board.goal(Player::Red) == Cell{2, 2});
    REQUIRE(board.goal(Player::Blue) == Cell{0, 2});

    REQUIRE(board.legal_directions(Player::Red).size() == 2);
    REQUIRE(board.legal_directions(Player::Blue).size() == 2);

    REQUIRE(board.legal_walls().size() == 12);

    REQUIRE(board.legal_actions(Player::Red).size() == 14);
    REQUIRE(board.legal_actions(Player::Blue).size() == 14);

    REQUIRE(board.winner() == Winner::Undecided);
}

TEST_CASE("Big board", "[Game State]") {
    Board board{8, 8};

    board.place_wall(Player::Red, {{1, 3}, Direction::Right});
    board.place_wall(Player::Red, {{4, 4}, Direction::Right});

    board.take_step(Player::Blue, Direction::Down);
    board.take_step(Player::Blue, Direction::Down);

    board.place_wall(Player::Red, {{0, 3}, Direction::Down});
    board.place_wall(Player::Red, {{0, 7}, Direction::Right});

    board.take_step(Player::Blue, Direction::Down);
    board.take_step(Player::Blue, Direction::Down);

    board.place_wall(Player::Red, {{1, 4}, Direction::Down});
    board.place_wall(Player::Red, {{4, 3}, Direction::Right});

    board.take_step(Player::Blue, Direction::Left);
    board.take_step(Player::Blue, Direction::Down);

    board.take_step(Player::Red, Direction::Right);
    board.take_step(Player::Red, Direction::Right);

    board.place_wall(Player::Blue, {{2, 0}, Direction::Right});
    board.place_wall(Player::Blue, {{2, 0}, Direction::Down});

    REQUIRE(board.legal_directions(Player::Red).size() == 1);
}

TEST_CASE("Big board 2", "[Game State]") {
    Board board{8, 8};

    board.do_action(Player::Red, Direction::Right);
    board.do_action(Player::Red, Direction::Down);

    board.do_action(Player::Blue, Wall{{2, 0}, Direction::Down});
    board.do_action(Player::Blue, Wall{{4, 2}, Direction::Down});

    board.do_action(Player::Red, Direction::Right);
    board.do_action(Player::Red, Wall{{1, 1}, Direction::Right});

    board.do_action(Player::Blue, Wall{{2, 0}, Direction::Right});
    board.do_action(Player::Blue, Wall{{1, 3}, Direction::Right});

    board.do_action(Player::Red, Direction::Right);
    board.do_action(Player::Red, Wall{{2, 1}, Direction::Right});

    board.do_action(Player::Blue, Wall{{3, 1}, Direction::Right});
    board.do_action(Player::Blue, Wall{{3, 1}, Direction::Down});

    board.do_action(Player::Red, Wall{{1, 7}, Direction::Right});
    board.do_action(Player::Red, Wall{{1, 6}, Direction::Right});
}

TEST_CASE("Advance to win", "[Game State]") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};

    board.take_step(Player::Red, Direction::Right);
    REQUIRE(board.winner() == Winner::Undecided);
    board.take_step(Player::Red, Direction::Down);
    REQUIRE(board.winner() == Winner::Undecided);
    board.take_step(Player::Red, Direction::Down);
    REQUIRE(board.winner() == Winner::Undecided);
    board.take_step(Player::Red, Direction::Right);
    REQUIRE(board.position(Player::Red) == board.goal(Player::Red));
    REQUIRE(board.winner() == Winner::Red);
}

TEST_CASE("Block walls", "[Game State]") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    board.place_wall(Player::Blue, {{0, 0}, Direction::Right});
    REQUIRE(board.is_blocked({{0, 0}, Direction::Right}));
    REQUIRE(board.legal_directions(Player::Red).size() == 1);
}

TEST_CASE("Distance", "[Game State]") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    REQUIRE(board.distance({0, 0}, {2, 2}) == 4);
    REQUIRE(board.distance({2, 1}, {2, 2}) == 1);
}

TEST_CASE("Can't disconnect players from goals", "[Game State]") {
    Board board{3, 3};
    board.place_wall(Player::Blue, {{0, 0}, Direction::Right});
    REQUIRE(board.legal_walls().size() == 10);
}

TEST_CASE("Removing random legal walls doesn't disconnect players", "[Game State]") {
    std::mt19937_64 twister{Catch::getSeed()};

    for (int i = 0; i < 10; ++i) {
        Board board{5, 5};
        auto legal_walls = board.legal_walls();

        while (!legal_walls.empty()) {
            std::uniform_int_distribution<std::size_t> dist(0, legal_walls.size() - 1);
            board.place_wall(Player::Red, legal_walls[dist(twister)]);
            legal_walls = board.legal_walls();
        }

        CHECK(board.distance(board.position(Player::Red), board.goal(Player::Red)) != -1);
        CHECK(board.distance(board.position(Player::Blue), board.goal(Player::Blue)) != -1);
    }
}

TEST_CASE("Fill relative distances", "[Game State]") {
    Board board{3, 3};

    std::vector<float> dists(9, 1.0f);
    board.fill_relative_distances({0, 0}, dists);

    CHECK(dists[0] == 0.0f);
    CHECK(dists[1] == Catch::Approx(0.111111f));
    CHECK(dists[3] == Catch::Approx(0.111111f));
    CHECK(dists[4] == Catch::Approx(0.222222f));
    CHECK(dists[8] == Catch::Approx(0.444444f));

    board.place_wall(Player::Red, {{0, 0}, Direction::Right});
    board.place_wall(Player::Red, {{0, 0}, Direction::Down});
    board.fill_relative_distances({0, 0}, dists);

    CHECK(dists[0] == 0.0f);
    CHECK(dists[1] == 1.0f);
    CHECK(dists[3] == 1.0f);

    std::vector<std::array<bool, 4>> blocked_dirs = board.blocked_directions();
    std::vector<std::pair<Cell, int>> queue_vec;
    std::fill(dists.begin(), dists.end(), 1.0f);

    board.fill_relative_distances({0, 0}, dists, blocked_dirs, queue_vec);
    CHECK(dists[0] == 0.0f);
    CHECK(dists[1] == 1.0f);
    CHECK(dists[3] == 1.0f);
}

TEST_CASE("Fill relative distances matches distance", "[Game State]") {
    Board board{5, 5};
    std::mt19937_64 twister{Catch::getSeed()};

    for (int i = 0; i < 10; ++i) {
        auto legal_walls = board.legal_walls();
        if (legal_walls.empty())
            break;

        std::uniform_int_distribution<std::size_t> dist(0, legal_walls.size() - 1);
        board.place_wall(Player::Red, legal_walls[dist(twister)]);
    }

    for (int start_row = 0; start_row < 5; start_row += 2) {
        for (int start_col = 0; start_col < 5; start_col += 2) {
            Cell start{start_col, start_row};
            std::vector<float> dists(25, 1.0f);
            board.fill_relative_distances(start, dists);

            for (int row = 0; row < 5; ++row) {
                for (int col = 0; col < 5; ++col) {
                    Cell target{col, row};
                    int actual_dist = board.distance(start, target);
                    if (actual_dist != -1) {
                        float expected = static_cast<float>(actual_dist) / 25.0f;
                        CHECK(dists[board.index_from_cell(target)] == Catch::Approx(expected));
                    } else {
                        CHECK(dists[board.index_from_cell(target)] == 1.0f);
                    }
                }
            }
        }
    }
}
