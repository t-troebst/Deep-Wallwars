#include "gamestate.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <ranges>
#include <sstream>

TEST_CASE("Empty board") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};

    REQUIRE(board.position(Player::Red) == Cell{0, 0});
    REQUIRE(board.position(Player::Blue) == Cell{2, 0});

    REQUIRE(board.goal(Player::Red) == Cell{2, 2});
    REQUIRE(board.goal(Player::Blue) == Cell{0, 2});

    REQUIRE(board.legal_directions(Player::Red).size() == 2);
    REQUIRE(board.legal_directions(Player::Blue).size() == 2);

    REQUIRE(board.legal_walls().size() == 12);

    REQUIRE(board.legal_moves(Player::Red).size() == 14);
    REQUIRE(board.legal_moves(Player::Blue).size() == 14);

    REQUIRE_FALSE(board.winner());
}

TEST_CASE("Advance to win") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};

    board.take_step(Player::Red, Direction::Right);
    REQUIRE_FALSE(board.winner());
    board.take_step(Player::Red, Direction::Up);
    REQUIRE_FALSE(board.winner());
    board.take_step(Player::Red, Direction::Up);
    REQUIRE_FALSE(board.winner());
    board.take_step(Player::Red, Direction::Right);
    REQUIRE(board.position(Player::Red) == board.goal(Player::Red));
    REQUIRE(board.winner());
    REQUIRE(board.winner().value() == Player::Red);
}

TEST_CASE("Block walls") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    board.place_wall(Player::Blue, {{0, 0}, Direction::Right});
    REQUIRE(board.is_blocked({{0, 0}, Direction::Right}));
    REQUIRE(board.legal_directions(Player::Red).size() == 1);
}

TEST_CASE("Distance") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    REQUIRE(board.distance({0, 0}, {2, 2}) == 4);
    REQUIRE(board.distance({2, 1}, {2, 2}) == 1);
}

TEST_CASE("Can't disconnect players from goals") {
    Board board{3, 3, {0, 0}, {2, 2}, {2, 0}, {0, 2}};
    board.place_wall(Player::Blue, {{0, 0}, Direction::Right});
    REQUIRE(board.legal_walls().size() == 10);
    // board.advance(Player::Red, Place{Wall{0, 3}});
    // REQUIRE(board.legal_walls().size() == 7);
    // board.advance(Player::Red, Place{Wall{1, 0}});
    // REQUIRE(board.legal_walls().size() == 5);
}

TEST_CASE("Input / Output") {
    Direction dir;
    std::stringstream dir_str{"Right"};
    dir_str >> dir;
    REQUIRE(dir == Direction::Right);

    Wall wall;
    std::stringstream wall_str{"{ (3, 4), Right }"};
    wall_str >> wall;
    REQUIRE(wall.cell == Cell{3, 4});
    REQUIRE(wall.direction == Direction::Right);
}
