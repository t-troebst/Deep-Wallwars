#pragma once

#include <string>
#include <vector>

#include "gamestate.hpp"

class GameRecorder {
public:
    GameRecorder(Board initial_board, std::string red_name = "Red", std::string blue_name = "Blue");

    void record_move(Player player, Move move);
    void record_winner(Winner winner);

    // This is for wallwars.net.
    std::string to_json() const;

    // This is for bayeselo. PGN is a chess format and we do not really try to implement it
    // correctly, this is just the minimal output that is parsed by bayeselo.
    std::string to_pgn() const;

private:
    std::string m_red_name;
    std::string m_blue_name;
    std::vector<Board> m_board_states;
    std::vector<Move> m_moves;
    Winner m_outcome;
};

std::string all_to_json(std::vector<GameRecorder> const& recorders);
std::string all_to_pgn(std::vector<GameRecorder> const& recorders);
