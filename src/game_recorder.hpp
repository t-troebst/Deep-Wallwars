#pragma once

#include <string>
#include <vector>

#include "gamestate.hpp"

class GameRecorder {
public:
    GameRecorder(Board initial_board, std::string red_name = "Red", std::string blue_name = "Blue");

    void record_move(Player player, Move move);
    void record_winner(Winner winner);

    std::string to_json() const;

private:
    std::string m_red_name;
    std::string m_blue_name;
    std::vector<Board> m_board_states;
    std::vector<Move> m_moves;
    Winner m_outcome;
};
