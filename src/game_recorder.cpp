#include "game_recorder.hpp"

#include <sstream>

GameRecorder::GameRecorder(Board initial_board, std::string red_name, std::string blue_name)
    : m_red_name{red_name}, m_blue_name{blue_name}, m_board_states{initial_board} {}

void GameRecorder::record_move(Player player, Move move) {
    Board last_board = m_board_states.back();
    last_board.do_action(player, move.first);
    last_board.do_action(player, move.second);
    m_board_states.push_back(std::move(last_board));
    m_moves.push_back(move);
}

void GameRecorder::record_winner(Winner winner) {
    m_outcome = winner;
}

std::string GameRecorder::to_json() const {
    std::stringstream result;

    result << "{\"creator\": \"" << m_red_name << "\", \"joiner\": \"" << m_blue_name
           << "\", \"rows\": " << m_board_states.front().rows()
           << ", \"columns\": " << m_board_states.front().columns() << ", \"moves\": \"";

    Player player = Player::Red;

    for (std::size_t i = 0; i < m_moves.size(); ++i) {
        result << i + 1 << ". " << m_moves[i].standard_notation(m_board_states[i].position(player));
        player = other_player(player);

        if (i + 1 < m_moves.size()) {
            result << " ";
        }
    }

    result << "\"}";
    return result.str();
}
