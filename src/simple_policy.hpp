#pragma once

#include "mcts.hpp"

class SimplePolicy {
public:
    SimplePolicy(float move_prior, float good_move_bias, float bad_move_bias);

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn,
                                             std::optional<Cell> previous_position);

private:
    float m_move_prior;
    float m_good_move_bias;
    float m_bad_move_bias;
};
