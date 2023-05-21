#pragma once

#include "mcts.hpp"
#include <array>

struct SimplePolicy : MCTSPolicy {
    enum class Quality {
        Good,
        Bad,
        Neutral
    };

    double move_probability;
    double good_move_bias;
    double bad_move_bias;

    std::array<Quality, 4> directions;
    int legal_walls;
    double total_move_prior;

    SimplePolicy(double move_probability, double good_move_bias, double bad_move_bias);

    void evaluate(TreeNode const& node, Turn turn) override;

    double step_prior(Direction dir) const override;
    double wall_prior(Wall wall) const override;

    std::optional<double> value() const override;

    std::unique_ptr<MCTSPolicy> clone() const override;
};
