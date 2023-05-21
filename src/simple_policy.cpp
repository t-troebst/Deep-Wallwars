#include "simple_policy.hpp"
#include <exception>

SimplePolicy::SimplePolicy(double move_probability, double good_move_bias, double bad_move_bias)
    : move_probability{move_probability},
      good_move_bias{good_move_bias},
      bad_move_bias{bad_move_bias} {}

void SimplePolicy::evaluate(TreeNode const& node, Turn turn) {
    Cell const current_pos = node.board.position(turn.player);
    int const current_dist = node.board.distance(current_pos, node.board.goal(turn.player));
    total_move_prior = 0.0;

    for (Direction dir : node.board.legal_directions(turn.player)) {
        Cell const new_pos = current_pos.step(dir);

        int const new_dist = node.board.distance(new_pos, node.board.goal(turn.player));

        if (new_dist < current_dist) {
            directions[int(dir)] = Quality::Good;
            total_move_prior += good_move_bias;
        } else if (new_dist > current_dist) {
            directions[int(dir)] = Quality::Bad;
            total_move_prior += bad_move_bias;
        } else {
            directions[int(dir)] = Quality::Neutral;
            total_move_prior += 1;
        }
    }

    total_move_prior /= move_probability;
    legal_walls = node.board.legal_walls().size();
}

double SimplePolicy::step_prior(Direction dir) const {
    if (directions[int(dir)] == Quality::Good) {
        return good_move_bias / total_move_prior;
    }

    if (directions[int(dir)] == Quality::Bad) {
        return bad_move_bias / total_move_prior;
    }

    return total_move_prior;
}

double SimplePolicy::wall_prior(Wall) const {
    return (1 - move_probability) / legal_walls;
}

std::optional<double> SimplePolicy::value() const {
    return {};
}

std::unique_ptr<MCTSPolicy> SimplePolicy::clone() const {
    return std::make_unique<SimplePolicy>(*this);
}
