#include "simple_policy.hpp"

SimplePolicy::SimplePolicy(float move_prior, float good_move_bias, float bad_move_bias)
    : m_move_prior{move_prior}, m_good_move_bias{good_move_bias}, m_bad_move_bias{bad_move_bias} {}

folly::SemiFuture<TreeNode*> SimplePolicy::evaluate_position(Board board, Turn turn,
                                                             TreeNode* parent) {
    std::vector<Wall> legal_walls;
    if (m_move_prior < 1) {
        legal_walls = board.legal_walls();
    }

    std::vector<TreeEdge> edges;
    edges.reserve(legal_walls.size() + 4);

    Cell const pos = board.position(turn.player);
    Cell const goal = board.goal(turn.player);
    int const dist = board.distance(pos, goal);
    float total_prior = 0;

    for (Direction dir : board.legal_directions(turn.player)) {
        int const new_dist = board.distance(pos.step(dir), goal);
        float prior = 1;

        if (new_dist < dist) {
            prior = m_good_move_bias;
        } else if (new_dist > dist) {
            prior = m_bad_move_bias;
        }

        if (prior > 0) {
            edges.emplace_back(dir, prior);
            total_prior += prior;
        }
    }

    for (TreeEdge& te : edges) {
        te.prior *= m_move_prior / total_prior;
    }


    float wall_prior = (1 - m_move_prior) / legal_walls.size();
    for (Wall wall : legal_walls) {
        edges.emplace_back(wall, wall_prior);
    }

    TreeNode* result = new TreeNode{parent,
                                    std::move(board),
                                    turn,
                                    parent ? parent->depth + 1 : 0,
                                    TreeNode::Value(board.score_for(turn.player), 1),
                                    std::move(edges)};

    return result;
}
