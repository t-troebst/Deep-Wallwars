#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <ranges>

#include "util.hpp"

namespace ranges = std::ranges;
namespace views = std::ranges::views;

int MCTSPolicy::depth_limit() const {
    return 50;
}

double MCTSPolicy::prior(Move move) const {
    return std::visit(overload{[&](Direction dir) { return step_prior(dir); },
                               [&](Wall wall) { return wall_prior(wall); }},
                      move);
}

MCTS::MCTS(Board board, Turn turn, std::unique_ptr<MCTSPolicy> policy)
    : policy{std::move(policy)}, root{nullptr}, turn{turn} {
    root = create_node(std::move(board), nullptr, turn).first;
    current_root = root.get();
}

Board const& MCTS::current_board() const {
    return current_root->board;
}

Turn MCTS::current_turn() const {
    return turn;
}

TreeNode const& MCTS::current_node() const {
    return *current_root;
}

double MCTS::sample() {
    return sample_rec(*current_root, turn, 0);
}

double MCTS::sample(int count) {
    double total = 0.0;

    for (int i = 0; i < count; ++i) {
        total += sample();
    }

    return total / count;
}

void MCTS::force_move(Move move) {
    auto const child_it = ranges::find_if(current_root->children,
                                          [&](NodeInfo const& ni) { return ni.move == move; });

    if (child_it == current_root->children.end()) {
        throw std::runtime_error("Could not find move - not legal?");
    }

    if (!child_it->node) {
        Board board = current_root->board;
        board.do_move(turn.player, move);
        child_it->node = create_node(std::move(board), current_root, turn.next()).first;
    }

    current_root = child_it->node.get();
    turn = turn.next();
}

Move MCTS::commit_to_move() {
    int max_samples = 0;
    auto max_it = current_root->children.begin();

    for (auto it = current_root->children.begin(); it != current_root->children.end(); ++it) {
        if (it->num_samples > max_samples) {
            max_samples = it->num_samples;
            max_it = it;
        }
    }

    Move const move = max_it->move;
    current_root = max_it->node.get();
    turn = turn.next();

    return move;
}

Move MCTS::commit_to_move(std::mt19937_64& twister, double temperature) {
    auto const weights = views::transform(current_root->children, [=](NodeInfo const& ni) {
        return ni.num_samples ? std::pow(ni.num_samples, 1.0 / temperature) : 0;
    });

    std::discrete_distribution<std::size_t> weight_dist(weights.begin(), weights.end());
    auto const it = std::next(current_root->children.begin(), weight_dist(twister));

    Move const move = it->move;
    current_root = it->node.get();
    turn = turn.next();

    return move;
}

std::pair<std::unique_ptr<TreeNode>, std::optional<double>> MCTS::create_node(
    Board&& board, TreeNode* parent, Turn local_turn) const {
    std::unique_ptr<TreeNode> node = std::make_unique<TreeNode>(std::move(board), 0, parent);

    policy->evaluate(*node, local_turn);
    std::vector<Move> legal_moves = node->board.legal_moves(local_turn.player);
    double total_prior = 0.0;

    for (Move move : legal_moves) {
        total_prior += policy->prior(move);
    }

    for (Move move : legal_moves) {
        node->children.push_back(
            NodeInfo{0.0, 0, policy->prior(move) / total_prior, move, nullptr});
    }

    return {std::move(node), policy->value()};
}

double MCTS::sample_rec(TreeNode& local_root, Turn local_turn, int depth) {
    Board const& board = local_root.board;

    if (auto winner = board.winner(); winner) {
        return *winner == local_turn.player ? 1 : -1;
    }

    if (depth >= policy->depth_limit()) {
        return 0.0;  // TODO: or -1?
    }

    if (local_root.children.empty()) {
        throw std::runtime_error("No legal moves!");
    }

    local_root.total_samples += 1;

    NodeInfo& ni =
        *ranges::max_element(local_root.children, ranges::less(), [&](NodeInfo const& ni) {
            double prior = &local_root == current_root ? ni.prior + 0.05 : ni.prior;
            return ni.cumulative_weight / std::max(ni.num_samples, 1) +
                   prior * std::sqrt(local_root.total_samples) / (1 + ni.num_samples);
        });

    Turn const next_turn = local_turn.next();

    if (!ni.node) {
        Board new_board = board;
        new_board.do_move(local_turn.player, ni.move);
        auto [new_node, value] = create_node(std::move(new_board), &local_root, next_turn);
        ni.node = std::move(new_node);

        if (value) {
            ni.num_samples += 1;
            ni.cumulative_weight += *value;
            return *value;
        }
    } else {
        // Depth limit only applies to new parts of the tree.
        depth = 0;
    }

    double value = sample_rec(*ni.node, next_turn, depth + 1);
    if (next_turn.player != local_turn.player) {
        value *= -1;
    }

    ni.num_samples += 1;
    ni.cumulative_weight += value;

    return value;
}
