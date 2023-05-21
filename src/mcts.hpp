#pragma once

#include <compare>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <utility>

#include "gamestate.hpp"

struct TreeNode;

struct NodeInfo {
    double cumulative_weight;
    int num_samples;
    double prior;

    Move move;
    mutable std::unique_ptr<TreeNode> node;
};

struct TreeNode {
    Board board;
    int total_samples;
    TreeNode const* parent;  // useful if you want the board history
    std::vector<NodeInfo> children;
};

struct MCTSPolicy {
    virtual void evaluate(TreeNode const& node, Turn turn) = 0;

    virtual double step_prior(Direction dir) const = 0;
    virtual double wall_prior(Wall wall) const = 0;

    virtual int depth_limit() const;

    double prior(Move move) const;

    virtual std::optional<double> value() const = 0;

    virtual std::unique_ptr<MCTSPolicy> clone() const = 0;

    virtual ~MCTSPolicy() = default;
};

class MCTS {
public:
    MCTS(Board board, Turn turn, std::unique_ptr<MCTSPolicy> policy);

    Board const& current_board() const;
    Turn current_turn() const;
    TreeNode const& current_node() const;

    double sample();
    double sample(int count);

    void force_move(Move move);

    Move commit_to_move();
    Move commit_to_move(std::mt19937_64& twister, double temperature);

private:
    std::unique_ptr<MCTSPolicy> policy;

    std::unique_ptr<TreeNode> root;
    TreeNode* current_root;
    Turn turn;

    std::pair<std::unique_ptr<TreeNode>, std::optional<double>> create_node(Board&& board,
                                                                            TreeNode* parent,
                                                                            Turn local_turn) const;

    double sample_rec(TreeNode& local_root, Turn local_turn, int depth);
};
