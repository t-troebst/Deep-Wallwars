#pragma once

#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>

#include <atomic>
#include <memory>
#include <random>

#include "gamestate.hpp"

struct TreeNode;

struct TreeEdge {
    Action action;
    float prior;
    std::atomic<int> active_samples = 0;
    std::atomic<TreeNode*> child = nullptr;

    TreeEdge() = default;
    TreeEdge(Action action, float prior);

    TreeEdge(TreeEdge const& other);
    TreeEdge& operator=(TreeEdge const& other);
};

struct TreeNode {
    struct Value {
        float total_weight;
        int total_samples;
    };

    TreeNode* parent;
    Board board;
    Turn turn;
    int depth;
    std::atomic<Value> value;
    std::vector<TreeEdge> edges;

    void add_sample(float weight);
    ~TreeNode();
};

struct MCTSPolicy {
    struct Evaluation {
        float value;
        std::vector<TreeEdge> edges;
    };

    virtual folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn,
                                                            TreeNode const* parent) = 0;
    virtual void snapshot(TreeNode const& current_root);

    MCTSPolicy() = default;

    MCTSPolicy(MCTSPolicy const& other) = delete;
    MCTSPolicy(MCTSPolicy&& other) = delete;

    MCTSPolicy& operator=(MCTSPolicy const& other) = delete;
    MCTSPolicy& operator=(MCTSPolicy&& other) = delete;

    virtual ~MCTSPolicy() = default;
};

class MCTS {
public:
    struct Options {
        float puct = 3.0;
        int max_depth = 50;
        int max_parallelism = 4;
        float direchlet_alpha = 0.2;
        float active_sample_penalty = 1.0;
        Turn starting_turn = {Player::Red, Turn::First};
        std::uint32_t seed = 42;
    };

    MCTS(std::shared_ptr<MCTSPolicy> policy, Board board);
    MCTS(std::shared_ptr<MCTSPolicy> policy, Board board, Options opts);

    Board const& current_board() const;
    float root_value() const;
    int root_samples() const;
    int wasted_inferences() const;

    folly::coro::Task<float> sample(int iterations);

    Action commit_to_action();
    Action commit_to_action(float temperature);

    Move commit_to_move();
    Move commit_to_move(float temperature);

    void force_action(Action const& action);
    void force_move(Move const& move);

private:
    std::shared_ptr<MCTSPolicy> m_policy;
    std::unique_ptr<TreeNode> m_root;
    TreeNode* m_current_root;
    Options m_opts;
    std::gamma_distribution<float> m_gamma_dist;
    std::mt19937_64 m_twister;
    std::atomic<int> m_wasted_inferences = 0;

    void add_root_noise();
    folly::coro::Task<float> sample_rec(TreeNode& root);

    folly::coro::Task<TreeNode*> create_tree_node(Board board, Turn turn, TreeNode* parent);
};
