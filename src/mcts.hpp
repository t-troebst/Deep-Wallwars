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
};

struct EdgeInfo {
    Action action;
    int num_samples;
};

struct NodeInfo {
    Board board;
    Turn turn;
    float q_value;
    int num_samples;

    std::vector<EdgeInfo> edges;
};

struct MCTSPolicy {
    struct Evaluation {
        float value;
        std::vector<TreeEdge> edges;
    };

    virtual folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn) = 0;
    virtual void snapshot(NodeInfo const& current_root, std::optional<Player> winner);

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
        float noise_factor = 0.25;
        float active_sample_penalty = 1.0;
        Turn starting_turn = {Player::Red, Turn::First};
        std::uint32_t seed = 42;
    };

    MCTS(std::shared_ptr<MCTSPolicy> policy, Board board);
    MCTS(std::shared_ptr<MCTSPolicy> policy, Board board, Options opts);

    Board const& current_board() const;
    float root_value() const;
    int root_samples() const;
    NodeInfo root_info() const;
    std::vector<NodeInfo> const& history() const;
    int wasted_inferences() const;

    folly::coro::Task<float> sample(int iterations);

    Action commit_to_action();
    Action commit_to_action(float temperature);

    Move commit_to_move();
    Move commit_to_move(float temperature);

    void force_action(Action const& action);
    void force_move(Move const& move);

    void snapshot(std::optional<Player> winner);

    ~MCTS();

private:
    std::shared_ptr<MCTSPolicy> m_policy;
    TreeNode* m_root;
    Options m_opts;
    std::gamma_distribution<float> m_gamma_dist;
    std::mt19937_64 m_twister;
    std::atomic<int> m_wasted_inferences = 0;
    std::vector<NodeInfo> m_history;

    void add_root_noise();
    folly::coro::Task<void> single_sample();
    TreeEdge& get_best_edge(TreeNode& current) const;
    folly::coro::Task<float> initialize_child(TreeNode& current, TreeEdge& edge);
    folly::coro::Task<float> sample_rec(TreeNode& current);
    void delete_subtree(TreeNode& node);
    void move_root(TreeEdge const& edge);

    folly::coro::Task<TreeNode*> create_tree_node(Board board, Turn turn, TreeNode* parent);
};
