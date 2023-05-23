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
    float prior;  // make everything float because of space for atomics
    std::atomic<int> active_samples;
    std::atomic<TreeNode*> child;
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

    void add_sample(float weight) {
        TreeNode::Value old_val = value;
        TreeNode::Value new_val;

        do {
            new_val = {old_val.total_weight + weight, old_val.total_samples + 1};
        } while (!value.compare_exchange_weak(old_val, new_val));
    }
};

struct MCTSPolicy {
    virtual folly::SemiFuture<TreeNode*> evaluate_position(Board board, Turn turn,
                                                           TreeNode* parent) = 0;

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
        double direchlet_alpha = 0.2;
        Turn starting_turn = {Player::Red, Turn::First};
        std::uint32_t seed = 42;
    };

    MCTS(std::shared_ptr<MCTSPolicy> policy, Board board);
    MCTS(std::shared_ptr<MCTSPolicy> policy, Board board, Options opts);

    Board const& current_board() const;
    float root_value() const;
    int root_samples() const;

    folly::coro::Task<double> sample(int iterations);

    Action commit_to_action();
    Action commit_to_action(double temperature);

    Move commit_to_move();
    Move commit_to_move(double temperature);

    void force_action(Action const& action);
    void force_move(Move const& move);

private:
    std::shared_ptr<MCTSPolicy> m_policy;
    TreeNode* m_root;
    TreeNode* m_current_root;
    Options m_opts;
    std::gamma_distribution<> m_gamma_dist;
    std::mt19937_64 m_twister;
    std::atomic<int> m_wasted_inferences = 0;

    void add_root_noise();
    folly::coro::Task<> sample_worker(std::atomic<int>& remaining_iters);
    folly::coro::Task<float> sample_rec(TreeNode& root);
};
