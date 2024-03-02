#pragma once

#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>

#include <atomic>
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

struct Evaluation {
    float value;
    std::vector<TreeEdge> edges;
};

// Coroutine that takes the current board and player turn and "evaluates" it, either by some
// heuristic or ML model. The last argument is the previous position of the current player, which is
// needed because we may not return to that position in the same move.
using EvaluationFunction =
    std::function<folly::coro::Task<Evaluation>(Board const&, Turn, std::optional<Cell>)>;

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

    MCTS(EvaluationFunction evaluate, Board board);
    MCTS(EvaluationFunction evaluate, Board board, Options opts);

    Board const& current_board() const;
    float root_value() const;
    int root_samples() const;
    NodeInfo root_info() const;
    std::vector<NodeInfo> const& history() const;
    int wasted_inferences() const;

    folly::coro::Task<float> sample(int iterations);

    // Selects the best action from the perspective of the current player and commits to it.
    // In rare cases there may be no valid action at all (either because the EvaluationFunction is
    // arbitrarily restricting the set of possible actions or because our previous action ran us
    // into a dead-end).
    std::optional<Action> commit_to_action();
    std::optional<Action> commit_to_action(float temperature);

    // Selects the move (two actions) from the perspective of the current player and commits to it.
    // If the first action wins the game, the second action will place a wall arbitrarily.
    folly::coro::Task<std::optional<Move>> sample_and_commit_to_move(int iterations);

    void force_action(Action const& action);
    void force_move(Move const& move);

    ~MCTS();

private:
    EvaluationFunction m_evaluate;
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

    folly::coro::Task<TreeNode*> create_tree_node(Board board, Turn turn,
                                                  std::optional<Cell> previous_position,
                                                  TreeNode* parent);
};
