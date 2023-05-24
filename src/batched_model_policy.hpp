#pragma once

#include "mcts.hpp"

struct BatchedModel;

class BatchedModelPolicy : MCTSPolicy {
public:
    folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn,
                                                    TreeNode const* parent) override;

    void snapshot(TreeNode const& current_root) override;

private:
    std::shared_ptr<BatchedModel> m_model;
};
