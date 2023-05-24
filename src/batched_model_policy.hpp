#pragma once

#include <iosfwd>
#include <mutex>

#include "batched_model.hpp"
#include "mcts.hpp"

class BatchedModelPolicy : MCTSPolicy {
public:
    folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn,
                                                    TreeNode const* parent) override;

    void snapshot(TreeNode const& current_root) override;

private:
    std::shared_ptr<BatchedModel> m_model;
    std::shared_ptr<std::ostream> m_snapshot_stream;
    std::mutex m_snapshot_mutex;

    std::vector<float> convert_to_state(Board const& bord, Turn turn) const;
    BatchedModel::Output convert_to_output(TreeNode const& node) const;
};
