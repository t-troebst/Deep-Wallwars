#pragma once

#include <iosfwd>
#include <mutex>

#include "batched_model.hpp"
#include "mcts.hpp"

class BatchedModelPolicy : public MCTSPolicy {
public:
    BatchedModelPolicy(std::shared_ptr<BatchedModel> model,
                       std::shared_ptr<std::ostream> snapshot_stream = nullptr);

    folly::coro::Task<Evaluation> evaluate_position(Board const& board, Turn turn) override;

    void snapshot(NodeInfo const& node_info, std::optional<Player> winner) override;

private:
    std::shared_ptr<BatchedModel> m_model;
    std::shared_ptr<std::ostream> m_snapshot_stream;
    std::mutex m_snapshot_mutex;

    std::vector<float> convert_to_state(Board const& bord, Turn turn) const;
    BatchedModel::Output convert_to_output(NodeInfo const& node_info,
                                           std::optional<Player> player) const;
};
