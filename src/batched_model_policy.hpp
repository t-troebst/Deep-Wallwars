#pragma once

#include "batched_model.hpp"
#include "mcts.hpp"

class BatchedModelPolicy {
public:
    BatchedModelPolicy(std::shared_ptr<BatchedModel> model);

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn,
                                             std::optional<Cell> previous_position);

    // Expose statistics through the policy interface
    uint64_t total_inferences() const {
        return m_model->total_inferences();
    }
    uint64_t total_batches() const {
        return m_model->total_batches();
    }

private:
    std::shared_ptr<BatchedModel> m_model;
};
