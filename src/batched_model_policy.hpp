#pragma once

#include "batched_model.hpp"
#include "mcts.hpp"

class BatchedModelPolicy {
public:
    BatchedModelPolicy(std::shared_ptr<BatchedModel> model);

    folly::coro::Task<Evaluation> operator()(Board const& board, Turn turn);

private:
    std::shared_ptr<BatchedModel> m_model;
};
