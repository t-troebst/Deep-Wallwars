#pragma once

#include <folly/MPMCQueue.h>
#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>

#include <memory>
#include <thread>
#include <vector>

#include "state_conversions.hpp"

class Model;

class BatchedModel {
public:
    BatchedModel(std::unique_ptr<Model> model);
    BatchedModel(std::unique_ptr<Model> model, int queue_size);
    BatchedModel(std::vector<std::unique_ptr<Model>> models);
    BatchedModel(std::vector<std::unique_ptr<Model>> models, int queue_size);

    ~BatchedModel();

    BatchedModel(BatchedModel const& other) = delete;
    BatchedModel(BatchedModel&& other) = delete;

    BatchedModel& operator=(BatchedModel const& other) = delete;
    BatchedModel& operator=(BatchedModel&& other) = delete;

    folly::SemiFuture<ModelOutput> inference(std::vector<float> state);

    std::size_t total_inferences() const;
    std::size_t total_batches() const;

private:
    struct InferenceTask {
        std::vector<float> state;
        folly::Promise<ModelOutput> output;
    };

    folly::MPMCQueue<InferenceTask> m_tasks;
    std::vector<std::unique_ptr<Model>> m_models;

    std::atomic<std::size_t> m_batches = 0;
    std::atomic<std::size_t> m_inferences = 0;

    // Worker threads need to come last so everything else is still alive while we join them.
    std::vector<std::jthread> m_workers;

    void run_worker(std::size_t idx);
};
