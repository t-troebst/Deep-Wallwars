#pragma once

#include <folly/MPMCQueue.h>
#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>

#include <memory>
#include <thread>

struct Model;

class BatchedModel {
public:
    struct Output {
        std::vector<float> wall_prior;
        std::vector<float> step_prior;
        float value;
    };

    BatchedModel(std::unique_ptr<Model> model);
    BatchedModel(std::unique_ptr<Model> model, int queue_size);

    ~BatchedModel();

    BatchedModel(BatchedModel const& other) = delete;
    BatchedModel(BatchedModel&& other) = delete;

    BatchedModel& operator=(BatchedModel const& other) = delete;
    BatchedModel& operator=(BatchedModel&& other) = delete;

    folly::SemiFuture<Output> inference(std::vector<float> state);

private:
    struct InferenceTask {
        std::vector<float> state;
        folly::Promise<Output> output;
    };

    std::unique_ptr<Model> m_model;
    folly::MPMCQueue<InferenceTask> m_tasks;
    std::jthread m_worker;

    void run_worker();
};
